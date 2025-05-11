import json
import os
import pickle
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
COLLECTION_NAME = "claude_conversations"
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model_singleton = None

def get_model():
    """Get or create the sentence transformer model singleton."""
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = SentenceTransformer(DENSE_EMBEDDING_MODEL)
    return _model_singleton

def setup_directories():
    """Create necessary directories for data storage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def setup_chroma_db() -> chromadb.PersistentClient:
    """Initialize and return a ChromaDB client."""
    setup_directories()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the ChromaDB collection for conversations."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection not found: {e}")
        print(f"Creating new collection: {COLLECTION_NAME}")
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Claude.ai conversation history"}
        )
    
    return collection

def clean_text(text: str) -> str:
    """Clean message text by removing code blocks and markdown formatting."""
    text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK]', text)
    
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) < 10: 
        return ""
        
    if not re.search(r'[a-zA-Z]{2,}.*[a-zA-Z]{2,}', text):
        return ""
    
    return text

def should_index_message(message: Dict[str, Any]) -> bool:
    """Determine if a message should be indexed based on content and sender."""
    text = ""
    if message.get("text"):
        text = message["text"]
    elif isinstance(message.get("content"), dict) and message["content"].get("text"):
        text = message["content"]["text"]
    
    if not text or message.get("sender") == "system":
        return False
        
    cleaned = clean_text(text)
    return bool(cleaned) 

def get_checkpoint_path(file_path: str) -> str:
    """Generate a checkpoint file path based on the input file."""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:10]
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{file_hash}.pkl")

def save_checkpoint(checkpoint_data: Dict[str, Any], file_path: str) -> None:
    """Save checkpoint data to disk."""
    checkpoint_path = get_checkpoint_path(file_path)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(file_path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint data if it exists."""
    checkpoint_path = get_checkpoint_path(file_path)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return None

def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse the conversations.json export file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} conversations from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks, preserving word boundaries and sentences."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
            
        # Try to break at sentence boundary first
        sentence_break = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end)
        )
        
        # If we found a sentence break within reasonable range, use it
        if sentence_break > start + (chunk_size // 2):
            end = sentence_break + 1
        else:
            # Fall back to word boundary
            space_break = text.rfind(' ', start, end)
            if space_break > start:
                end = space_break
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position for next chunk, accounting for overlap
        start = max(start + chunk_size - overlap, end - overlap)
    
    return chunks

def extract_messages(conversations: List[Dict[str, Any]], 
                     indexed_msg_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Extract and prepare messages for indexing from all conversations."""
    all_messages = []
    indexed_msg_ids = indexed_msg_ids or set()
    
    for conv in tqdm(conversations, desc="Processing conversations"):
        conv_id = conv.get("uuid", "unknown")
        conv_name = conv.get("name", "Untitled conversation")
        
        messages = conv.get("chat_messages", [])
        turns = []
        current_turn = []
        
        for msg in messages:
            if msg.get("sender") == "human" and current_turn:
                turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        
        if current_turn:
            turns.append(current_turn)
            
        for turn_idx, turn in enumerate(turns):
            turn_id = f"{conv_id}_turn_{turn_idx}"
            
            if turn_id in indexed_msg_ids:
                continue
            
            turn_text = ""
            for msg in turn:
                if not should_index_message(msg):
                    continue
                    
                text = ""
                if isinstance(msg.get("content"), dict) and msg["content"].get("text"):
                    text = msg["content"]["text"]
                elif msg.get("text"):
                    text = msg["text"]
                
                if text:
                    cleaned = clean_text(text)
                    if cleaned:
                        turn_text += f"{msg.get('sender', 'unknown')}: {cleaned}\n"
            
            if not turn_text:
                continue
                
            chunks = chunk_text(turn_text, chunk_size=768, overlap=100)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{turn_id}_chunk_{chunk_idx}" if len(chunks) > 1 else turn_id
                
                message_record = {
                    "conversation_id": conv_id,
                    "conversation_name": conv_name,
                    "parent_message_id": turn_id,
                    "turn_index": turn_idx,
                    "chunk_index": chunk_idx if len(chunks) > 1 else 0,
                    "total_chunks": len(chunks),
                    "text": chunk,
                    "original_text": turn_text,
                    "message_id": chunk_id,
                    "timestamp": turn[0].get("timestamp", ""),
                    "message_index": turn_idx
                }
                
                all_messages.append(message_record)
    
    print(f"Extracted {len(all_messages)} chunks for indexing")
    return all_messages

def generate_embeddings(messages: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, List]:
    """Generate embeddings for messages"""
    model = get_model()  # Use the singleton model
    
    ids = []
    texts = []
    metadatas = []
    
    for msg in messages:
        ids.append(msg["message_id"])
        texts.append(msg["text"])
        
        metadata = {k: v for k, v in msg.items() if k != "text" and k != "original_text"}
        metadatas.append(metadata)
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings.tolist())
    
    return {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": texts
    }
    
def index_conversations(file_path: str, batch_size: int = 1000, checkpoint_interval: int = 5000, progress_callback=None) -> None:
    """Main function to index conversations from a JSON file into ChromaDB with checkpointing."""
    setup_directories()
    
    checkpoint = load_checkpoint(file_path)
    indexed_msg_ids = set(checkpoint["indexed_msg_ids"]) if checkpoint else set()
    
    conversations = load_conversations(file_path)
    if not conversations:
        print("No conversations to index. Exiting.")
        return
    
    messages = extract_messages(conversations, indexed_msg_ids)
    if not messages:
        print("No new messages to index. Exiting.")
        return

    if progress_callback:
        progress_callback(0.0, len(messages), 0)
    
    client = setup_chroma_db()
    collection = get_or_create_collection(client)
    
    total_indexed = 0
    for chunk_start in range(0, len(messages), checkpoint_interval):
        chunk_end = min(chunk_start + checkpoint_interval, len(messages))
        chunk_messages = messages[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//checkpoint_interval + 1}: messages {chunk_start} to {chunk_end-1}")
        
        data = generate_embeddings(chunk_messages)
        
        total_in_chunk = len(data["ids"])
        for i in tqdm(range(0, total_in_chunk, batch_size), desc="Indexing in ChromaDB"):
            end_idx = min(i + batch_size, total_in_chunk)
            
            batch_data = {
                "ids": data["ids"][i:end_idx],
                "embeddings": data["embeddings"][i:end_idx],
                "metadatas": data["metadatas"][i:end_idx],
                "documents": data["documents"][i:end_idx]
            }
            
            collection.add(**batch_data)
        
        indexed_msg_ids.update(data["ids"])
        total_indexed += total_in_chunk
        
        checkpoint_data = {
            "file_path": file_path,
            "indexed_msg_ids": list(indexed_msg_ids),
            "total_indexed": total_indexed,
            "last_processed": chunk_end
        }
        save_checkpoint(checkpoint_data, file_path)
        
        print(f"Checkpoint saved after processing {total_indexed} messages")

        if progress_callback:
            progress = total_indexed / len(messages) if messages else 1.0
            progress_callback(progress, len(messages), total_indexed)
    
    print(f"Successfully indexed {total_indexed} messages into ChromaDB")
    print(f"Collection stats: {collection.count()} total documents")

    if progress_callback:
        progress_callback(1.0, len(messages), total_indexed)

# if __name__ == "__main__":
#     file_path = "conversations.json"
#     print(f"Loading conversations from {file_path}")
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             all_conversations = json.load(f)
            
#         sample_conversations = all_conversations[:5]
#         print(f"\nProcessing sample of {len(sample_conversations)} conversations")
        
#         messages = extract_messages(sample_conversations)
        
#         print("\nChunking Statistics:")
#         print(f"Total chunks generated: {len(messages)}")
        
#         chunks_by_conv = {}
#         for msg in messages:
#             conv_id = msg["conversation_id"]
#             if conv_id not in chunks_by_conv:
#                 chunks_by_conv[conv_id] = []
#             chunks_by_conv[conv_id].append(msg)
        
#         for conv_id, chunks in chunks_by_conv.items():
#             print(f"\nConversation: {chunks[0]['conversation_name']} ({conv_id})")
#             print(f"Number of turns: {max(c['turn_index'] for c in chunks) + 1}")
#             print(f"Number of chunks: {len(chunks)}")
            
#             sample_chunk = chunks[0]
#             print("\nSample chunk:")
#             print(f"Turn index: {sample_chunk['turn_index']}")
#             print(f"Chunk index: {sample_chunk['chunk_index']} of {sample_chunk['total_chunks']}")
#             print("Content:")
#             print(sample_chunk['text'])
            
#     except FileNotFoundError:
#         print(f"Error: {file_path} not found. Please ensure the conversations.json file is in the root directory.")
#     except json.JSONDecodeError:
#         print(f"Error: {file_path} is not valid JSON.")
#     except Exception as e:
#         print(f"Error: {str(e)}")

def extract_entities(text, tagger):
    """
    Extract named entities from input text using Flair.
    Returns a list of dictionaries containing entity information.
    """
    # Create a Flair sentence
    sentence = Sentence(text)
    
    # Run NER model
    tagger.predict(sentence)
    
    # Extract entities
    entities = []
    for entity in sentence.get_spans('ner'):
        entities.append({
            'text': entity.text,
            'entity_type': entity.tag,
            'score': entity.score,
            'start_pos': entity.start_position,
            'end_pos': entity.end_position
        })
    
    return entities

if __name__ == "__main__":
    # Load Flair NER model - using the standard English model
    # You can also use 'ner-fast' for faster but less accurate predictions
    tagger = SequenceTagger.load('ner')
    
    # Test cases representing different types of text you might encounter
    test_cases = [
        "I had a discussion with Claude about BERT and GPT models last week.",
        "The project uses React and Node.js for the frontend development.",
        "We implemented CRISPR techniques at Harvard Medical School in Boston.",
        "During my vacation in Paris, I learned about European history.",
        "The TikTok marketing campaign reached 1 million views on AWS servers."
    ]
    
    print("Testing Flair NER extraction on various text types:\n")
    
    for text in test_cases:
        print(f"Input text: {text}")
        entities = extract_entities(text, tagger)
        print("Extracted entities:")
        if entities:
            for entity in entities:
                print(f"- {entity['text']} ({entity['entity_type']}) [confidence: {entity['score']:.3f}]")
        else:
            print("- No entities detected")
        print()
    
    # Example of how this could be used to enhance embeddings
    print("Example of enhancing a query for vector search:")
    query = "How to implement BERT for sentiment analysis?"
    print(f"Original query: {query}")
    
    entities = extract_entities(query, tagger)
    enhanced_terms = [entity['text'] for entity in entities]
    
    print(f"Extracted entities: {enhanced_terms}")
    print(f"Enhanced query could weight these terms more heavily in vector search")
