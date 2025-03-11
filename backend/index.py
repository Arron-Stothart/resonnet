import json
import os
import pickle
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib

DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
COLLECTION_NAME = "claude_conversations"

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
    
    # Add minimum length and content checks
    if len(text) < 10:  # Filter out very short texts
        return ""
        
    # Filter out texts that are just punctuation or single words
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
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            last_sentence = max(text.rfind('.', start, end),
                              text.rfind('!', start, end),
                              text.rfind('?', start, end))
            if last_sentence > start + chunk_size - 100:
                end = last_sentence + 1
            else:
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks

def extract_messages(conversations: List[Dict[str, Any]], 
                     indexed_msg_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Extract and prepare messages for indexing from all conversations."""
    all_messages = []
    indexed_msg_ids = indexed_msg_ids or set()
    
    for conv in tqdm(conversations, desc="Processing conversations"):
        conv_id = conv.get("uuid", "unknown")
        conv_name = conv.get("name", "Untitled conversation")
        
        for idx, msg in enumerate(conv.get("chat_messages", [])):
            msg_id = msg.get("uuid", f"msg_{idx}")
            
            if msg_id in indexed_msg_ids:
                continue
                
            if not should_index_message(msg):
                continue
            
            text = ""
            if isinstance(msg.get("content"), dict) and msg["content"].get("text"):
                text = msg["content"]["text"]
            elif msg.get("text"):
                text = msg["text"]
            
            if not text:
                continue
            
            cleaned_text = clean_text(text)
            
            chunks = chunk_text(cleaned_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{msg_id}_chunk_{chunk_idx}" if len(chunks) > 1 else msg_id
                
                message_record = {
                    "conversation_id": conv_id,
                    "conversation_name": conv_name,
                    "message_id": chunk_id,
                    "parent_message_id": msg_id,  
                    "sender": msg.get("sender", "unknown"),
                    "timestamp": msg.get("created_at", ""),
                    "message_index": idx,
                    "chunk_index": chunk_idx if len(chunks) > 1 else 0,
                    "total_chunks": len(chunks),
                    "text": chunk,
                    "original_text": text
                }
                
                all_messages.append(message_record)
    
    print(f"Extracted {len(all_messages)} chunks for indexing")
    return all_messages

def generate_embeddings(messages: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, List]:
    """Generate embeddings for messages"""
    model = SentenceTransformer('Snowflake/snowflake-arctic-embed-xs')
    
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
