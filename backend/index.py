import json
import os
import pickle
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib
import spacy
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
COLLECTION_NAME = "claude_conversations"
TARGET_CHUNK_SIZE = 256  
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Markers that indicate topic transitions
TRANSITION_MARKERS = [
    r"^#+\s+.+$",
    r"^\d+\.\s+.+$",
    
    r"\b(additionally|furthermore|moreover|besides|also|next|then|subsequently)\b",
    r"\b(on another note|regarding|as for|speaking of|turning to|moving on to)\b",
    r"\b(let's discuss|let's talk about|let's move on to|let's consider)\b",
    
    r"\b(in summary|to summarize|to conclude|in conclusion|finally|lastly|to wrap up)\b",
    r"\b(overall|all in all|in the end|ultimately|to sum up)\b"
]

TRANSITION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TRANSITION_MARKERS]

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
    """Clean message text by removing code blocks and markdown formatting.
    TODO: Replace
    """
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

def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.
    TODO: Replace
    """
    return len(text) // 4

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def has_discourse_marker(sentence: str) -> bool:
    """Check if a sentence contains a discourse marker indicating topic transition."""
    for pattern in TRANSITION_PATTERNS:
        if pattern.search(sentence):
            return True
    return False

def get_sentence_embeddings(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for a list of sentences."""
    return model.encode(sentences)

def calculate_semantic_shifts(embeddings: np.ndarray) -> List[float]:
    """Calculate semantic shifts between adjacent sentences based on cosine similarity."""
    if len(embeddings) <= 1:
        return []
    
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(1.0 - sim) 
    
    return similarities

def detect_topic_boundaries(text: str, min_segment_tokens: int = 50, 
                           similarity_threshold: float = 0.25) -> List[int]:
    """
    Detect topic boundaries in text using a hybrid approach:
    1. Sentence-level embedding analysis
    2. Discourse marker detection
    3. Paragraph structure analysis
    
    Returns a list of sentence indices where boundaries occur.
    """
    model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
    
    sentences = split_into_sentences(text)
    paragraphs = split_into_paragraphs(text)
    
    if len(sentences) <= 1:
        return []
    
    # 1. Sentence-Level Embedding Analysis
    embeddings = get_sentence_embeddings(sentences, model)
    semantic_shifts = calculate_semantic_shifts(embeddings)
    
    mean_shift = np.mean(semantic_shifts)
    std_shift = np.std(semantic_shifts)
    dynamic_threshold = min(similarity_threshold, mean_shift + std_shift)
    
    potential_boundaries = []
    for i, shift in enumerate(semantic_shifts):
        if shift > dynamic_threshold:
            potential_boundaries.append(i + 1)
    
    # 2. Discourse Marker Detection
    discourse_boundaries = []
    for i, sentence in enumerate(sentences):
        if i > 0 and has_discourse_marker(sentence):
            discourse_boundaries.append(i)
    
    # 3. Paragraph Structure Analysis
    paragraph_boundaries = []
    sentence_to_paragraph = {}
    
    current_para = 0
    sentence_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence_to_paragraph[i] = current_para
        sentence_count += 1
        
        current_text = " ".join(sentences[:i+1])
        if current_para < len(paragraphs) and current_text.endswith(paragraphs[current_para]):
            current_para += 1
            paragraph_boundaries.append(i + 1)
    
    # 4. Boundary Verification & Refinement
    boundary_scores = {i: 0.0 for i in range(1, len(sentences))}
    
    for i in range(1, len(sentences)):
        if i in potential_boundaries:
            boundary_scores[i] += 1.0
        if i > 0:
            boundary_scores[i] += semantic_shifts[i-1] / max(dynamic_threshold, 0.01)
    
    for i in discourse_boundaries:
        boundary_scores[i] += 1.5
    
    for i in paragraph_boundaries:
        if i < len(sentences):
            boundary_scores[i] += 1.0
    
    final_boundaries = []
    last_boundary = 0
    
    sorted_boundaries = sorted([(score, idx) for idx, score in boundary_scores.items()], 
                              reverse=True)
    
    for score, idx in sorted_boundaries:
        if score >= 1.0:
            valid = True
            
            for b in final_boundaries:
                if abs(idx - b) < min_segment_tokens // 10:
                    valid = False
                    break
            
            if valid:
                final_boundaries.append(idx)
    
    final_boundaries.sort()
    
    filtered_boundaries = []
    last_boundary = 0
    
    for boundary in final_boundaries:
        segment_size = estimate_token_count(" ".join(sentences[last_boundary:boundary]))
        if segment_size >= min_segment_tokens:
            filtered_boundaries.append(boundary)
            last_boundary = boundary
    
    return filtered_boundaries

def chunk_ai_response_with_topic_detection(ai_text: str) -> List[str]:
    """
    Split AI response into semantic chunks using topic boundary detection.
    """
    if estimate_token_count(ai_text) <= TARGET_CHUNK_SIZE:
        return [ai_text]
    
    paragraphs = split_into_paragraphs(ai_text)
    
    if len(paragraphs) > 1 and all(estimate_token_count(p) < TARGET_CHUNK_SIZE for p in paragraphs):
        return paragraphs
    
    sentences = split_into_sentences(ai_text)
    
    if len(sentences) <= 1:
        return [ai_text]
    
    boundaries = detect_topic_boundaries(ai_text, min_segment_tokens=TARGET_CHUNK_SIZE // 2)
    
    if not boundaries:
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_token_count(sentence)
            
            if current_token_count + sentence_tokens > TARGET_CHUNK_SIZE and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    chunks = []
    start_idx = 0
    
    for boundary in boundaries:
        if boundary > start_idx:
            chunk = " ".join(sentences[start_idx:boundary])
            chunks.append(chunk)
            start_idx = boundary
    
    if start_idx < len(sentences):
        chunk = " ".join(sentences[start_idx:])
        chunks.append(chunk)
    
    return chunks

def extract_topic_tags(text: str, max_tags: int = 3) -> List[str]:
    """Extract simple topic tags from text based on frequency."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    words = cleaned.split()
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                 'through', 'over', 'before', 'between', 'after', 'since', 'without',
                 'under', 'of', 'this', 'that', 'these', 'those', 'it', 'they', 'them',
                 'assistant', 'user'}
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    tags = [word for word, _ in sorted_words[:max_tags]]
    
    return tags

def get_recency_score(timestamp: str) -> float:
    """Calculate a recency score based on timestamp (0-1 scale)."""
    if not timestamp:
        return 0.0
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now().astimezone()
        
        days_diff = (now - dt).days
        
        score = max(0.0, min(1.0, 1.0 * (0.9 ** days_diff)))
        return score
    except Exception:
        return 0.0

def create_conversation_turns(conversation: Dict[str, Any], 
                             indexed_turn_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Create conversation turns (user message + AI response pairs) from a conversation."""
    turns = []
    indexed_turn_ids = indexed_turn_ids or set()
    
    messages = conversation.get("chat_messages", [])
    conv_id = conversation.get("uuid", "unknown")
    conv_name = conversation.get("name", "Untitled conversation")
    
    i = 0
    while i < len(messages) - 1:
        user_msg = messages[i]
        ai_msg = messages[i+1]
        
        if user_msg.get("sender") == "human" and ai_msg.get("sender") == "assistant":
            turn_id = f"{user_msg.get('uuid', '')}_{ai_msg.get('uuid', '')}"
            
            if turn_id in indexed_turn_ids:
                i += 2
                continue
            
            user_text = ""
            if user_msg.get("text"):
                user_text = user_msg.get("text")
            elif isinstance(user_msg.get("content"), dict) and user_msg["content"].get("text"):
                user_text = user_msg["content"]["text"]
            
            ai_text = ""
            if ai_msg.get("text"):
                ai_text = ai_msg.get("text")
            elif isinstance(ai_msg.get("content"), dict) and ai_msg["content"].get("text"):
                ai_text = ai_msg["content"]["text"]
            
            user_text_clean = clean_text(user_text)
            ai_text_clean = clean_text(ai_text)
            
            if user_text_clean and ai_text_clean:
                user_timestamp = user_msg.get("created_at", "")
                ai_timestamp = ai_msg.get("created_at", "")
                
                timestamp = max(user_timestamp, ai_timestamp) if user_timestamp and ai_timestamp else (user_timestamp or ai_timestamp)
                
                turn = {
                    "turn_id": turn_id,
                    "conversation_id": conv_id,
                    "conversation_name": conv_name,
                    "user_message_id": user_msg.get("uuid", ""),
                    "ai_message_id": ai_msg.get("uuid", ""),
                    "user_text": user_text,
                    "user_text_clean": user_text_clean,
                    "ai_text": ai_text,
                    "ai_text_clean": ai_text_clean,
                    "timestamp": timestamp,
                    "message_index": i,
                    "recency_score": get_recency_score(timestamp),
                    "topic_tags": extract_topic_tags(user_text_clean + " " + ai_text_clean)
                }
                
                turns.append(turn)
        
        i += 1
    
    return turns

def create_overlapping_chunks(turn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create overlapping chunks from a conversation turn with sliding windows."""
    chunks = []
    
    ai_chunks = chunk_ai_response_with_topic_detection(turn["ai_text_clean"])
    
    if len(ai_chunks) <= 1:
        chunk_id = f"{turn['turn_id']}_chunk_0"
        
        chunk = {
            "chunk_id": chunk_id,
            "turn_id": turn["turn_id"],
            "conversation_id": turn["conversation_id"],
            "conversation_name": turn["conversation_name"],
            "user_message_id": turn["user_message_id"],
            "ai_message_id": turn["ai_message_id"],
            "user_text": turn["user_text_clean"],
            "ai_text": turn["ai_text_clean"],
            "text": f"User: {turn['user_text_clean']}\n\nAssistant: {turn['ai_text_clean']}",
            "timestamp": turn["timestamp"],
            "message_index": turn["message_index"],
            "chunk_index": 0,
            "total_chunks": 1,
            "recency_score": turn["recency_score"],
            "topic_tags": turn["topic_tags"],
            "adjacent_context": ""
        }
        
        chunks.append(chunk)
        return chunks
    
    for i, ai_chunk in enumerate(ai_chunks):
        chunk_id = f"{turn['turn_id']}_chunk_{i}"
        
        adjacent_context = ""
        
        if i > 0:
            prev_sentences = split_into_sentences(ai_chunks[i-1])
            if prev_sentences:
                num_sentences = min(2, len(prev_sentences))
                adjacent_context += "Previous: " + " ".join(prev_sentences[-num_sentences:]) + "\n\n"
        
        if i < len(ai_chunks) - 1:
            next_sentences = split_into_sentences(ai_chunks[i+1])
            if next_sentences:
                num_sentences = min(2, len(next_sentences))
                adjacent_context += "Next: " + " ".join(next_sentences[:num_sentences])
        
        chunk_text = f"User: {turn['user_text_clean']}\n\nAssistant: {ai_chunk}"
        
        chunk_topic_tags = extract_topic_tags(turn['user_text_clean'] + " " + ai_chunk)
        
        chunk = {
            "chunk_id": chunk_id,
            "turn_id": turn["turn_id"],
            "conversation_id": turn["conversation_id"],
            "conversation_name": turn["conversation_name"],
            "user_message_id": turn["user_message_id"],
            "ai_message_id": turn["ai_message_id"],
            "user_text": turn["user_text_clean"],
            "ai_text": ai_chunk,
            "text": chunk_text,
            "timestamp": turn["timestamp"],
            "message_index": turn["message_index"],
            "chunk_index": i,
            "total_chunks": len(ai_chunks),
            "recency_score": turn["recency_score"],
            "topic_tags": chunk_topic_tags,
            "adjacent_context": adjacent_context
        }
        
        chunks.append(chunk)
    
    return chunks

def extract_conversation_turns(conversations: List[Dict[str, Any]], 
                              indexed_turn_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Extract conversation turns and create chunks for indexing."""
    all_chunks = []
    indexed_turn_ids = indexed_turn_ids or set()
    
    for conv in tqdm(conversations, desc="Processing conversations"):
        turns = create_conversation_turns(conv, indexed_turn_ids)
        
        for turn in turns:
            chunks = create_overlapping_chunks(turn)
            all_chunks.extend(chunks)
    
    print(f"Extracted {len(all_chunks)} chunks for indexing")
    return all_chunks

def generate_embeddings(chunks: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, List]:
    """Generate embeddings for chunks"""
    model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
    
    ids = []
    texts = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        texts.append(chunk["text"])
        
        metadata = {
            "conversation_id": chunk["conversation_id"],
            "conversation_name": chunk["conversation_name"],
            "turn_id": chunk["turn_id"],
            "user_message_id": chunk["user_message_id"],
            "ai_message_id": chunk["ai_message_id"],
            "timestamp": chunk["timestamp"],
            "message_index": chunk["message_index"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "recency_score": chunk["recency_score"],
            "topic_tags": ",".join(chunk["topic_tags"]),
            "adjacent_context": chunk["adjacent_context"]
        }
        
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
    indexed_turn_ids = set(checkpoint["indexed_turn_ids"]) if checkpoint and "indexed_turn_ids" in checkpoint else set()
    
    conversations = load_conversations(file_path)
    if not conversations:
        print("No conversations to index. Exiting.")
        return
     
    chunks = extract_conversation_turns(conversations, indexed_turn_ids)
    if not chunks:
        print("No new chunks to index. Exiting.")
        return

    if progress_callback:
        progress_callback(0.0, len(chunks), 0)
    
    client = setup_chroma_db()
    collection = get_or_create_collection(client)
    
    processed_turn_ids = set()
    
    total_indexed = 0
    for chunk_start in range(0, len(chunks), checkpoint_interval):
        chunk_end = min(chunk_start + checkpoint_interval, len(chunks))
        batch_chunks = chunks[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//checkpoint_interval + 1}: chunks {chunk_start} to {chunk_end-1}")
        
        data = generate_embeddings(batch_chunks)
        
        total_in_batch = len(data["ids"])
        for i in tqdm(range(0, total_in_batch, batch_size), desc="Indexing in ChromaDB"):
            end_idx = min(i + batch_size, total_in_batch)
            
            batch_data = {
                "ids": data["ids"][i:end_idx],
                "embeddings": data["embeddings"][i:end_idx],
                "metadatas": data["metadatas"][i:end_idx],
                "documents": data["documents"][i:end_idx]
            }
            
            collection.add(**batch_data)
        
        for chunk in batch_chunks:
            processed_turn_ids.add(chunk["turn_id"])
        
        total_indexed += total_in_batch
        
        all_indexed_turn_ids = indexed_turn_ids.union(processed_turn_ids)
        
        checkpoint_data = {
            "file_path": file_path,
            "indexed_turn_ids": list(all_indexed_turn_ids),
            "total_indexed": total_indexed,
            "last_processed": chunk_end
        }
        save_checkpoint(checkpoint_data, file_path)
        
        print(f"Checkpoint saved after processing {total_indexed} chunks")

        if progress_callback:
            progress = total_indexed / len(chunks) if chunks else 1.0
            progress_callback(progress, len(chunks), total_indexed)
    
    print(f"Successfully indexed {total_indexed} chunks into ChromaDB")
    print(f"Collection stats: {collection.count()} total documents")

    if progress_callback:
        progress_callback(1.0, len(chunks), total_indexed)

# # Test function to display sample chunks from conversations
# def test_chunking(file_path: str, num_conversations: int = 20, verbose: bool = True):
#     """
#     Test the chunking process on a sample of conversations and display the results.
    
#     Args:
#         file_path: Path to the conversations.json file
#         num_conversations: Number of conversations to sample
#         verbose: Whether to print detailed information about each chunk
#     """
#     print(f"\n{'='*80}\nTESTING CHUNKING ON SAMPLE CONVERSATIONS\n{'='*80}")
    
#     # Load conversations
#     conversations = load_conversations(file_path)
#     if not conversations:
#         print("No conversations found. Exiting test.")
#         return
    
#     sample = conversations[:num_conversations]
#     print(f"Testing on {len(sample)} conversations")
    
#     total_chunks = 0
    
#     for conv_idx, conv in enumerate(sample):
#         print(f"\n{'-'*80}\nConversation {conv_idx+1}: {conv.get('name', 'Untitled')}\n{'-'*80}")
        
#         turns = create_conversation_turns(conv, set())
#         print(f"Found {len(turns)} turns in this conversation")
        
#         for turn_idx, turn in enumerate(turns):
#             print(f"\nTurn {turn_idx+1}:")
#             print(f"User: {turn['user_text_clean'][:100]}..." if len(turn['user_text_clean']) > 100 else f"User: {turn['user_text_clean']}")
            
#             chunks = create_overlapping_chunks(turn)
#             total_chunks += len(chunks)
            
#             print(f"AI response split into {len(chunks)} chunks")
            
#             if verbose:
#                 for chunk_idx, chunk in enumerate(chunks):
#                     print(f"\n  Chunk {chunk_idx+1}/{len(chunks)}:")
#                     print(f"  Topic tags: {', '.join(chunk['topic_tags'])}")
#                     print(f"  AI text ({estimate_token_count(chunk['ai_text'])} tokens):")
#                     print(f"  {chunk['ai_text'][:150]}..." if len(chunk['ai_text']) > 150 else f"  {chunk['ai_text']}")
#                     if chunk['adjacent_context']:
#                         print(f"  Adjacent context: {chunk['adjacent_context'][:100]}..." if len(chunk['adjacent_context']) > 100 else f"  Adjacent context: {chunk['adjacent_context']}")
    
#     print(f"\n{'='*80}\nSummary: Processed {len(sample)} conversations with {total_chunks} total chunks\n{'='*80}")

# if __name__ == "__main__":
#     import sys
#     file_path = "conversations.json"
    
#     if len(sys.argv) > 1:
#         file_path = sys.argv[1]
    
#     test_chunking(file_path, num_conversations=20, verbose=True)
