import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from datetime import datetime
import time

CHROMA_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "claude_conversations"
DENSE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
CLAUDE_BASE_URL = "https://claude.ai/chat/"

_model_singleton = None

def get_model():
    """Get or create the sentence transformer model singleton."""
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = SentenceTransformer(DENSE_EMBEDDING_MODEL)
    return _model_singleton

def setup_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return a ChromaDB client."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DIR}. Please run indexing first.")
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client

def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get the ChromaDB collection for conversations."""
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except ValueError:
        raise ValueError(f"Collection '{COLLECTION_NAME}' not found. Please run indexing first.")

def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for the search query."""
    model = get_model() 
    embedding = model.encode(query)
    return embedding.tolist()

def search_conversations(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search for conversations using both keyword and dense retrieval."""
    start_time = time.time()
    
    client = setup_chroma_client()
    collection = get_collection(client)
    
    embedding_start = time.time()
    query_embedding = generate_query_embedding(query)
    embedding_time = time.time() - embedding_start
    
    query_start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )
    query_time = time.time() - query_start
    
    formatted_results = []
    
    if not results["metadatas"] or not results["metadatas"][0]:
        total_time = time.time() - start_time
        print(f"Search completed in {total_time:.2f}s (embedding: {embedding_time:.2f}s, query: {query_time:.2f}s)")
        return formatted_results
    
    for i, (metadata, document, distance) in enumerate(zip(
            results["metadatas"][0], 
            results["documents"][0],
            results["distances"][0])):
        
        result = {
            "rank": i + 1,
            "relevance_score": 1 - distance,
            "conversation_id": metadata["conversation_id"],
            "conversation_name": metadata["conversation_name"],
            "message_id": metadata["message_id"],
            "sender": metadata["sender"],
            "timestamp": metadata["timestamp"],
            "message_index": metadata["message_index"],
            "text": document,
            "url": generate_conversation_url(metadata["conversation_id"])
        }
        
        formatted_results.append(result)
    
    total_time = time.time() - start_time
    print(f"Search completed in {total_time:.2f}s (embedding: {embedding_time:.2f}s, query: {query_time:.2f}s)")
    
    return formatted_results

def generate_conversation_url(conversation_id: str) -> str:
    """Generate a URL to the conversation on Claude.ai."""
    return f"{CLAUDE_BASE_URL}{conversation_id}"

def format_timestamp(timestamp_str: str) -> str:
    """Format a timestamp string for display."""
    if not timestamp_str:
        return "Unknown time"
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str

def display_search_results(results: List[Dict[str, Any]]) -> None:
    """Display search results in a readable format."""
    if not results:
        print("No matching conversations found.")
        return
    
    print(f"\n{'=' * 80}\n")
    print(f"Found {len(results)} matching conversations:\n")
    
    for result in results:
        print(f"#{result['rank']} - Score: {result['relevance_score']:.2f}")
        print(f"Conversation: {result['conversation_name']}")
        print(f"URL: {result['url']}")
        print(f"Time: {format_timestamp(result['timestamp'])}")
        print(f"Sender: {result['sender']}")
        print(f"\n{result['text']}\n")
        print(f"\n{'-' * 80}\n")

def search_and_display(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search for conversations and display results."""
    try:
        results = search_conversations(
            query=query, 
            top_k=top_k
        )
        display_search_results(results)
        return results
    except Exception as e:
        print(f"Error searching conversations: {e}")
        return []