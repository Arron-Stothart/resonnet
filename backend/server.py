import tempfile
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
import uvicorn

import index
import search

app = FastAPI(title="Claude Conversation Search API")

# Add CORS middleware to allow requests from your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store indexing progress information
indexing_tasks: Dict[str, Dict[str, Any]] = {}

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class IndexingStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    total_messages: int
    processed_messages: int
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Claude Conversation Search API"}

@app.post("/search")
async def search_conversations(search_query: SearchQuery):
    try:
        results = search.search_conversations(
            query=search_query.query,
            top_k=search_query.top_k
        )
        return {"results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Search failed: {str(e)}"}
        )

@app.get("/stats")
async def get_stats():
    try:
        client = search.setup_chroma_client()
        collection = search.get_collection(client)
        count = collection.count()
        return {"total_indexed_messages": count}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get stats: {str(e)}"}
        )

@app.delete("/conversations")
async def delete_conversations():
    """Delete all indexed conversations."""
    try:
        client = search.setup_chroma_client()
        collection = search.get_collection(client)
        collection.delete()
        return {"message": "All conversations deleted successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete conversations: {str(e)}"}
        )

@app.get("/has-conversations")
async def has_conversations():
    """Check if any conversations are indexed."""
    try:
        client = search.setup_chroma_client()
        collection = search.get_collection(client)
        count = collection.count()
        # Get unique conversation IDs from metadata
        unique_conversations = len(set(doc['conversation_id'] for doc in collection.get()['metadatas']))
        return {
            "has_conversations": count > 0,
            "conversation_count": unique_conversations,
            "message_count": count
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to check conversations: {str(e)}"}
        )

def progress_callback(task_id: str, progress: float, total: int, processed: int):
    """Update the progress of an indexing task."""
    if task_id in indexing_tasks:
        indexing_tasks[task_id].update({
            "progress": progress,
            "total_messages": total,
            "processed_messages": processed
        })

def index_file_task(task_id: str, file_path: str):
    """Background task to index a file."""
    try:
        # Create a callback that includes the task_id
        def callback(p, t, n):
            return progress_callback(task_id, p, t, n)
        
        # Start indexing
        indexing_tasks[task_id]["status"] = "processing"
        index.index_conversations(
            file_path=file_path,
            batch_size=1000,
            checkpoint_interval=5000,
            progress_callback=callback
        )
        
        # Update task status on completion
        indexing_tasks[task_id]["status"] = "completed"
        indexing_tasks[task_id]["progress"] = 1.0
        
    except Exception as e:
        # Update task status on error
        indexing_tasks[task_id]["status"] = "failed"
        indexing_tasks[task_id]["error"] = str(e)
        print(f"Indexing error: {e}")

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and index a Claude conversations export file."""
    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            # Write the uploaded file to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Generate a task ID
        task_id = f"task_{len(indexing_tasks) + 1}"
        
        # Initialize task status
        indexing_tasks[task_id] = {
            "status": "starting",
            "progress": 0.0,
            "total_messages": 0,
            "processed_messages": 0,
            "file_name": file.filename
        }
        
        # Start indexing in the background
        background_tasks.add_task(index_file_task, task_id, temp_file_path)
        
        return {"task_id": task_id, "message": "File upload started"}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )

@app.get("/indexing-status/{task_id}")
async def get_indexing_status(task_id: str):
    """Get the status of an indexing task."""
    if task_id not in indexing_tasks:
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )
    
    task_info = indexing_tasks[task_id]
    return IndexingStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        total_messages=task_info["total_messages"],
        processed_messages=task_info["processed_messages"],
        error=task_info.get("error")
    )

if __name__ == "__main__":
    # Make sure the data directories exist
    index.setup_directories()
    
    # Run the server
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)