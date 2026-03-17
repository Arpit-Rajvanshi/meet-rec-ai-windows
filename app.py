from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import requests
import os

# **[NEW]: Import both the local processor AND the cloud processor from Checkpoint 2**
from orchestrator_modular import process_video, process_video_from_url

# ==========================================
# 1. API Schema Definitions
# ==========================================
class VideoRequest(BaseModel):
    video_id: str
    callback_url: str
    # **[NEW]: Both fields are Optional. The AI will figure out which one you sent.**
    file_path: Optional[str] = None
    s3_url: Optional[str] = None

# ==========================================
# 2. Initialize FastAPI
# ==========================================
app = FastAPI(title="Meet Recorder AI Microservice")

# ==========================================
# 3. The Background Worker (Outbound)
# ==========================================
def run_ai_pipeline(video_id: str, callback_url: str, file_path: str = None, s3_url: str = None):
    """Runs the heavy AI tasks without blocking the web server."""
    print(f"\n⚙️ Background Task Started for Video ID: {video_id}")
    
    try:
        # **[NEW]: Intelligent Routing Router**
        if s3_url:
            print(f"🌐 Cloud Mode Detected. Fetching from: {s3_url}")
            result_json = process_video_from_url(video_id, s3_url)
        elif file_path:
            print(f"💻 Local Mode Detected. Processing file: {file_path}")
            result_json = process_video(file_path)
        else:
            raise ValueError("Fatal Error: Neither file_path nor s3_url was provided.")
        
        # Step B: Format the payload for Node.js
        payload = {
            "video_id": video_id,
            "status": "completed",
            "data": result_json
        }
        
        # Step C: Send the payload to Node.js (The Webhook Push)
        headers = {"Authorization": "Bearer meet_recorder_secret_123"}
        
        print(f"📤 Sending results to Node.js Webhook: {callback_url}")
        response = requests.post(callback_url, json=payload, headers=headers)
        response.raise_for_status()
        
        print(f"✅ Webhook delivered successfully. Node.js responded: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        error_payload = {"video_id": video_id, "status": "failed", "error": str(e)}
        try:
            requests.post(callback_url, json=error_payload)
        except:
            print("❌ Failed to reach Node.js webhook with error payload.")

# ==========================================
# 4. The API Endpoints (Inbound)
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "AI Pipeline is online and ready"}

@app.post("/process-video", status_code=202)
def trigger_processing(request: VideoRequest, background_tasks: BackgroundTasks):
    """Receives the trigger, validates it, and spawns the background task."""
    
    # **[NEW]: Validation to ensure at least ONE method is provided**
    if not request.file_path and not request.s3_url:
        raise HTTPException(status_code=400, detail="You must provide either 'file_path' or 's3_url'")
        
    if request.file_path and not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"Local file not found: {request.file_path}")
    
    # Spawn the background task, passing whichever arguments exist
    background_tasks.add_task(run_ai_pipeline, request.video_id, request.callback_url, request.file_path, request.s3_url)
    
    return {
        "message": "Processing started asynchronously",
        "video_id": request.video_id,
        "mode": "cloud" if request.s3_url else "local"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
