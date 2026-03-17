from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import requests
import uvicorn

from orchestrator_modular import process_video, process_video_from_source


class VideoRequest(BaseModel):
    video_id: str
    callback_url: str
    file_path: Optional[str] = None

    s3_url: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    video_url: Optional[str] = None
    video_name: Optional[str] = None


app = FastAPI(title="Meet Recorder AI Microservice")


def normalize_source_input(
    s3_url=None,
    source=None,
    source_url=None,
    video_url=None,
    video_name=None,
):
    # Priority: direct sources first
    direct_source = s3_url or source or source_url or video_url
    if direct_source:
        return direct_source

    # video_name fallback (NO hardcoded folder)
    if video_name:
        return video_name

    return None


def send_webhook(callback_url: str, payload: dict, is_error: bool = False):
    try:
        print(f"📤 Sending results to webhook: {callback_url}")

        response = requests.post(
            callback_url,
            json=payload,
            timeout=(10, 20)
        )
        response.raise_for_status()

        print(f"✅ Webhook delivered successfully: {response.status_code}")

    except Exception as e:
        print(f"⚠️ Webhook error: {str(e)}")


def run_ai_pipeline(video_id: str, callback_url: str, file_path: str = None, source: str = None):
    print(f"\n⚙️ Background Task Started for Video ID: {video_id}")

    try:
        if source:
            print(f"🌐 Cloud Mode Detected. Source: {source}")
            result_json = process_video_from_source(video_id=video_id, source=source)
        elif file_path:
            print(f"💻 Local Mode. File: {file_path}")
            result_json = process_video(file_path)
        else:
            raise ValueError("No input provided")

        payload = {
            "video_id": video_id,
            "status": "completed",
            "data": result_json
        }

        send_webhook(callback_url, payload)

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")

        error_payload = {
            "video_id": video_id,
            "status": "failed",
            "error": str(e)
        }

        send_webhook(callback_url, error_payload, is_error=True)


@app.post("/process-video", status_code=202)
def trigger_processing(request: VideoRequest, background_tasks: BackgroundTasks):
    resolved_source = normalize_source_input(
        request.s3_url,
        request.source,
        request.source_url,
        request.video_url,
        request.video_name
    )

    if not request.file_path and not resolved_source:
        raise HTTPException(status_code=400, detail="No valid input provided")

    print(f"📥 Resolved source: {resolved_source}")

    background_tasks.add_task(
        run_ai_pipeline,
        request.video_id,
        request.callback_url,
        request.file_path,
        resolved_source
    )

    return {
        "message": "Processing started",
        "video_id": request.video_id,
        "resolved_source": resolved_source
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
