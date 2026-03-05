"""
Meeting Intelligence Orchestrator - Production Architecture

This orchestrator runs on YOUR SERVER and coordinates:
1. Video → Audio conversion (local)
2. OCR name extraction (local)
3. Transcription + Diarization (RunPod GPU)
4. Speaker mapping (local - Gemini)
5. Meeting analysis (local - Gemini)

Only GPU work happens on RunPod. Everything else is local and easily changeable.
"""

import os
import requests 
import json
import time
import base64
import runpod
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Local modules (these run on your server)
import media_io
from speaker_mapper import map_speakers_to_names_gemini
from meeting_analysis_functions import MeetingAnalyzer

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not RUNPOD_API_KEY:
    raise ValueError("RUNPOD_API_KEY not found in environment")
if not RUNPOD_ENDPOINT_ID:
    raise ValueError("RUNPOD_ENDPOINT_ID not found in environment")

# Initialize RunPod
runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def video_to_audio(video_path: str, output_audio: str = None) -> str:
    """
    Convert video to audio (runs locally)
    
    Future: Replace with S3/R2 fetch
    """
    print("🎬 Converting video to audio...")
    
    if output_audio is None:
        output_audio = f"audio_{int(time.time())}.wav"
    
    media_io.extract_audio(video_path, output_audio_file=output_audio)
    
    file_size = Path(output_audio).stat().st_size / (1024 * 1024)
    print(f"✅ Audio extracted: {output_audio} ({file_size:.1f}MB)")
    
    return output_audio


def extract_participant_names(video_path: str, sample_timestamps: List[float] = None) -> List[str]:
    """
    Extract participant names using OCR (runs locally)
    """
    print("\n📹 Extracting participant names from video (OCR)...")
    
    participant_names = media_io.extract_names_from_video(
        video_path,
        sample_timestamps=sample_timestamps
    )
    
    print(f"✅ Detected {len(participant_names)} participants: {participant_names}")
    
    return participant_names


def transcribe_and_diarize_gpu(audio_path: str, min_speakers: int = None, max_speakers: int = None) -> Dict[str, Any]:
    """
    Send audio to RunPod for GPU processing (transcription + diarization)
    
    Returns raw transcript with speaker IDs (SPEAKER_00, SPEAKER_01, etc.)
    """
    print("\n⚡ Sending audio to RunPod for GPU processing...")
    print("   Tasks: Whisper Transcription → Pyannote Diarization")
    
    # Encode audio to base64
    print("   📦 Encoding audio...")
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    audio_size_mb = len(audio_data) / (1024 * 1024)
    print(f"   ✅ Encoded {audio_size_mb:.1f}MB")
    
    # Prepare payload
    payload = {
        "audio_base64": audio_base64
    }
    
    if min_speakers:
        payload["min_speakers"] = min_speakers
    if max_speakers:
        payload["max_speakers"] = max_speakers
    
    # Submit to RunPod
    print("   📤 Submitting to RunPod...")
    
    try:
        run_request = endpoint.run(payload)
        print(f"   ✅ Job submitted: {run_request.job_id}")
        print("   ⏳ Processing on GPU...")
        
        # Monitor progress
        start_time = time.time()
        last_status = None
        
        while True:
            status = run_request.status()
            
            if status != last_status:
                elapsed = time.time() - start_time
                print(f"   [{elapsed:.1f}s] Status: {status}")
                last_status = status
            
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                break
            
            time.sleep(2)
        
        if status != "COMPLETED":
            raise Exception(f"RunPod job failed with status: {status}")
        
        # Get results
        result = run_request.output()
        
        if result.get("status") != "success":
            raise Exception(f"GPU processing failed: {result.get('message')}")
        
        gpu_time = time.time() - start_time
        print(f"\n✅ GPU processing completed in {gpu_time:.1f}s")
        print(f"   Speakers detected: {result['metadata']['speakers_detected']}")
        print(f"   Transcript segments: {result['metadata']['total_segments']}")
        
        return result
        
    except Exception as e:
        print(f"❌ RunPod processing failed: {str(e)}")
        raise


def map_speakers_to_names(transcript: List[Dict], participant_names: List[str]) -> List[Dict]:
    """
    Map speaker IDs to real names using Gemini (runs locally)
    """
    print("\n🧠 Mapping speakers to names (Gemini on your server)...")
    
    if not participant_names:
        print("   ⚠️  No participant names - keeping speaker IDs")
        return transcript
    
    try:
        mapped_transcript = map_speakers_to_names_gemini(
            diarized_transcript=transcript,
            participant_names=participant_names
        )
        print("✅ Speaker mapping completed")
        return mapped_transcript
        
    except Exception as e:
        print(f"⚠️  Speaker mapping failed: {str(e)}")
        print("   Continuing with speaker IDs...")
        return transcript


def analyze_meeting(transcript_file: str, participant_names: List[str]) -> Dict[str, Any]:
    """
    Analyze meeting using Gemini (runs locally)
    """
    print("\n🧠 Analyzing meeting (Gemini on your server)...")
    
    try:
        analyzer = MeetingAnalyzer(
            transcript_file=transcript_file,
            participant_names=participant_names
        )
        
        analysis_result = analyzer.analyze()
        print("✅ Analysis completed")
        
        return analysis_result
        
    except Exception as e:
        print(f"⚠️  Analysis failed: {str(e)}")
        return None


# ==========================================
# MAIN PIPELINE
# ==========================================

def process_video(
    video_path: str,
    output_dir: str = "results",
    sample_timestamps: List[float] = None,
    min_speakers: int = None,
    max_speakers: int = None
) -> Dict[str, Any]:
    """
    Complete meeting intelligence pipeline with modular architecture
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save results
        sample_timestamps: Optional timestamps for OCR sampling
        min_speakers: Minimum speakers for diarization
        max_speakers: Maximum speakers for diarization
        
    Returns:
        Complete results dictionary
    """
    
    print("="*70)
    print("🎬 MEETING INTELLIGENCE PIPELINE - MODULAR ARCHITECTURE")
    print("="*70)
    print("📍 Local: Video→Audio, OCR, Speaker Mapping, Analysis")
    print("⚡ RunPod: Transcription + Diarization (GPU only)")
    print("="*70)
    
    start_time = time.time()
    temp_audio_path = None
    
    try:
        # =========================================
        # STEP 1: VIDEO → AUDIO (Local)
        # =========================================
        print(f"\n📹 Video: {video_path}")
        temp_audio_path = video_to_audio(video_path)
        
        # =========================================
        # STEP 2: OCR NAME EXTRACTION (Local)
        # =========================================
        participant_names = extract_participant_names(video_path, sample_timestamps)
        
        # Smart speaker constraints
        if participant_names and not min_speakers and not max_speakers:
            detected_count = len(participant_names)
            if detected_count > 1:
                min_speakers = 2
                max_speakers = detected_count
                print(f"\n🎯 Setting speaker constraints: {min_speakers}-{max_speakers}")
        
        # =========================================
        # STEP 3: TRANSCRIPTION + DIARIZATION (RunPod GPU)
        # =========================================
        gpu_result = transcribe_and_diarize_gpu(
            temp_audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        transcript = gpu_result['transcript']
        gpu_metadata = gpu_result['metadata']
        
        # =========================================
        # STEP 4: SPEAKER MAPPING (Local - Gemini)
        # =========================================
        mapped_transcript = map_speakers_to_names(transcript, participant_names)
        
        # =========================================
        # STEP 5: SAVE TRANSCRIPT
        # =========================================
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        transcript_file = output_path / f"transcript_{timestamp}.json"
        
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(mapped_transcript, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Transcript saved: {transcript_file}")
        
        # =========================================
        # STEP 6: MEETING ANALYSIS (Local - Gemini)
        # =========================================
        analysis_result = analyze_meeting(str(transcript_file), participant_names)
        
        if analysis_result:
            analysis_file = output_path / f"analysis_{timestamp}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            print(f"💾 Analysis saved: {analysis_file}")
        
        # =========================================
        # SUMMARY
        # =========================================
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED")
        print("="*70)
        print(f"Total time: {total_time:.1f}s")
        print(f"  - GPU time (RunPod): {gpu_metadata['processing_time_seconds']}s")
        print(f"  - Local CPU time: {total_time - gpu_metadata['processing_time_seconds']:.1f}s")
        print()
        print(f"Participants: {', '.join(participant_names) if participant_names else 'None detected'}")
        print(f"Speakers detected: {gpu_metadata['speakers_detected']}")
        print(f"Transcript segments: {len(mapped_transcript)}")
        print()
        print("Files saved:")
        print(f"  📄 {transcript_file}")
        if analysis_result:
            print(f"  📊 {analysis_file}")
        print("="*70)
        
        return {
            "transcript": mapped_transcript,
            "analysis": analysis_result,
            "participants": participant_names,
            "files": {
                "transcript": str(transcript_file),
                "analysis": str(analysis_file) if analysis_result else None
            },
            "timing": {
                "total_seconds": total_time,
                "gpu_seconds": gpu_metadata['processing_time_seconds'],
                "local_seconds": total_time - gpu_metadata['processing_time_seconds']
            },
            "metadata": gpu_metadata
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise
        
    finally:
        # Cleanup temporary audio
        if temp_audio_path and Path(temp_audio_path).exists():
            try:
                os.remove(temp_audio_path)
                print(f"\n🧹 Cleaned up: {temp_audio_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {temp_audio_path}: {e}")

def process_video_from_url(video_id: str, s3_url: str) -> dict:
    print(f"\n☁️ Downloading video {video_id} from cloud storage...")
    temp_video_path = f"temp_video_{video_id}.mp4" 
    
    try:
        response = requests.get(s3_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(temp_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Download complete: {temp_video_path}")
        result = process_video(temp_video_path)
        return result
        
    except Exception as e:
        print(f"❌ Cloud processing failed for {video_id}: {str(e)}")
        raise e
        
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"🧹 Cleaned up temporary cloud file: {temp_video_path}")


# ==========================================
# CLI INTERFACE
# ==========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python orchestrator_modular.py <video_file>")
        print()
        print("Example:")
        print("  python orchestrator_modular.py meeting.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    if not Path(video_file).exists():
        print(f"❌ Error: File not found: {video_file}")
        sys.exit(1)
    
    result = process_video(video_file)