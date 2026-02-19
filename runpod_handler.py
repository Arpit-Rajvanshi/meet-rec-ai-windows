import runpod
import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any
import base64

# Import ONLY GPU modules
import transcription
import diarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler - Transcription + Diarization ONLY
    
    Input:
    {
        "audio_base64": "base64_encoded_wav_audio",
        "min_speakers": 2,  # Optional
        "max_speakers": 10,  # Optional
        "num_speakers": None  # Optional - exact number
    }
    
    Output:
    {
        "status": "success",
        "transcript": [
            {
                "start": 0.0,
                "end": 5.2,
                "speaker": "SPEAKER_00",
                "text": "Hello everyone..."
            }
        ],
        "metadata": {
            "total_segments": 142,
            "speakers_detected": 3,
            "processing_time_seconds": 45.2,
            "audio_duration_seconds": 1800.5
        }
    }
    """
    
    import time
    start_time = time.time()
    
    try:
        # =========================================
        # INPUT VALIDATION
        # =========================================
        input_data = event.get('input', {})
        
        audio_base64 = input_data.get('audio_base64')
        min_speakers = input_data.get('min_speakers')
        max_speakers = input_data.get('max_speakers')
        num_speakers = input_data.get('num_speakers')
        
        if not audio_base64:
            return {
                "status": "error",
                "message": "'audio_base64' is required",
                "error_type": "ValidationError"
            }
        
        logger.info("="*70)
        logger.info("🎙️ TRANSCRIPTION + DIARIZATION PIPELINE - RUNPOD")
        logger.info("="*70)
        
        # =========================================
        # SETUP WORKSPACE
        # =========================================
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # =========================================
            # STEP 1: DECODE AUDIO
            # =========================================
            logger.info("\nSTEP 1: Decoding audio from base64")
            audio_path = temp_dir / "audio.wav"
            
            try:
                audio_data = base64.b64decode(audio_base64)
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                audio_size_mb = len(audio_data) / (1024 * 1024)
                logger.info(f"✅ Audio decoded: {audio_size_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"❌ Failed to decode audio: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Invalid audio_base64: {str(e)}",
                    "error_type": "DecodingError"
                }
            
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # =========================================
                # STEP 2: TRANSCRIPTION (GPU)
                # =========================================
                logger.info("\n⚡ STEP 2: Running Whisper transcription (GPU)")
                transcription_result = transcription.transcribe_audio_chunked(str(audio_path))
                logger.info(f"✅ Transcription complete: {len(transcription_result)} segments")
                
                # =========================================
                # STEP 3: DIARIZATION (GPU)
                # =========================================
                logger.info("\n⚡ STEP 3: Running Pyannote diarization (GPU)")
                
                if num_speakers:
                    logger.info(f"   🎯 Exact speakers: {num_speakers}")
                elif min_speakers or max_speakers:
                    logger.info(f"   🎯 Speaker range: {min_speakers or 'auto'} - {max_speakers or 'auto'}")
                else:
                    logger.info(f"   🎯 Auto-detecting speakers")
                
                pyannote_segments = diarize.run_pyannote(
                    str(audio_path),
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                logger.info(f"✅ Diarization complete: {len(pyannote_segments)} speaker segments")
                
                # =========================================
                # STEP 4: MERGE
                # =========================================
                logger.info("\n⚡ STEP 4: Merging transcription with speakers")
                raw_diarized = diarize.assign_speakers_to_transcript(
                    transcription_result,
                    pyannote_segments
                )
                logger.info(f"✅ Merged: {len(raw_diarized)} segments")
                
                # NOTE: Concatenation now happens on your server (no Docker rebuild needed!)
                # The orchestrator will call transcript_utils.concatenate_continuous_segments()
                final_transcript = raw_diarized
                
                # Ensure float timestamps
                for seg in final_transcript:
                    seg['start'] = float(seg['start'])
                    seg['end'] = float(seg['end'])
                
                # Count unique speakers
                unique_speakers = set(seg['speaker'] for seg in final_transcript)
                
                # =========================================
                # PREPARE RESPONSE
                # =========================================
                processing_time = time.time() - start_time
                audio_duration = max(seg['end'] for seg in final_transcript) if final_transcript else 0
                
                logger.info("\n" + "="*70)
                logger.info("✅ GPU PROCESSING COMPLETED")
                logger.info("="*70)
                logger.info(f"Processing time: {processing_time:.1f}s")
                logger.info(f"Audio duration: {audio_duration:.1f}s")
                logger.info(f"Speakers detected: {len(unique_speakers)}")
                logger.info(f"Transcript segments: {len(final_transcript)}")
                logger.info("="*70)
                
                return {
                    "status": "success",
                    "transcript": final_transcript,
                    "metadata": {
                        "total_segments": len(final_transcript),
                        "speakers_detected": len(unique_speakers),
                        "speaker_ids": sorted(list(unique_speakers)),
                        "processing_time_seconds": round(processing_time, 2),
                        "audio_duration_seconds": round(audio_duration, 2)
                    }
                }
                
            finally:
                os.chdir(original_cwd)
    
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }


# =========================================
# RUNPOD ENTRY POINT
# =========================================
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod GPU handler (Transcription + Diarization)")
    runpod.serverless.start({"handler": handler})