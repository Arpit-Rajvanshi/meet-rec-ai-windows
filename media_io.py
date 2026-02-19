import os
import subprocess
import uuid
import shutil
import cv2
import easyocr
import re
import torch

from typing import List, Optional
import numpy as np
from collections import Counter

# =============================
# OCR READER
# =============================
OCR_READER = None

IS_RUNPOD = os.getenv('RUNPOD_POD_ID') is not None

def get_ocr_reader():
    global OCR_READER
    if OCR_READER is None:
        
        print("📖 Initializing EasyOCR (one-time setup)...")
        
        # Auto-detect GPU availability
        use_gpu = torch.cuda.is_available()
        print(f"   GPU available: {use_gpu}")
        
        OCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
    return OCR_READER

def load_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    return video_path

def extract_audio(video_file: str, output_audio_file: str = None) -> str:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH.")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    if output_audio_file is None:
        output_audio_file = f"audio_{uuid.uuid4().hex}.wav"
    command = [
        "ffmpeg", "-y", "-i", video_file,
        "-ac", "1", "-ar", "16000", "-vn",
        output_audio_file
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return output_audio_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")

# =============================
# NAME CLEANING
# =============================
def clean_name(raw_text: str) -> Optional[str]:
    """Clean and validate extracted text as a potential participant name."""
    text = raw_text.strip()
    
    # Invalid patterns (UI elements, timestamps, etc.)
    invalid_patterns = [
        r'^\d+:\d+', r'^\d+\s*[AP]M', r'^\d{3,4}\s*[AP]M', r'^you$',
        r'presenting|recording|sharing|annotating',
        r'^[0-9]+$', r'^[^a-zA-Z]+$', r'^\s*$',
        r'demo|interviewgod',
        r'tier\s*\d+|strong|matches|match',
        r'good|bad|threshold|below',
        r'psychometric|situational|analysis',
        r'integrations?|description',
        r'step\s*\d+|automated|ranking',
        r'candidate|potential|hiring',
        r'farcentane|interrzbor|selup',
        r'\d+\s*-\s*\d+',
        r'^[a-z]{1,3}$',
        r'^\d+\s*[of]$',
        r'chiring|maich|gocd|beloi',
        r'feedback|very',
        r'^\d+\s*others?$',
        r'^others?$',
        r'started|joined|left',
        r'^profile$|^create$',
        r'^[A-Z]{2,5}\d+',
        r'^AM$|^PM$',
        r'but\s+diarerent|rtat|oren|rrerstomach',
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return None
    
    # Single word validation
    words = text.split()
    if len(words) == 1:
        if not (3 <= len(text) <= 15):
            return None
        if not text[0].isupper():
            return None
        # Reject all-caps single words (UI elements)
        if text.isupper() and len(text) > 2:
            return None
    
    # Clean up text
    text = re.sub(r'[^\w\s\-\']', '', text)
    text = ' '.join(text.split())
    text = re.sub(r'\(You.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(presenting.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(annotating.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+joined.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+left.*', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Final validations
    if not text or len(text) < 2:
        return None
    if not re.search(r'[a-zA-Z]{2,}', text):
        return None
    if not (2 <= len(text) <= 30):
        return None
    if re.search(r'^[A-Z]{1,3}\d+', text):
        return None
    if re.search(r'^\d+[A-Z]{1,3}$', text):
        return None
    
    return text

# =============================
# PARTICIPANT TILE DETECTION
# =============================
def detect_participant_tiles_region(frame: np.ndarray) -> tuple:
    """Detect the region where participant tiles are located."""
    height, width = frame.shape[:2]
    
    # Check right sidebar first (common in Google Meet)
    right_region = frame[:, int(width * 0.70):]
    gray = cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=80
    )
    
    if circles is not None and len(circles[0]) >= 2:
        return (int(width * 0.70), 0, width, height)
    
    # Check bottom region
    left_region = frame[:, :int(width * 0.70)]
    gray_left = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_left, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density > 0.05:
        return (int(width * 0.70), int(height * 0.60), width, height)
    
    # Default to bottom strip
    return (0, int(height * 0.60), width, height)


# =============================
# NAME EXTRACTION FROM VIDEO
# =============================
def extract_names_from_video(video_path: str, sample_timestamps: List[float] = None) -> List[str]:
    """
    Extract participant names from video using OCR.
    
    This function samples frames from the video and uses OCR to detect
    participant names from video tiles/overlays.
    
    Args:
        video_path: Path to video file
        sample_timestamps: Optional list of specific timestamps to sample
        
    Returns:
        List of unique participant names detected
    """
    print(f"📹 Extracting participant names from video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine sample points
    if sample_timestamps is None:
        # Sample 25 frames throughout the video to catch late joiners
        num_samples = 25
        start_offset = min(30 * fps, total_frames * 0.1)
        end_offset = max(total_frames - 30 * fps, total_frames * 0.9)
        sample_frames = np.linspace(start_offset, end_offset, num_samples, dtype=int)
        sample_timestamps = [f / fps for f in sample_frames]
    
    all_detected_names = []
    reader = get_ocr_reader()
    
    # Process each sample frame
    for i, timestamp in enumerate(sample_timestamps, 1):
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"   Analyzing frame {i}/{len(sample_timestamps)} at {timestamp:.1f}s...")
        
        # Detect participant tile region
        x1, y1, x2, y2 = detect_participant_tiles_region(frame)
        search_region = frame[y1:y2, x1:x2]
        
        # Run OCR on the region
        ocr_results = reader.readtext(
            search_region, 
            paragraph=False, 
            min_size=8,
            text_threshold=0.6, 
            low_text=0.3
        )
        
        # Extract and clean names
        for (bbox, text, confidence) in ocr_results:
            cleaned_name = clean_name(text)
            if cleaned_name and len(cleaned_name) <= 30:
                all_detected_names.append(cleaned_name)
                print(f"   Found: {cleaned_name} (conf: {confidence:.2f})")
    
    cap.release()
    
    # Count occurrences and filter
    name_counts = Counter(all_detected_names)
    
    # Threshold: name must appear in at least 2 frames to be considered valid
    # This helps filter out OCR noise while catching late joiners
    min_occurrences = 2
    unique_names = [name for name, count in name_counts.items() if count >= min_occurrences]
    unique_names = list(set(unique_names))
    
    # Show detection stats
    print(f"\n   Name occurrence statistics:")
    for name, count in name_counts.most_common():
        included = "✓" if name in unique_names else "✗"
        print(f"   {included} {name}: {count} times")
    
    print(f"\n✅ Detected {len(unique_names)} participants: {unique_names}")
    return unique_names