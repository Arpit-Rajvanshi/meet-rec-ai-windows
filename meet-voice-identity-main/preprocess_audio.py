import os
import sys
import subprocess

PROCESSED_DIR = "processed_audios"

def process_single_file(input_path):
    print("🚀 Starting Targeted Media Preprocessing...\n")
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    if not os.path.exists(input_path):
        print(f"❌ ERROR: Cannot find file at '{input_path}'")
        sys.exit(1)

    # Extract the filename from the path
    filename = os.path.basename(input_path)
    
    # Create the clean output filename
    name_without_ext = os.path.splitext(filename)[0]
    output_filename = f"{name_without_ext}_clean.wav"
    output_path = os.path.join(PROCESSED_DIR, output_filename)

    print(f"⏳ Extracting & Washing: {filename}...")

    # The FFmpeg command (Drops video, forces 16kHz Mono WAV)
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn", 
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ Success -> Saved to {output_path}\n")
    except subprocess.CalledProcessError:
        print(f"❌ FAILED to process {filename}. Check if file is corrupted.\n")

if __name__ == "__main__":
    # Check if the user passed a file path in the terminal
    if len(sys.argv) < 2:
        print("⚠️ Usage Error: You must provide a file path.")
        print("Example: python preprocess_audio.py my_video.mp4")
    else:
        # Pass the targeted file to the function
        target_file = sys.argv[1]
        process_single_file(target_file)