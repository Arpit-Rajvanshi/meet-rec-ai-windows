import os
import json
from voice_engine import VoiceIdentityEngine

PROCESSED_DIR = "processed_audios"
SEED_FILE = "seed_data.json"

# 🛡️ THE SAFETY GATE: Only files matching these prefixes will be enrolled.
EMPLOYEE_MAP = {
    "ketan": "Ketan Patil",
    "sanket": "Sanket Satpute",
    "ninad": "Ninad Baruah",
    "isanur": "Isanur Sardar",
    "tamanna": "Tamanna",
    "mann": "Mann Vishnoi",
    "vid": "Vid",
    "saurabh": "Saurabh Bassi",
    "pranjal": "Pranjal Pritam Gogoi"
}

def rebuild_database():
    print("🧹 Rebuilding pristine database from scratch (Multi-Language Mode)...\n")
    engine = VoiceIdentityEngine()
    db_data = []
    
    enrolled_count = 0

    for filename in os.listdir(PROCESSED_DIR):
        if not filename.endswith("_clean.wav"):
            continue
            
        # Extract the raw prefix (e.g., "sanket" from "sanket_eng_clean.wav")
        raw_prefix = filename.split('_')[0].lower()
        
        # Check if this prefix is in our approved Employee Map
        if raw_prefix in EMPLOYEE_MAP:
            full_name = EMPLOYEE_MAP[raw_prefix]
            
            # ---> WE REMOVED THE SKIP LOGIC HERE <---
            # The script will now generate a distinct vector for EVERY valid audio file.
            
            print(f"🔬 Enrolling: {full_name} (from {filename})")
            filepath = os.path.join(PROCESSED_DIR, filename)
            vector = engine.generate_embedding(filepath)
            
            db_data.append({
                "name": full_name,
                "source_file": filename,
                "embedding": vector,
                "model_version": "speechbrain/spkrec-ecapa-voxceleb"
            })
            enrolled_count += 1
        else:
            # Catch mystery files and meeting recordings!
            print(f"🗑️ Ignoring {filename} - Not a registered employee in the map.")

    # Save the pristine list
    print(f"\n💾 Saving {enrolled_count} total profiles (including bilingual variations) to {SEED_FILE}...")
    with open(SEED_FILE, "w") as f:
        json.dump(db_data, f, indent=2)
        
    print("✅ Database successfully rebuilt!")

if __name__ == "__main__":
    rebuild_database()