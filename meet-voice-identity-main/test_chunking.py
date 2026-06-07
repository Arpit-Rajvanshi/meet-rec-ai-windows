import json
import torch
import torchaudio
from voice_engine import VoiceIdentityEngine

# --- PRODUCTION THRESHOLDS ---
# Gate A: Clean Audio (High Score, Moderate Gap)
HIGH_SCORE_THRESHOLD = 0.70
HIGH_SCORE_GAP = 0.10

# Gate B: Acoustic Mismatch / Echoey Room (Mid Score, Massive Gap)
MID_SCORE_THRESHOLD = 0.50
MID_SCORE_GAP = 0.20

def test_multi_chunk_averaging():
    print("🚀 Initializing Test Suite...")
    engine = VoiceIdentityEngine()
    TARGET_AUDIO = "vid2_clean.wav"
    
    print("📂 Loading Voice Database (seed_data.json)...\n")
    with open("seed_data.json", "r") as f:
        db_data = json.load(f)
        
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    # --- OUR TEST SUITE ---
    test_cases = {
        "Expected: Sanket": [
            (120.0, 119.5),
            (340.0, 349.5),
            (710.0, 719.5)
        ],
        "Expected: Ninad": [
            (760.0, 769.5),
            (810.0, 819.5),
            (915.0, 924.5)
        ],
        "Expected: Mann": [   
            (1170.0,1179.5),
            (1230.0,1239.5),
            (1320.0,1329.5)
        ]
        
    }

    # Loop through each test case
    for test_name, timestamps in test_cases.items():
        print("=" * 60)
        print(f"🧪 RUNNING TEST: {test_name}")
        print("=" * 60)
        
        chunk_vectors = []
        for start_sec, end_sec in timestamps:
            duration = end_sec - start_sec
            frame_offset = int(start_sec * 16000)
            num_frames = int(duration * 16000)
            
            signal, fs = torchaudio.load(TARGET_AUDIO, frame_offset=frame_offset, num_frames=num_frames)
            vector = engine.model.encode_batch(signal).squeeze()
            chunk_vectors.append(vector)
            
        # Average the chunks
        golden_vector = torch.mean(torch.stack(chunk_vectors), dim=0)
        
        # Compare against DB
        results = []
        for person in db_data:
            enrolled_vector = torch.tensor(person["embedding"])
            score = cos(golden_vector, enrolled_vector).item()
            results.append({"name": person["name"], "score": score})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        top_match = results[0]
        
        # --- SMART GAP CALCULATION ---
        # Find the first person in the list who is NOT the top match
        next_best = None
        for res in results[1:]:
            if res["name"] != top_match["name"]:
                next_best = res
                break
                
        gap = top_match["score"] - next_best["score"] if next_best else 1.0

        # Output Results
        print("\n📊 MATCHING RESULTS:")
        for idx, res in enumerate(results[:3]):
            print(f"Rank {idx+1}: {res['name']} (Score: {res['score']:.4f})")
            
        print(f"\n📈 True Separation Gap (vs {next_best['name']}): {gap:.4f}")

        # --- DUAL THRESHOLD GATING ---
        condition_a = top_match["score"] >= HIGH_SCORE_THRESHOLD and gap >= HIGH_SCORE_GAP
        condition_b = top_match["score"] >= MID_SCORE_THRESHOLD and gap >= MID_SCORE_GAP

        if condition_a:
            print(f"✅ PASSED GATE A (Clean Match) -> Mapped to {top_match['name']}\n")
        elif condition_b:
            print(f"✅ PASSED GATE B (Acoustic Mismatch) -> Mapped to {top_match['name']}\n")
        else:
            print(f"⚠️ FAILED BOTH GATES (Too Ambiguous) -> Left as Generic Speaker\n")

if __name__ == "__main__":
    test_multi_chunk_averaging()