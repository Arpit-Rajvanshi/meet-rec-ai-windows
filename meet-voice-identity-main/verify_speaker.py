import json
import torch
import sys
from voice_engine import VoiceIdentityEngine

def load_seed_data(filepath="seed_data.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def verify(mystery_file_path):
    print(f"🔍 Analyzing mystery voice from: {mystery_file_path}")
    
    # 1. Generate fingerprint for the mystery audio
    engine = VoiceIdentityEngine()
    mystery_vector = torch.tensor(engine.generate_embedding(mystery_file_path))
    
    # 2. Load the enrolled database (our "Answer Key")
    print("📂 Loading Voice Database (seed_data.json)...")
    db_data = load_seed_data()
    
    results = []
    # Cosine Similarity measures the angle between two mathematical vectors
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    
    # 3. Compare mystery voice against everyone in the DB
    for person in db_data:
        enrolled_vector = torch.tensor(person["embedding"])
        score = cos(mystery_vector, enrolled_vector).item()
        results.append({"name": person["name"], "score": score})
        
    # 4. Sort results from highest score (closest match) to lowest
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 5. Display the matching logic
    print("\n📊 MATCHING RESULTS:")
    print("-" * 50)
    for idx, res in enumerate(results[:3]): # Show top 3 matches
        print(f"Rank {idx+1}: {res['name']} (Confidence: {res['score']:.4f})")
    print("-" * 50)
    
    top_match = results[0]
    gap = results[0]["score"] - results[1]["score"] if len(results) > 1 else 1.0
    
    print(f"📈 Separation Gap (Top 1 vs Top 2): {gap:.4f}")
    
    # Apply the strict Confidence Gate validated in your Jupyter Notebook
    THRESHOLD = 0.82
    if top_match["score"] > THRESHOLD and gap > 0.10:
        print(f"\n✅ FINAL DECISION: Identity Confirmed -> {top_match['name']}")
    else:
        print(f"\n⚠️ FINAL DECISION: [REVIEW NEEDED] Voice is ambiguous. Left as Generic Speaker.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_speaker.py <path_to_mystery_audio>")
    else:
        verify(sys.argv[1])