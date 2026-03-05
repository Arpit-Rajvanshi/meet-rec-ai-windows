import json

# **[NEW]: This is the strict Phase 1 Master Schema that our AI must output.**
# We use this to validate the final output before sending it to the Node backend.
EXPECTED_SCHEMA = {
    "metrics": {
        "engagement_score": float,
        "participation_score": float
    },
    "participants": list, # List of strings (names)
    "summary": str,
    "action_items": [
        {
            "task": str,
            "owner": str,
            "deadline": str
        }
    ],
    "ai_analysis": {
        "meeting_effectiveness": str,
        "decision_clarity": str,
        "participant_alignment": str,
        "open_questions": str,
        "risks": str,
        "ambiguity": str,
        "repeated_topics": str
    },
    "transcript": [
        {
            "timestamp": str, # Format: "MM:SS"
            "speaker": str,
            "text": str,
            "category": str # Must be: "action", "decision", "question", or "discussion"
        }
    ]
}

# **[NEW]: Validation logic to strictly check every key, type, and missing field.**
def validate_output(output_file_path="results/analysis_1772718759.json"):
    try:
        with open(output_file_path, 'r') as f:
            data = json.load(f)
            
        print(f"🔍 Validating {output_file_path} against strict Phase 1 Schema...")
        
        # 1. Check Top-Level Keys
        for key in EXPECTED_SCHEMA.keys():
            if key not in data:
                print(f"❌ FAILED: Missing top-level key: '{key}'")
                return False
                
        # 2. Check AI Analysis Structure
        for analysis_key in EXPECTED_SCHEMA["ai_analysis"].keys():
            if analysis_key not in data.get("ai_analysis", {}):
                print(f"❌ FAILED: Missing AI Analysis key: '{analysis_key}'")
                return False
                
        # 3. Check Transcript Categorization
        transcript = data.get("transcript", [])
        if not transcript:
            print("⚠️ WARNING: Transcript is empty.")
        
        valid_categories = ["action", "decision", "question", "discussion"]
        for line in transcript:
            if line.get("category") not in valid_categories:
                print(f"❌ FAILED: Invalid transcript category '{line.get('category')}' at timestamp {line.get('timestamp')}")
                return False
                
        print("✅ SUCCESS: JSON Output perfectly matches Phase 1 Schema!")
        return True
        
    except FileNotFoundError:
        print(f"❌ FAILED: Could not find {output_file_path}. Run the pipeline first.")
        return False
    except Exception as e:
        print(f"❌ FAILED: Error parsing JSON - {str(e)}")
        return False

if __name__ == "__main__":
    # When you run `python test_schema.py`, it will check the latest results file.
    validate_output()