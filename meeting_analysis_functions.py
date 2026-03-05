import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from google import genai
    NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    NEW_SDK = False

# =============================
# CONFIGURATION
# =============================

API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not set!\n"
        "Set it in your .env file: GOOGLE_API_KEY=your-api-key-here"
    )

if NEW_SDK:
    gemini_client = genai.Client(api_key=API_KEY)
    GEMINI_MODEL = 'gemini-2.0-flash'
else:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    GEMINI_MODEL = 'gemini-2.0-flash'

PRICING = {
    "short_context": {"input": 0.10, "output": 0.40},
    "long_context": {"input": 0.20, "output": 0.80}
}

# =============================
# UTILITY FUNCTIONS
# =============================

def format_timestamp(seconds) -> str:
    """Convert seconds to MM:SS format"""
    try:
        seconds = int(float(seconds))
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "00:00"

def calculate_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    total_tokens = input_tokens + output_tokens
    tier = "long_context" if total_tokens > 128000 else "short_context"
    input_cost = (input_tokens / 1_000_000) * PRICING[tier]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[tier]["output"]
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": total_tokens, "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6), "total_cost_usd": round(input_cost + output_cost, 6),
        "tier": tier
    }

def safe_json_parse(text: str) -> Dict[str, Any]:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON response: {text[:200]}...")

# =============================
# MAIN ANALYZER CLASS
# =============================

class MeetingAnalyzer:
    def __init__(self, transcript_file: str, participant_names: List[str] = None):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            self.transcript_data = json.load(f)
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = []
        
        if participant_names:
            self.participants = sorted(list(participant_names))
        else:
            self.participants = self._extract_participants()
        
        self.full_transcript = self._prepare_transcript()
    
    def _extract_participants(self) -> List[str]:
        participants = set()
        for entry in self.transcript_data:
            speaker = entry.get('speaker', 'UNKNOWN')
            if speaker and speaker != 'UNKNOWN':
                participants.add(speaker)
        return sorted(list(participants))
    
    def _prepare_transcript(self) -> str:
        lines = []
        for entry in self.transcript_data:
            timestamp = format_timestamp(entry.get('start', 0))
            speaker = entry.get('speaker', 'UNKNOWN')
            text = entry.get('text', '').strip()
            lines.append(f"[{timestamp}] {speaker}: {text}")
        return "\n".join(lines)
    
    def _call_gemini(self, prompt: str, call_name: str) -> str:
        print(f"\n🤖 API Call: {call_name}")
        if NEW_SDK:
            response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        else:
            response = gemini_model.generate_content(prompt)
        
        if hasattr(response, 'usage_metadata'):
            cost_data = calculate_cost(
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count
            )
            self.total_input_tokens += cost_data['input_tokens']
            self.total_output_tokens += cost_data['output_tokens']
            self.api_calls.append({'name': call_name, **cost_data})
        return response.text
    
    # **[CHANGED]: Refactored to return EXACTLY "engagement_score" and "participation_score"**
    def _calculate_engagement_metrics(self) -> Dict[str, float]:
        total_speaking_time = 0
        speaker_times = set()
        
        for entry in self.transcript_data:
            try:
                start = float(entry.get('start', 0))
                end = float(entry.get('end', start + 3))
                duration = end - start
                total_speaking_time += duration
                
                speaker = entry.get('speaker', 'UNKNOWN')
                if speaker != 'UNKNOWN':
                    speaker_times.add(speaker)
            except (ValueError, TypeError):
                continue
        
        total_duration = max(float(entry.get('end', 0)) for entry in self.transcript_data) if self.transcript_data else 0
        
        engagement_rate = (total_speaking_time / total_duration * 100) if total_duration > 0 else 0.0
        # Participation Score = % of identified participants who actually spoke
        participation_rate = (len(speaker_times) / len(self.participants) * 100) if self.participants else 0.0
        
        return {
            "engagement_score": round(min(engagement_rate, 100.0), 2),
            "participation_score": round(min(participation_rate, 100.0), 2)
        }
    
    # **[CHANGED]: Rewritten prompt to force exact `summary` and `ai_analysis` Schema keys**
    def analyze_summary_and_analysis(self) -> Dict[str, Any]:
        prompt = f"""You are analyzing a business meeting transcript.

PARTICIPANTS: {', '.join(self.participants)}
TRANSCRIPT:
{self.full_transcript}

Provide a comprehensive analysis in this EXACT JSON format (no markdown):
{{
  "summary": "A detailed 3-4 paragraph summary of the entire meeting, decisions, and outcomes.",
  "ai_analysis": {{
    "meeting_effectiveness": "Evaluate how effective the meeting was and why.",
    "decision_clarity": "Were decisions clearly made? Explain.",
    "participant_alignment": "Were participants aligned or in conflict? Explain.",
    "open_questions": "List any questions that remained unanswered.",
    "risks": "Identify any project or business risks mentioned.",
    "ambiguity": "Identify areas of the conversation that lacked ownership or clarity.",
    "repeated_topics": "Identify any topics that went in circles or were repeated."
  }}
}}
"""
        return safe_json_parse(self._call_gemini(prompt, "Summary & AI Analysis"))
    
    # **[CHANGED]: Rewritten prompt to extract exact Action Items and map timestamp Categories**
    def analyze_actions_and_categories(self) -> Dict[str, Any]:
        prompt = f"""Analyze this meeting transcript for action items and categorize key moments.

PARTICIPANTS: {', '.join(self.participants)}
TRANSCRIPT:
{self.full_transcript}

Provide complete analysis in this EXACT JSON format (no markdown):
{{
  "action_items": [
    {{
      "task": "Specific action to be taken",
      "owner": "Person responsible",
      "deadline": "Specific deadline or 'Not specified'"
    }}
  ],
  "transcript_categories": [
    {{
      "timestamp": "MM:SS",
      "category": "action" 
    }}
  ]
}}

CRITICAL INSTRUCTIONS FOR 'transcript_categories':
- You must map important timestamps to one of these EXACT string values: "action", "decision", or "question".
- Ignore general discussion lines. Only return timestamps that are strictly actions, decisions, or questions.
"""
        return safe_json_parse(self._call_gemini(prompt, "Actions & Transcript Categories"))
    
    def analyze(self) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("MEETING INTELLIGENCE ANALYSIS")
        print("="*70)
        
        # 1. API Calls
        core_data = self.analyze_summary_and_analysis()
        actions_data = self.analyze_actions_and_categories()
        
        # 2. Local Metrics
        metrics_data = self._calculate_engagement_metrics()
        
        # **[NEW]: Stitching the transcript array locally to guarantee Phase 1 format**
        # Default all lines to "discussion". Overwrite if Gemini flagged it as an action/decision/question.
        category_map = {item.get('timestamp'): item.get('category', 'discussion') 
                        for item in actions_data.get('transcript_categories', [])}
        
        final_transcript = []
        for entry in self.transcript_data:
            ts = format_timestamp(entry.get('start', 0))
            # Fallback to 'discussion' if Gemini gave a weird category
            raw_cat = category_map.get(ts, "discussion").lower()
            safe_cat = raw_cat if raw_cat in ["action", "decision", "question", "discussion"] else "discussion"
            
            final_transcript.append({
                "timestamp": ts,
                "speaker": entry.get('speaker', 'UNKNOWN'),
                "text": entry.get('text', '').strip(),
                "category": safe_cat
            })

        # **[NEW]: Constructing the final rigid JSON matching `test_schema.py`**
        complete_analysis = {
            "metrics": metrics_data,
            "participants": self.participants,
            "summary": core_data.get("summary", "No summary provided."),
            "action_items": actions_data.get("action_items", []),
            "ai_analysis": core_data.get("ai_analysis", {
                "meeting_effectiveness": "Data missing",
                "decision_clarity": "Data missing",
                "participant_alignment": "Data missing",
                "open_questions": "Data missing",
                "risks": "Data missing",
                "ambiguity": "Data missing",
                "repeated_topics": "Data missing"
            }),
            "transcript": final_transcript
        }
        
        self._print_cost_summary()
        return complete_analysis
    
    def _print_cost_summary(self):
        print("\n" + "="*70)
        print("💰 COST SUMMARY")
        print("="*70)
        total_cost = sum(call['total_cost_usd'] for call in self.api_calls)
        for i, call in enumerate(self.api_calls, 1):
            print(f"\nCall {i}: {call['name']}")
            print(f"  Total:  {call['total_tokens']:,} tokens → ${call['total_cost_usd']:.6f}")
        print(f"\nGRAND TOTAL: ${total_cost:.6f}")
        print("="*70)
    
    def save(self, output_file: str = "results/final_output.json") -> Dict[str, Any]:
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        analysis = self.analyze()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Analysis saved to: {output_file}")
        return analysis