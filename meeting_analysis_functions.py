"""
Optimized Meeting Analysis Module with Gemini 2.0 Flash

Key Optimizations:
1. Batched API calls - 2 calls instead of 5+ (reduces cost by ~60%)
2. Smart prompting - get all related data in one call
3. Structured JSON output matching UI design
4. Timestamp format: MM:SS only (no raw seconds)
5. Proper participant tracking with names
6. Cost tracking and reporting

Architecture:
- Call 1: Summary + Topics + Key Moments + Decisions
- Call 2: Action Items + Next Steps + AI Insights
- Local: Engagement metrics (no API cost)
"""

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

# Configure Gemini
if NEW_SDK:
    gemini_client = genai.Client(api_key=API_KEY)
    GEMINI_MODEL = 'gemini-2.0-flash'
else:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    GEMINI_MODEL = 'gemini-2.0-flash'
# Gemini 2.0 Flash Pricing (USD per 1M tokens)
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
    """Calculate Gemini API cost"""
    total_tokens = input_tokens + output_tokens
    tier = "long_context" if total_tokens > 128000 else "short_context"
    
    input_cost = (input_tokens / 1_000_000) * PRICING[tier]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[tier]["output"]
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "tier": tier
    }


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini response, handling markdown blocks"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
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
    """
    Optimized meeting analyzer with structured output
    
    Features:
    - 2 API calls (vs 5+ in legacy version)
    - Structured JSON matching UI design
    - MM:SS timestamps only
    - Comprehensive cost tracking
    - Participant name extraction
    """
    
    def __init__(self, transcript_file: str, participant_names: List[str] = None):
        """
        Initialize analyzer
        
        Args:
            transcript_file: Path to JSON transcript with format:
                [{"start": 0.0, "end": 5.2, "speaker": "Name", "text": "..."}]
            participant_names: Optional list of participant names from pipeline (e.g., OCR/Vision detection).
                              If provided, these will be used instead of extracting from transcript.
                              This ensures clean participant lists without SPEAKER_XX or UNKNOWN entries.
        """
        with open(transcript_file, 'r', encoding='utf-8') as f:
            self.transcript_data = json.load(f)
        
        # Track API usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.api_calls = []
        
        # Use provided participant names OR extract from transcript
        if participant_names:
            # Use the clean participant names from pipeline (OCR/Vision detection)
            self.participants = sorted(list(participant_names))
            print(f"   Using {len(self.participants)} participants from pipeline: {', '.join(self.participants)}")
        else:
            # Fallback: extract from transcript (may include SPEAKER_XX, UNKNOWN)
            self.participants = self._extract_participants()
            print(f"   Extracted {len(self.participants)} participants from transcript")
        
        # Prepare transcript text
        self.full_transcript = self._prepare_transcript()
    
    
    def _extract_participants(self) -> List[str]:
        """Extract unique participant names from transcript"""
        participants = set()
        for entry in self.transcript_data:
            speaker = entry.get('speaker', 'UNKNOWN')
            if speaker and speaker != 'UNKNOWN':
                participants.add(speaker)
        return sorted(list(participants))
    
    
    def _prepare_transcript(self) -> str:
        """Convert transcript to formatted text"""
        lines = []
        for entry in self.transcript_data:
            timestamp = format_timestamp(entry.get('start', 0))
            speaker = entry.get('speaker', 'UNKNOWN')
            text = entry.get('text', '').strip()
            lines.append(f"[{timestamp}] {speaker}: {text}")
        return "\n".join(lines)
    
    
    def _call_gemini(self, prompt: str, call_name: str) -> str:
        """Make Gemini API call and track usage"""
        print(f"\n🤖 API Call: {call_name}")
        
        if NEW_SDK:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        else:
            response = gemini_model.generate_content(prompt)
        
        # Track tokens
        if hasattr(response, 'usage_metadata'):
            cost_data = calculate_cost(
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count
            )
            
            self.total_input_tokens += cost_data['input_tokens']
            self.total_output_tokens += cost_data['output_tokens']
            self.api_calls.append({
                'name': call_name,
                **cost_data
            })
            
            print(f"   Tokens: {cost_data['input_tokens']:,} in / {cost_data['output_tokens']:,} out")
            print(f"   Cost: ${cost_data['total_cost_usd']:.6f} ({cost_data['tier']})")
        
        return response.text
    
    
    def _calculate_engagement_metrics(self) -> Dict[str, Any]:
        """Calculate engagement metrics locally (no API cost)"""
        total_speaking_time = 0
        speaker_times = {}
        
        for entry in self.transcript_data:
            try:
                start = float(entry.get('start', 0))
                end = float(entry.get('end', start + 3))
                duration = end - start
                
                total_speaking_time += duration
                
                speaker = entry.get('speaker', 'UNKNOWN')
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                
            except (ValueError, TypeError):
                continue
        
        # Calculate total meeting duration
        total_duration = max(
            float(entry.get('end', 0)) for entry in self.transcript_data
        ) if self.transcript_data else 0
        
        engagement_rate = (total_speaking_time / total_duration * 100) if total_duration > 0 else 0
        
        # Per-speaker stats
        speaker_stats = []
        for speaker, time in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (time / total_speaking_time * 100) if total_speaking_time > 0 else 0
            speaker_stats.append({
                "name": speaker,
                "speaking_time_seconds": round(time, 2),
                "percentage_of_total": round(percentage, 2)
            })
        
        return {
            "total_speaking_time_seconds": round(total_speaking_time, 2),
            "total_meeting_duration_seconds": round(total_duration, 2),
            "engagement_rate_percentage": round(engagement_rate, 2),
            "speaker_breakdown": speaker_stats
        }
    
    
    def analyze_core_content(self) -> Dict[str, Any]:
        """
        API CALL 1: Summary + Topics + Key Moments + Decisions
        
        This is the main content analysis that covers:
        - Meeting summary and context
        - Topics discussed
        - Key moments/highlights
        - Confirmed decisions
        """
        
        prompt = f"""You are analyzing a business meeting transcript to extract structured insights.

PARTICIPANTS:
{', '.join(self.participants)}

TRANSCRIPT:
{self.full_transcript}

Provide a comprehensive analysis in this EXACT JSON format (no markdown, no code blocks):

{{
  "meeting_summary": {{
    "overview": "2-3 sentence executive summary of the meeting",
    "meeting_type": "sales call / team standup / client meeting / strategy session / etc",
    "duration_minutes": estimated_duration,
    "participant_count": {len(self.participants)},
    "participant_names": {json.dumps(self.participants)},
    "key_points": [
      "Most important point from the meeting",
      "Second most important point",
      "Third key point",
      "Fourth key point",
      "Fifth key point"
    ]
  }},
  "topics_discussed": [
    {{
      "topic": "Main topic name",
      "description": "What was discussed about this topic",
      "time_range": "MM:SS-MM:SS when discussed",
      "duration_minutes": estimated_minutes,
      "key_speakers": ["Speaker 1", "Speaker 2"],
      "importance": "high / medium / low"
    }}
  ],
  "key_moments": [
    {{
      "timestamp": "MM:SS",
      "speaker": "Person name",
      "moment": "Description of what happened",
      "significance": "Why this moment matters",
      "category": "decision / insight / concern / opportunity / milestone"
    }}
  ],
  "decisions_made": [
    {{
      "decision": "What was decided",
      "made_by": "Decision maker name",
      "timestamp": "MM:SS",
      "rationale": "Why this decision was made",
      "impact": "Expected impact or consequence",
      "action_required": "What needs to be done to implement"
    }}
  ]
}}

CRITICAL REQUIREMENTS:
- All timestamps in MM:SS format (never show raw seconds)
- Participant names must match exactly as shown above
- Include 3-5 topics minimum
- Include 4-8 key moments minimum
- Be specific and detailed, avoid generic statements
- Extract actual quotes when highlighting key moments
"""
        
        response_text = self._call_gemini(prompt, "Core Content Analysis")
        return safe_json_parse(response_text)
    
    
    def analyze_actions_and_insights(self) -> Dict[str, Any]:
        """
        API CALL 2: Action Items + Next Steps + AI Insights
        
        Covers:
        - Action items with assignments
        - Next steps and timeline
        - AI-generated business insights
        """
        
        prompt = f"""Analyze this meeting transcript for actionable items and business insights.

PARTICIPANTS:
{', '.join(self.participants)}

TRANSCRIPT:
{self.full_transcript}

Provide complete analysis in this JSON format:

{{
  "action_items": [
    {{
      "action": "Specific action to be taken",
      "assigned_to": "Person or team responsible",
      "deadline": "Specific deadline or 'Not specified'",
      "priority": "high / medium / low",
      "context": "Why this action is needed",
      "timestamp": "MM:SS when mentioned",
      "dependencies": ["Any blocking items or prerequisites"]
    }}
  ],
  "next_steps": [
    {{
      "step": "What needs to happen next",
      "owner": "Person responsible",
      "timeline": "Expected timeframe",
      "timestamp": "MM:SS when discussed",
      "details": [
        "Specific detail or requirement",
        "Another detail or consideration"
      ]
    }}
  ],
  "ai_insights": {{
    "overview": "2-3 sentence summary of key business insights",
    "opportunities": [
      {{
        "opportunity": "Business opportunity identified",
        "potential_value": "Expected benefit or value",
        "requirements": ["What's needed to capture this"],
        "risk_level": "low / medium / high"
      }}
    ],
    "challenges": [
      {{
        "challenge": "Challenge or risk identified",
        "impact": "Potential impact if not addressed",
        "mitigation": "Suggested mitigation strategy",
        "severity": "low / medium / high"
      }}
    ],
    "client_sentiment": {{
      "overall_tone": "positive / neutral / negative / mixed",
      "interest_level": "high / medium / low",
      "engagement_indicators": [
        "Specific indicator of engagement (e.g., asked 5 questions)",
        "Another engagement signal"
      ],
      "concerns_raised": [
        "Concern 1 with timestamp",
        "Concern 2 with timestamp"
      ],
      "positive_signals": [
        "Positive signal 1 with timestamp",
        "Positive signal 2 with timestamp"
      ]
    }},
    "strategic_recommendations": [
      "Recommendation 1 based on meeting context",
      "Recommendation 2 for follow-up",
      "Recommendation 3 for next steps"
    ]
  }}
}}

REQUIREMENTS:
- All timestamps in MM:SS format
- Be specific with action items - include WHO, WHAT, WHEN
- Identify real business insights, not generic observations
- Client sentiment analysis should be evidence-based
"""
        
        response_text = self._call_gemini(prompt, "Actions & Insights Analysis")
        return safe_json_parse(response_text)
    
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run complete analysis pipeline
        
        Returns:
            Complete structured analysis matching UI design
        """
        print("\n" + "="*70)
        print("MEETING INTELLIGENCE ANALYSIS")
        print("="*70)
        print(f"Participants: {len(self.participants)}")
        print(f"Names: {', '.join(self.participants)}")
        print(f"Transcript entries: {len(self.transcript_data)}")
        
        # API Call 1: Core content
        print("\n--- Phase 1: Core Content ---")
        core_content = self.analyze_core_content()
        
        # API Call 2: Actions and insights
        print("\n--- Phase 2: Actions & Insights ---")
        actions_insights = self.analyze_actions_and_insights()
        
        # Local calculation: Engagement
        print("\n--- Phase 3: Engagement Metrics (Local) ---")
        engagement = self._calculate_engagement_metrics()
        print("✓ Calculated locally (no API cost)")
        
        # Combine results
        complete_analysis = {
            # Core sections
            "meeting_summary": core_content.get("meeting_summary", {}),
            "participants": {
                "count": len(self.participants),
                "names": self.participants
            },
            
            # Content analysis
            "topics_discussed": core_content.get("topics_discussed", []),
            "key_moments": core_content.get("key_moments", []),
            "decisions_made": core_content.get("decisions_made", []),
            
            # Action items
            "action_items": actions_insights.get("action_items", []),
            "next_steps": actions_insights.get("next_steps", []),
            
            # AI insights
            "ai_insights": actions_insights.get("ai_insights", {}),
            
            # Metrics
            "engagement_metrics": engagement,
            
            # Metadata
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": GEMINI_MODEL,
                "transcript_entries": len(self.transcript_data),
                "api_calls_made": len(self.api_calls)
            }
        }
        
        # Print cost summary
        self._print_cost_summary()
        
        return complete_analysis
    
    
    def _print_cost_summary(self):
        """Print detailed cost breakdown"""
        print("\n" + "="*70)
        print("💰 COST SUMMARY")
        print("="*70)
        
        total_cost = sum(call['total_cost_usd'] for call in self.api_calls)
        
        for i, call in enumerate(self.api_calls, 1):
            print(f"\nCall {i}: {call['name']}")
            print(f"  Input:  {call['input_tokens']:,} tokens → ${call['input_cost_usd']:.6f}")
            print(f"  Output: {call['output_tokens']:,} tokens → ${call['output_cost_usd']:.6f}")
            print(f"  Total:  {call['total_tokens']:,} tokens → ${call['total_cost_usd']:.6f}")
            print(f"  Tier:   {call['tier']}")
        
        print(f"\n{'─'*70}")
        print(f"GRAND TOTAL: ${total_cost:.6f}")
        print(f"Total API Calls: {len(self.api_calls)}")
        print(f"Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}")
        print("="*70)
    
    
    def save(self, output_file: str = "meeting_analysis.json") -> Dict[str, Any]:
        """
        Run analysis and save to file
        
        Args:
            output_file: Output JSON file path
            
        Returns:
            Complete analysis dictionary
        """
        analysis = self.analyze()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis saved to: {output_file}")
        return analysis

