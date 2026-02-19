"""
Speaker Mapping Module - Gemini AI (Local Server Only)

This module ONLY contains speaker mapping functions:
- map_speakers_with_gemini()
- apply_speaker_mapping()
- map_speakers_to_names_gemini()

Uses Gemini AI to intelligently map speaker IDs to participant names.
Runs on your local server, NOT on RunPod.
"""

import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict

try:
    from google import genai
    NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    NEW_SDK = False

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# =============================
# CONFIGURATION
# =============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables!")

# Configure Gemini
if NEW_SDK:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    GEMINI_MODEL = 'gemini-2.0-flash'
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    GEMINI_MODEL = 'gemini-2.0-flash'
    
# Gemini 2.0 Flash Pricing (USD per 1M tokens)
GEMINI_PRICING = {
    "short_context": {"input": 0.10, "output": 0.40},
    "long_context": {"input": 0.20, "output": 0.80}  # Over 128k context
}

def calculate_gemini_cost(usage_metadata):
    """
    Calculates cost based on token counts using Gemini 2.0 Flash pricing.
    
    Args:
        usage_metadata: Response usage metadata from Gemini API
        
    Returns:
        Dict with token counts, cost breakdown, and pricing tier
    """
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    total_tokens = usage_metadata.total_token_count
    
    # Check if long context pricing applies (> 128,000 tokens)
    tier = "long_context" if total_tokens > 128000 else "short_context"
    
    in_cost = (input_tokens / 1_000_000) * GEMINI_PRICING[tier]["input"]
    out_cost = (output_tokens / 1_000_000) * GEMINI_PRICING[tier]["output"]
    total_cost = in_cost + out_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(in_cost, 6),
        "output_cost_usd": round(out_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "tier_used": tier
    }

# =============================
# UTILITIES
# =============================
def format_timestamp(seconds):
    """Converts seconds (float/int) to MM:SS or HH:MM:SS string."""
    try:
        seconds = float(seconds)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return "00:00"

# =============================
# GEMINI-BASED SPEAKER MAPPING
# =============================

def map_speakers_with_gemini(
    diarized_transcript: List[Dict],
    participant_names: List[str]
) -> Dict[str, str]:
    """
    Use Gemini 2.0 Flash to intelligently map SPEAKER_XX to actual participant names.
    
    Args:
        diarized_transcript: Transcript with speaker IDs (SPEAKER_00, SPEAKER_01, etc.)
        participant_names: List of detected participant names from OCR
        
    Returns:
        Dict mapping speaker IDs to names (e.g., {"SPEAKER_00": "John Doe"})
    """
    print(f"\n🤖 Using Gemini to map speakers to names...")
    print(f"   Participants: {participant_names}")
    
    # Get unique speakers
    unique_speakers = sorted(set(seg['speaker'] for seg in diarized_transcript))
    print(f"   Speakers found: {unique_speakers}")
    
    if not participant_names:
        print("   ⚠️  No participant names provided, skipping mapping")
        return {}
    
    # Prepare transcript sample (first 100 segments for context)
    sample_size = min(100, len(diarized_transcript))
    transcript_sample = []
    
    for seg in diarized_transcript[:sample_size]:
        timestamp = format_timestamp(seg['start'])
        transcript_sample.append(f"[{timestamp}] {seg['speaker']}: {seg['text']}")
    
    transcript_text = "\n".join(transcript_sample)
    
    # Build prompt
    prompt = f"""You are analyzing a meeting transcript to identify which speaker label corresponds to which participant.

PARTICIPANT NAMES (detected from video):
{', '.join(participant_names)}

SPEAKER LABELS DETECTED:
{', '.join(unique_speakers)}

TRANSCRIPT SAMPLE (first {sample_size} segments):
{transcript_text}

TASK:
Map each SPEAKER_XX label to the most likely participant name based on:
1. Self-introductions ("I'm [name]", "This is [name]", "My name is [name]")
2. How others address them
3. Speaking patterns and context
4. First-person references

CRITICAL RULES:
- Each speaker label must map to exactly ONE name
- Each name can only be used ONCE
- If unsure about a speaker, leave them unmapped (don't guess)
- Only map speakers you are confident about

Return ONLY a valid JSON object in this EXACT format (no markdown, no code blocks):
{{
  "SPEAKER_00": "Full Name",
  "SPEAKER_01": "Full Name",
  "reasoning": {{
    "SPEAKER_00": "Brief explanation of why this mapping is correct",
    "SPEAKER_01": "Brief explanation of why this mapping is correct"
  }}
}}

If you cannot confidently map a speaker, omit them from the mapping.
Return ONLY the JSON object, nothing else."""

    try:
        # Call Gemini
        print("   Calling Gemini API...")
        if NEW_SDK:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        else:
            response = gemini_model.generate_content(prompt)
        
        # Track tokens with detailed cost report
        if hasattr(response, 'usage_metadata'):
            cost_report = calculate_gemini_cost(response.usage_metadata)
            
            print(f"\n   📊 GEMINI SPEAKER MAPPING - USAGE & COST REPORT")
            print(f"   {'='*60}")
            print(f"   Prompt Tokens:      {cost_report['input_tokens']:,}")
            print(f"   Candidate Tokens:   {cost_report['output_tokens']:,}")
            print(f"   Total Tokens:       {cost_report['total_tokens']:,}")
            print(f"   {'─'*60}")
            print(f"   Pricing Tier:       {cost_report['tier_used']}")
            print(f"   Input Cost:         ${cost_report['input_cost_usd']:.6f}")
            print(f"   Output Cost:        ${cost_report['output_cost_usd']:.6f}")
            print(f"   Total Cost:         ${cost_report['total_cost_usd']:.6f}")
            print(f"   {'='*60}\n")
        else:
            print("   ⚠️  Token usage metadata not available")
        
        # Parse response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        import re
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON
        mapping_result = json.loads(response_text)
        
        # Extract just the mapping (remove reasoning)
        speaker_mapping = {
            k: v for k, v in mapping_result.items() 
            if k.startswith('SPEAKER_') and isinstance(v, str)
        }
        
        # Print reasoning if available
        if 'reasoning' in mapping_result:
            print("\n   Gemini's reasoning:")
            for speaker, reason in mapping_result['reasoning'].items():
                if speaker in speaker_mapping:
                    print(f"   • {speaker} → {speaker_mapping[speaker]}")
                    print(f"     Reason: {reason}")
        
        print(f"\n✅ Gemini mapped {len(speaker_mapping)} speakers:")
        for speaker_id, name in speaker_mapping.items():
            print(f"   {speaker_id} → {name}")
        
        return speaker_mapping
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse Gemini response as JSON: {e}")
        print(f"   Response was: {response_text[:500]}")
        return {}
    except Exception as e:
        print(f"   ❌ Gemini mapping failed: {e}")
        logger.error(f"Gemini mapping error: {e}", exc_info=True)
        return {}


def apply_speaker_mapping(
    diarized_transcript: List[Dict],
    speaker_mapping: Dict[str, str]
) -> List[Dict]:
    """
    Apply speaker-to-name mapping to the transcript.
    
    Args:
        diarized_transcript: Transcript with speaker IDs
        speaker_mapping: Dict mapping speaker IDs to names
        
    Returns:
        Updated transcript with names replacing speaker IDs
    """
    if not speaker_mapping:
        print("   ⚠️  No speaker mapping provided, keeping speaker IDs")
        return diarized_transcript
    
    mapped_count = 0
    unmapped_speakers = set()
    
    for seg in diarized_transcript:
        original_speaker = seg['speaker']
        if original_speaker in speaker_mapping:
            seg['speaker'] = speaker_mapping[original_speaker]
            mapped_count += 1
        else:
            unmapped_speakers.add(original_speaker)
    
    print(f"\n✅ Applied mapping to {mapped_count}/{len(diarized_transcript)} segments")
    
    if unmapped_speakers:
        print(f"   ⚠️  Unmapped speakers (keeping IDs): {sorted(unmapped_speakers)}")
    
    return diarized_transcript


def map_speakers_to_names_gemini(
    diarized_transcript: List[Dict],
    participant_names: List[str]
) -> List[Dict]:
    """
    Main function: Map speakers to names using Gemini AI.
    
    Args:
        diarized_transcript: Transcript with speaker IDs
        participant_names: List of participant names from OCR
        
    Returns:
        Updated transcript with mapped speaker names
    """
    print("\n" + "="*60)
    print("GEMINI-BASED SPEAKER MAPPING")
    print("="*60)
    
    # Step 1: Use Gemini to create mapping
    speaker_mapping = map_speakers_with_gemini(diarized_transcript, participant_names)
    
    # Step 2: Apply mapping to transcript
    updated_transcript = apply_speaker_mapping(diarized_transcript, speaker_mapping)
    
    print("="*60 + "\n")
    
    return updated_transcript
