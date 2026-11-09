"""
Prompt templates, output schema definitions, and response parsing
for clinical trial criterion eligibility and evidence extraction.
"""

import json
import re
from typing import Dict, Any, List, Optional

# Valid expert label categories as defined by the TrialGPT dataset
INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["not excluded", "excluded", "not enough information", "not applicable"]
ALL_LABELS = list(set(INCLUSION_LABELS + EXCLUSION_LABELS))

SYSTEM_PROMPT = """You are an expert clinical trial screening assistant.
Evaluate whether a patient meets an individual clinical trial eligibility criterion based strictly on the provided patient record.

You must output a valid JSON object with the following three fields:
1. "eligibility": The eligibility category.
   - For an INCLUSION criterion, choose exactly one from: ["included", "not included", "not enough information", "not applicable"].
   - For an EXCLUSION criterion, choose exactly one from: ["not excluded", "excluded", "not enough information", "not applicable"].
2. "evidence_sentences": A JSON list of integer sentence IDs from the patient record that directly support your decision. If no sentences contain relevant evidence, return [].
3. "reasoning": A concise clinical justification based only on the patient record. Do not extrapolate unsupported assumptions.

Output JSON only. Do not wrap in markdown or add conversational preamble."""


def format_criterion_prompt(note: str, criterion_type: str, criterion_text: str) -> str:
    """Format input prompt for decoder model instruction."""
    crit_type_upper = criterion_type.strip().upper()
    user_content = f"""[PATIENT RECORD]
{note.strip()}

[CRITERION TYPE]
{crit_type_upper}

[CRITERION TEXT]
{criterion_text.strip()}

Evaluate whether this patient meets the criterion above. Return your decision as a JSON object with keys "eligibility", "evidence_sentences", and "reasoning"."""
    return user_content


def format_training_target(expert_eligibility: str, expert_sentences: List[int], reasoning: Optional[str] = None) -> str:
    """Format the gold supervision target as valid JSON."""
    if reasoning is None or not reasoning.strip():
        reasoning = f"Criterion determined as '{expert_eligibility}' based on patient sentences {expert_sentences}."
    
    target_obj = {
        "eligibility": expert_eligibility.strip().lower(),
        "evidence_sentences": [int(s) for s in expert_sentences],
        "reasoning": reasoning.strip()
    }
    return json.dumps(target_obj, ensure_ascii=False)


def build_chat_messages(note: str, criterion_type: str, criterion_text: str, target_json: Optional[str] = None) -> List[Dict[str, str]]:
    """Build standard chat messages structure (compatible with Qwen / Llama chat templates)."""
    user_prompt = format_criterion_prompt(note, criterion_type, criterion_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    if target_json is not None:
        messages.append({"role": "assistant", "content": target_json})
    return messages


def parse_model_response(raw_output: str, criterion_type: str) -> Dict[str, Any]:
    """
    Robust JSON parser for generated model outputs.
    Handles potential markdown code fences, trailing commas, or partial formatting.
    """
    cleaned = raw_output.strip()
    # Strip markdown code blocks if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try direct JSON parsing
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return normalize_prediction(data, criterion_type)
    except Exception:
        pass

    # Regex fallback if response has extraneous surrounding text
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                return normalize_prediction(data, criterion_type)
        except Exception:
            pass

    # Heuristic fallback if JSON parsing failed completely
    return extract_heuristics(cleaned, criterion_type)


def normalize_prediction(data: Dict[str, Any], criterion_type: str) -> Dict[str, Any]:
    """Normalize and validate extracted prediction dictionary."""
    raw_eligibility = str(data.get("eligibility", "")).strip().lower()
    
    # Validate against known classes
    if raw_eligibility not in ALL_LABELS:
        # Match closest label
        if "not enough" in raw_eligibility or "insufficient" in raw_eligibility:
            raw_eligibility = "not enough information"
        elif "not applicable" in raw_eligibility:
            raw_eligibility = "not applicable"
        elif "not excluded" in raw_eligibility:
            raw_eligibility = "not excluded"
        elif "excluded" in raw_eligibility:
            raw_eligibility = "excluded"
        elif "not included" in raw_eligibility:
            raw_eligibility = "not included"
        elif "included" in raw_eligibility:
            raw_eligibility = "included"
        else:
            raw_eligibility = "not enough information"

    raw_sentences = data.get("evidence_sentences", [])
    valid_sentences = []
    if isinstance(raw_sentences, list):
        for s in raw_sentences:
            try:
                valid_sentences.append(int(s))
            except (ValueError, TypeError):
                continue
    elif isinstance(raw_sentences, (int, str)):
        try:
            valid_sentences.append(int(raw_sentences))
        except (ValueError, TypeError):
            pass

    reasoning = str(data.get("reasoning", "")).strip()

    return {
        "eligibility": raw_eligibility,
        "evidence_sentences": sorted(list(set(valid_sentences))),
        "reasoning": reasoning
    }


def extract_heuristics(text: str, criterion_type: str) -> Dict[str, Any]:
    """Rule-based extractor for unstructured generations."""
    lower = text.lower()
    if "not enough information" in lower or "insufficient" in lower:
        label = "not enough information"
    elif "not applicable" in lower:
        label = "not applicable"
    elif criterion_type.lower() == "inclusion":
        if "not included" in lower:
            label = "not included"
        elif "included" in lower:
            label = "included"
        else:
            label = "not enough information"
    else:  # exclusion
        if "not excluded" in lower:
            label = "not excluded"
        elif "excluded" in lower:
            label = "excluded"
        else:
            label = "not enough information"

    # Find integer sentences
    sentence_ids = [int(m) for m in re.findall(r"sentence\s*(\d+)|\[(\d+)\]|s(\d+)", lower)]
    flat_ids = []
    for match in re.findall(r"\b\d+\b", text):
        val = int(match)
        if val < 50:  # Sensible upper bound for patient note sentences
            flat_ids.append(val)

    return {
        "eligibility": label,
        "evidence_sentences": sorted(list(set(flat_ids[:5]))),
        "reasoning": text[:200]
    }
