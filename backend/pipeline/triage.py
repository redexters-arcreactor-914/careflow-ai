"""
Triage scoring (escalate-only fusion)

    urgency = max(rule_floor, model_proposal)


"""

import json
import re
import requests
from backend.config import (
    RED_FLAG_KEYWORDS, SLM_PROVIDER, SLM_BASE_URL, SLM_MODEL, URGENCY_LEVELS,
    INPUT_CONFIDENCE_FLOOR,
)
from backend.models.schemas import Flag, TriageResult

RANK = {lvl: i for i, lvl in enumerate(URGENCY_LEVELS)}
SLM_TIMEOUT_SECONDS = 60
FAILURE_URGENCY = "high"  # elevated priority when the model cannot be trusted


def check_red_flags(text: str) -> list[str]:
    lowered = text.lower()
    return [kw for kw in RED_FLAG_KEYWORDS if kw in lowered]


def fuse(rule_floor: str, model_proposal: str) -> str:
    """Escalate-only: the result is never below the rule floor."""
    return max(rule_floor, model_proposal, key=lambda u: RANK[u])


def parse_slm_output(raw: str) -> dict | None:
    """Extract {"urgency","reason"} from messy model output. None if unusable."""
    match = re.search(r"\{.*?\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    urgency = str(data.get("urgency", "")).strip().lower()
    if urgency not in RANK:
        return None
    return {"urgency": urgency, "reason": str(data.get("reason", "")).strip()[:300]}


def call_slm_for_triage(text: str, flags: list[Flag]) -> dict | None:
    """Returns {"urgency","reason"} or None when the model is unavailable/unusable."""
    if SLM_PROVIDER == "none":
        return None
    prompt = f"""You are assisting a healthcare triage dashboard. You do NOT diagnose.
Given this patient-reported text, suggest an urgency level (low, medium, high)
and a one-sentence reason. Respond ONLY as JSON: {{"urgency": "...", "reason": "..."}}

Patient text: {text}
Known flags on this case: {[f.description for f in flags]}
"""
    
    for attempt in (1, 2):
        try:
            response = requests.post(
                f"{SLM_BASE_URL}/api/generate",
                json={"model": SLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                timeout=SLM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            raw = response.json().get("response", "")
            parsed = parse_slm_output(raw)
            if parsed is not None:
                return parsed
            print(f"[triage] SLM output unusable (attempt {attempt}): {raw[:200]!r}")
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"[triage] SLM call failed (attempt {attempt}): {type(e).__name__}: {e}")
    return None


def apply_uncertainty_gate(result: TriageResult, source: str, input_confidence: float) -> TriageResult:
    """Ask, do not guess: if a voice transcript or OCR read is low confidence, the case
    cannot be ranked low on that text. Escalate-only: raise to at least medium and
    require manual review."""
    if source == "typed_text" or input_confidence >= INPUT_CONFIDENCE_FLOOR:
        return result
    result.urgency_level = fuse(result.urgency_level, "medium")
    result.needs_manual_review = True
    result.urgency_reason = (f"Input was read with low confidence ({input_confidence:.0%}); "
                             f"manual review required. {result.urgency_reason}")
    return result


def rule_floor_for(raw_text: str, flags: list[Flag]) -> tuple[str, str]:
    hits = check_red_flags(raw_text)
    if hits:
        return "critical", f"Red-flag keyword(s): {', '.join(hits)}."
    if any(f.severity == "critical" for f in flags):
        return "high", "Unresolved critical-severity flag(s) on this case."
    return "low", "No rule triggered."


def score_case(case_id: str, raw_text: str, flags: list[Flag]) -> TriageResult:
    floor, floor_reason = rule_floor_for(raw_text, flags)

    # Fast path: nothing the model says can lower a critical floor.
    if floor == "critical":
        return TriageResult(
            case_id=case_id, urgency_level="critical",
            urgency_reason=f"{floor_reason} Escalate immediately; the model cannot lower this.",
            scored_by="rule_based", confidence=0.95, rule_floor=floor,
            suggested_followup_questions=[
                "Confirm current symptom status with the patient directly.",
                "Has emergency care already been sought?",
            ],
        )

    proposal = call_slm_for_triage(raw_text, flags)

    if proposal is None:
        
        urgency = fuse(floor, FAILURE_URGENCY)
        return TriageResult(
            case_id=case_id, urgency_level=urgency,
            urgency_reason="Model unavailable or output unusable. Routed to manual "
                           f"review at elevated priority. {floor_reason}",
            scored_by="rule_based", confidence=0.5, rule_floor=floor,
            needs_manual_review=True,
        )

    urgency = fuse(floor, proposal["urgency"])
    raised = RANK[urgency] > RANK[floor]
    return TriageResult(
        case_id=case_id, urgency_level=urgency,
        urgency_reason=proposal["reason"] or floor_reason,
        scored_by="rule_based+slm" if raised or floor != "low" else "slm",
        confidence=0.6,  # capped: not trusted without human review
        rule_floor=floor, model_proposal=proposal["urgency"],
        suggested_followup_questions=[
            "Ask the patient to describe how the symptom has changed in the last 24 hours.",
        ],
    )