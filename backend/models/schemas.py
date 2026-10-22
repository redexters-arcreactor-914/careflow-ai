"""
Shared data shapes used across the pipeline. Every stage reads/writes
these so the FastAPI routes, the pipeline modules, and the frontend
dashboard all agree on one contract.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class PatientInput(BaseModel):
    """Raw input as it arrives — before any processing."""
    case_id: str
    patient_consent_given: bool = False
    input_type: Literal["text", "voice", "report_image", "mixed"]
    raw_text: Optional[str] = None          # typed symptom description
    voice_file_path: Optional[str] = None   # path to uploaded audio
    report_image_path: Optional[str] = None # path to uploaded report image
    declared_language: Optional[str] = None # if patient/staff specifies it
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractedField(BaseModel):
    """One piece of structured info, always tagged with where it came from
    and how confident we are — this is what powers the 'link back to source'
    requirement in the PS."""
    field_name: str
    value: str
    source: Literal["typed_text", "voice_transcript", "ocr_report"]
    confidence: float  # 0.0 - 1.0
    original_language: Optional[str] = None
    translated: bool = False


class Flag(BaseModel):
    """A single thing that needs human attention before this case is trusted."""
    flag_type: Literal[
        "missing_info", "low_confidence", "conflicting_info",
        "unclear_input", "suspicious_instruction", "red_flag_symptom"
    ]
    field_name: Optional[str] = None
    description: str
    severity: Literal["info", "warning", "critical"] = "warning"


class TriageResult(BaseModel):
    model_config = {"protected_namespaces": ()}
    case_id: str
    urgency_level: Literal["low", "medium", "high", "critical"]
    urgency_reason: str
    scored_by: Literal["rule_based", "slm", "rule_based+slm"]
    confidence: float
    suggested_followup_questions: list[str] = []
    # Escalate-only fusion audit fields: urgency = max(rule_floor, model_proposal)
    rule_floor: Optional[Literal["low", "medium", "high", "critical"]] = None
    model_proposal: Optional[Literal["low", "medium", "high", "critical"]] = None
    needs_manual_review: bool = False


class PatientCase(BaseModel):
    """The fully assembled record a reviewer sees on the dashboard."""
    case_id: str
    input: PatientInput
    extracted_fields: list[ExtractedField] = []
    flags: list[Flag] = []
    triage: Optional[TriageResult] = None
    review_status: Literal["pending", "approved", "edited", "escalated"] = "pending"
    reviewer_notes: Optional[str] = None


class AuditEntry(BaseModel):
    case_id: str
    actor: Literal["system", "ai", "reviewer"]
    action: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[str] = None
