"""
FastAPI entrypoint. Run with:
    uvicorn backend.main:app --reload --port 8000


"""

import os
import pathlib
import re
import uuid
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.config import ALLOWED_AUDIO_EXT, ALLOWED_IMAGE_EXT, MAX_UPLOAD_MB, UPLOAD_DIR
from backend.models.schemas import AuditEntry, Flag, PatientCase, PatientInput
from backend.pipeline import intake, language, extraction, flagging, triage, audit

app = FastAPI(title="CareFlow AI")

# Dashboard runs on a different port/file during dev — open CORS for now,
# tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for the hackathon demo. Swap for a real DB before
# anything resembling production use.
CASES: dict[str, PatientCase] = {}


def _process_case(patient_input: PatientInput) -> PatientCase:
    """Stage 1 to Stage 5 for one case. Shared by the JSON and the upload routes."""
    try:
        intake.validate_intake(patient_input)
    except (intake.ConsentNotGivenError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    cid = patient_input.case_id
    audit.log_action(AuditEntry(case_id=cid, actor="system", action="intake_received"))

    # Stage 1b: get raw text regardless of input type, with source and confidence
    extra_flags: list[Flag] = []
    source, base_conf = "typed_text", 0.95
    if patient_input.input_type == "voice":
        asr_lang = {"hi": "hi", "mixed": "hi", "en": "en"}.get((patient_input.declared_language or "").lower())
        raw_text, base_conf = intake.transcribe_voice(patient_input.voice_file_path, asr_lang)
        source = "voice_transcript"
        failed_msg = "Voice note could not be transcribed. Review the original audio."
    elif patient_input.input_type == "report_image":
        raw_text, base_conf = extraction.ocr_report_image(patient_input.report_image_path)
        source = "ocr_report"
        failed_msg = "Report image could not be read. Review the original image."
    else:
        raw_text = patient_input.raw_text or ""
        failed_msg = None

    if failed_msg and not raw_text.strip():
        extra_flags.append(Flag(flag_type="unclear_input", description=failed_msg, severity="critical"))
        raw_text = ""
    audit.log_action(AuditEntry(
        case_id=cid, actor="ai", action="input_converted",
        details=f"source={source} confidence={base_conf:.2f} chars={len(raw_text)}",
    ))

    # Stage 2: language
    lang_result = language.process_language(raw_text)
    audit.log_action(AuditEntry(
        case_id=cid, actor="ai", action="language_processed",
        details=f"detected={lang_result['detected_language']}",
    ))

    # Stage 3: extraction
    extracted_fields = extraction.extract_symptom_timeline(
        lang_result["translated_text"], source=source, base_confidence=base_conf)
    missing_fields = extraction.identify_missing_fields(extracted_fields)

    # Stage 4: flagging
    flags = flagging.run_all_flags(extracted_fields, missing_fields, raw_text) + extra_flags
    audit.log_action(AuditEntry(
        case_id=cid, actor="ai", action="flags_generated", details=f"{len(flags)} flag(s)",
    ))

    # Stage 5: triage scoring (escalate-only fusion)
    triage_result = triage.score_case(cid, raw_text, flags)
    triage_result = triage.apply_uncertainty_gate(triage_result, source, base_conf)
    audit.log_action(AuditEntry(
        case_id=cid, actor="ai", action="triage_scored", details=f"urgency={triage_result.urgency_level}",
    ))

    # Show the reviewer what the system actually read (transcript or OCR text)
    if raw_text and patient_input.raw_text != raw_text:
        patient_input = patient_input.model_copy(update={"raw_text": raw_text})

    case = PatientCase(
        case_id=cid, input=patient_input, extracted_fields=extracted_fields,
        flags=flags, triage=triage_result, review_status="pending",
    )
    CASES[case.case_id] = case
    return case


@app.post("/cases/intake", response_model=PatientCase)
def submit_case(patient_input: PatientInput):
    """JSON intake (text, or server-side file paths)."""
    return _process_case(patient_input)


@app.post("/cases/upload", response_model=PatientCase)
async def upload_case(
    consent: bool = Form(False),
    input_type: str = Form("text"),
    text: str = Form(""),
    language: str = Form(""),
    case_id: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    """Dashboard intake: text, voice note, or report image, with explicit consent."""
    if input_type not in ("text", "voice", "report_image"):
        raise HTTPException(status_code=400, detail="input_type must be text, voice or report_image")
    cid = (case_id.strip() or f"case-{uuid.uuid4().hex[:8]}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", cid):
        raise HTTPException(status_code=400, detail="case_id may only use letters, digits, - and _")

    voice_path = image_path = None
    if input_type in ("voice", "report_image"):
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="A file is required for this input type.")
        ext = pathlib.Path(file.filename).suffix.lower()
        allowed = ALLOWED_AUDIO_EXT if input_type == "voice" else ALLOWED_IMAGE_EXT
        if ext not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type {ext!r}.")
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
        if not consent:   # never store a file without consent
            raise HTTPException(status_code=400, detail="Consent is required before any file is stored.")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        saved = os.path.join(UPLOAD_DIR, f"{cid}{ext}")      # server-chosen name, never the client's
        with open(saved, "wb") as fh:
            fh.write(data)
        voice_path, image_path = (saved, None) if input_type == "voice" else (None, saved)

    patient_input = PatientInput(
        case_id=cid, patient_consent_given=consent, input_type=input_type,
        raw_text=text or None, voice_file_path=voice_path, report_image_path=image_path,
        declared_language=language or None,
    )
    # Whisper/OCR are slow and blocking: run off the event loop so the API stays responsive
    return await run_in_threadpool(_process_case, patient_input)


AGING_POINTS_PER_HOUR = 5   # waiting cases creep upward so none starve
URGENCY_BASE = {"critical": 100, "high": 70, "medium": 40, "low": 10}


def priority_score(case: PatientCase) -> float:
    """Urgency base + aging (capped so aging never outranks a higher urgency tier)."""
    base = URGENCY_BASE.get(case.triage.urgency_level if case.triage else "low", 10)
    waited_h = max(0.0, (datetime.utcnow() - case.input.submitted_at).total_seconds() / 3600)
    aging = min(waited_h * AGING_POINTS_PER_HOUR, 25)
    bonus = 5 if (case.triage and case.triage.needs_manual_review) else 0
    return base + aging + bonus


@app.get("/cases", response_model=list[PatientCase])
def list_cases():
    """Stage 6: queue ranked by urgency tier plus waiting-time aging."""
    return sorted(CASES.values(), key=priority_score, reverse=True)


@app.get("/cases/{case_id}", response_model=PatientCase)
def get_case(case_id: str):
    if case_id not in CASES:
        raise HTTPException(status_code=404, detail="Case not found")
    return CASES[case_id]


@app.post("/cases/{case_id}/review")
def review_case(case_id: str, action: str, notes: str = ""):
    """Stage 7: reviewer approves, edits, or escalates."""
    if case_id not in CASES:
        raise HTTPException(status_code=404, detail="Case not found")
    if action not in ("approved", "edited", "escalated"):
        raise HTTPException(status_code=400, detail="Invalid action")

    CASES[case_id].review_status = action
    CASES[case_id].reviewer_notes = notes
    audit.log_action(AuditEntry(
        case_id=case_id, actor="reviewer", action=f"case_{action}", details=notes,
    ))
    return {"status": "ok", "case_id": case_id, "new_status": action}


@app.get("/audit")
def audit_log(limit: int = 300):
    """Whole audit log, newest first (read-only)."""
    return audit.get_recent(max(1, min(limit, 1000)))


@app.get("/config")
def public_config():
    """Read-only, non-secret view of how the system is set up (for the Settings page)."""
    return {
        "slm_provider": config.SLM_PROVIDER, "slm_model": config.SLM_MODEL,
        "whisper_model": config.WHISPER_MODEL_SIZE, "ocr_languages": config.OCR_LANGS,
        "supported_languages": config.SUPPORTED_LANGUAGES,
        "consent_required": config.REQUIRE_CONSENT_BEFORE_PROCESSING,
        "red_flag_keyword_count": len(config.RED_FLAG_KEYWORDS),
        "input_confidence_floor": config.INPUT_CONFIDENCE_FLOOR,
        "ocr_confidence_threshold": config.OCR_CONFIDENCE_THRESHOLD,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "aging_points_per_hour": AGING_POINTS_PER_HOUR,
        "urgency_levels": config.URGENCY_LEVELS,
    }


@app.get("/cases/{case_id}/audit-trail")
def audit_trail(case_id: str):
    return audit.get_case_audit_trail(case_id)


@app.delete("/cases/{case_id}")
def withdraw_consent(case_id: str):
    """Patient-initiated data deletion."""
    audit.withdraw_consent_and_delete(case_id)
    case = CASES.pop(case_id, None)
    if case:
        for path in (case.input.voice_file_path, case.input.report_image_path):
            if path and os.path.exists(path):
                os.remove(path)
    return {"status": "deleted", "case_id": case_id}