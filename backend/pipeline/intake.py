"""
 Patient input

"""

from backend.models.schemas import PatientInput
from backend.config import REQUIRE_CONSENT_BEFORE_PROCESSING


class ConsentNotGivenError(Exception):
    pass


def validate_intake(patient_input: PatientInput) -> PatientInput:
    """Entry point for every incoming case. Raises if consent is missing."""
    if REQUIRE_CONSENT_BEFORE_PROCESSING and not patient_input.patient_consent_given:
        raise ConsentNotGivenError(
            f"Case {patient_input.case_id} has no recorded consent — "
            f"cannot proceed to processing."
        )

    if patient_input.input_type == "text" and not patient_input.raw_text:
        raise ValueError("input_type is 'text' but raw_text is empty.")
    if patient_input.input_type == "voice" and not patient_input.voice_file_path:
        raise ValueError("input_type is 'voice' but voice_file_path is missing.")
    if patient_input.input_type == "report_image" and not patient_input.report_image_path:
        raise ValueError("input_type is 'report_image' but report_image_path is missing.")

    return patient_input


_WHISPER = None


def transcribe_voice(voice_file_path: str, language: str | None = None) -> tuple[str, float]:
    """
    Local speech-to-text with faster-whisper (no API key, patient audio stays on the machine).
    `language` ("hi" or "en") should be passed whenever it is known: auto-detection often
    mistakes spoken Hindi for Urdu and writes Arabic script.
    Returns (transcript, confidence 0-1). On any failure returns ("", 0.0): the caller
    must then route the case to manual review instead of guessing.
    """
    global _WHISPER
    try:
        import math
        from faster_whisper import WhisperModel
        from backend.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_PROMPT
        if _WHISPER is None:
            _WHISPER = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
        segments, _info = _WHISPER.transcribe(
            voice_file_path, language=language, vad_filter=True, beam_size=5,
            initial_prompt=WHISPER_PROMPT.get(language or ""),
        )
        segs = list(segments)
        text = " ".join(seg.text.strip() for seg in segs).strip()
        if not segs:
            return "", 0.0
        conf = sum(math.exp(seg.avg_logprob) for seg in segs) / len(segs)
        return text, max(0.0, min(1.0, conf))
    except Exception as e:  # missing package, bad audio, model download failure
        print(f"[intake] transcription failed: {type(e).__name__}: {e}")
        return "", 0.0