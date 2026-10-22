"""
 Extraction & structuring
"""

import re
from backend.config import OCR_CONFIDENCE_THRESHOLD
from backend.models.schemas import ExtractedField

# --- Rule-based first pass (English, Hinglish, Devanagari). The SLM replaces this later. ---
_NUM = r"(?:\d+|ek|do|teen|char|paanch|one|two|three|four|five|few|kuch|एक|दो|तीन|चार|पांच)"
_UNIT = (r"(?:days?|din|weeks?|hafta|hafte|saptah|hours?|ghante|ghanta|months?|mahina|mahine|"
         r"दिन|हफ्ते|हफ़्ते|घंटे|महीने)")
_DURATION_NUMERIC = re.compile(rf"(?<![\w])({_NUM})\s*(?:-|\s)?\s*({_UNIT})(?![a-z])", re.I)
_DURATION_RELATIVE = re.compile(
    r"(since\s+(?:this\s+|last\s+)?(?:yesterday|morning|night|week|today|evening)|"
    r"(?:kal|parso|subah|raat)\s+se|aaj\s+subah|कल\s+से|सुबह\s+से|रात\s+से)", re.I)
_SEVERITY = re.compile(
    r"(?<![\w])(severe(?:ly)?|mild(?:ly)?|slight(?:ly)?|unbearable|very\s+bad|worst|"
    r"bahut|bohot|halka|halki|thoda|thodi|zyada|tez|"
    r"असहनीय|हल्का|हल्की|बहुत|तेज)(?![a-z])", re.I)


def _find_duration(text: str):
    m = _DURATION_NUMERIC.search(text)
    if m:
        return m.group(0).strip(), 0.75
    m = _DURATION_RELATIVE.search(text)
    if m:
        return m.group(0).strip(), 0.65
    return None


def _find_severity(text: str):
    m = _SEVERITY.search(text)
    return m.group(0).strip().lower() if m else None


def ocr_report_image(image_path: str) -> tuple[str, float]:
    """
    Tesseract OCR. Returns (text, mean word confidence 0-1). On any failure returns
    ("", 0.0) so the caller routes the case to manual review instead of guessing.
    """
    try:
        import pytesseract
        from PIL import Image
        from backend.config import TESSERACT_CMD, OCR_LANGS
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        langs = OCR_LANGS
        available = set(pytesseract.get_languages(config=""))
        langs = "+".join(l for l in OCR_LANGS.split("+") if l in available) or "eng"
        data = pytesseract.image_to_data(Image.open(image_path), lang=langs,
                                         output_type=pytesseract.Output.DICT)
        words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        text = " ".join(w for w, _ in words)
        conf = sum(c for _, c in words) / len(words) / 100
        return text, max(0.0, min(1.0, conf))
    except Exception as e:
        print(f"[extraction] OCR failed: {type(e).__name__}: {e}")
        return "", 0.0


def extract_symptom_timeline(text: str, source: str = "typed_text", base_confidence: float = 0.95) -> list[ExtractedField]:
    """
    TODO: this is the natural place for an SLM call — prompt it to pull out
    {symptom, onset, duration, severity, changes_over_time} as structured
    JSON. Keep the prompt strict about returning valid JSON only, and
    always carry the confidence/source metadata through.

    Rule-based placeholder below does a very rough first pass so the
    pipeline has *something* to show end-to-end before the SLM is wired in.
    """
    fields = []

    duration = _find_duration(text)
    if duration:
        value, confidence = duration
        fields.append(ExtractedField(
            field_name="duration_mentioned", value=value,
            source=source, confidence=min(confidence, base_confidence),
        ))

    severity = _find_severity(text)
    if severity:
        fields.append(ExtractedField(
            field_name="severity", value=severity,
            source=source, confidence=min(0.65, base_confidence),
        ))

    fields.append(ExtractedField(
        field_name="raw_symptom_text",
        value=text,
        source=source,
        confidence=base_confidence,  # 0.95 for typed text; ASR/OCR confidence otherwise
    ))

    return fields


def identify_missing_fields(extracted: list[ExtractedField]) -> list[str]:
    """Checks which expected fields never got extracted at all —
    feeds directly into the flagging stage."""
    expected = {"raw_symptom_text", "duration_mentioned", "severity"}
    found = {f.field_name for f in extracted}
    return list(expected - found)
