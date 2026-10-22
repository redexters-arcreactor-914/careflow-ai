"""
Flag gaps & conflicts

"""

from backend.config import OCR_CONFIDENCE_THRESHOLD, TRANSLATION_CONFIDENCE_THRESHOLD
from backend.models.schemas import ExtractedField, Flag


def flag_low_confidence_fields(extracted: list[ExtractedField]) -> list[Flag]:
    flags = []
    for field in extracted:
        threshold = OCR_CONFIDENCE_THRESHOLD if field.source == "ocr_report" else 0.6
        if field.confidence < threshold:
            flags.append(Flag(
                flag_type="low_confidence",
                field_name=field.field_name,
                description=f"'{field.field_name}' extracted with confidence "
                            f"{field.confidence:.2f}, below threshold {threshold}.",
                severity="warning",
            ))
    return flags


def flag_missing_fields(missing_fields: list[str]) -> list[Flag]:
    return [
        Flag(
            flag_type="missing_info",
            field_name=field,
            description=f"'{field}' was not found in the patient's input.",
            severity="info",
        )
        for field in missing_fields
    ]


def flag_conflicting_fields(extracted: list[ExtractedField]) -> list[Flag]:
    """
    TODO: real conflict detection compares fields pulled from different
    sources (e.g. voice transcript says '3 days', the OCR'd report says
    '2 weeks') and flags the mismatch instead of picking one silently.

    Placeholder: groups fields by name and checks for differing values
    across different sources.
    """
    flags = []
    by_field_name: dict[str, list[ExtractedField]] = {}
    for f in extracted:
        by_field_name.setdefault(f.field_name, []).append(f)

    for field_name, instances in by_field_name.items():
        if len(instances) > 1:
            values = {i.value for i in instances}
            if len(values) > 1:
                flags.append(Flag(
                    flag_type="conflicting_info",
                    field_name=field_name,
                    description=f"Conflicting values for '{field_name}': {values}",
                    severity="warning",
                ))
    return flags


def flag_suspicious_instructions(text: str) -> list[Flag]:
    """
    Guards against prompt-injection-style content hiding inside patient
    input (e.g. a report that contains text trying to instruct the AI
    directly). Keep this rule-based and conservative — false positives
    here are far cheaper than missing one.
    """
    suspicious_markers = [
        "ignore previous instructions", "system prompt", "you are now",
        "disregard the above", "act as if",
    ]
    lowered = text.lower()
    flags = []
    for marker in suspicious_markers:
        if marker in lowered:
            flags.append(Flag(
                flag_type="suspicious_instruction",
                description=f"Input contains a phrase resembling an instruction "
                            f"override: '{marker}'. Treat as patient-reported text only.",
                severity="critical",
            ))
    return flags


def flag_unsupported_script(text: str) -> list[Flag]:
    """Our red-flag rules cover Latin and Devanagari only. Text in another script
    (for example Arabic/Urdu) cannot be checked by the rules, so a human must read it."""
    if any("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" for ch in text):
        return [Flag(
            flag_type="unclear_input",
            description="Text is in a script the safety rules do not cover (Arabic/Urdu). "
                        "Review the original input.",
            severity="critical",
        )]
    return []


def run_all_flags(
    extracted: list[ExtractedField],
    missing_fields: list[str],
    raw_text: str,
) -> list[Flag]:
    return (
        flag_low_confidence_fields(extracted)
        + flag_missing_fields(missing_fields)
        + flag_conflicting_fields(extracted)
        + flag_suspicious_instructions(raw_text)
        + flag_unsupported_script(raw_text)
    )