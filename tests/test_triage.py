from unittest.mock import patch
from datetime import datetime, timedelta
from backend.pipeline import triage
from backend.models.schemas import Flag, PatientCase, PatientInput, TriageResult
from backend import main


def test_fuse_never_lowers():
    assert triage.fuse("high", "low") == "high"
    assert triage.fuse("low", "high") == "high"


def test_red_flag_overrides_slm():
    with patch.object(triage, "call_slm_for_triage", return_value={"urgency": "low", "reason": "x"}):
        r = triage.score_case("c1", "severe chest pain since morning", [])
    assert r.urgency_level == "critical"


def test_model_can_raise_floor():
    with patch.object(triage, "call_slm_for_triage", return_value={"urgency": "high", "reason": "x"}):
        r = triage.score_case("c2", "fever for two days", [])
    assert r.urgency_level == "high" and r.model_proposal == "high"


def test_model_failure_goes_manual_review_elevated():
    with patch.object(triage, "call_slm_for_triage", return_value=None):
        r = triage.score_case("c3", "mild cough", [])
    assert r.needs_manual_review and r.urgency_level == "high"


def test_parse_messy_output():
    assert triage.parse_slm_output('Sure! {"urgency":"Medium","reason":"ok"} thanks')["urgency"] == "medium"
    assert triage.parse_slm_output("no json") is None
    assert triage.parse_slm_output('{"urgency":"extreme"}') is None


def _case(cid, urg, hours):
    pi = PatientInput(case_id=cid, input_type="text", raw_text="x",
                      submitted_at=datetime.utcnow() - timedelta(hours=hours))
    t = TriageResult(case_id=cid, urgency_level=urg, urgency_reason="", scored_by="rule_based", confidence=.5)
    return PatientCase(case_id=cid, input=pi, triage=t)


def test_aging_lifts_old_case_within_tier_but_not_over_higher_tier():
    old_medium, new_medium = _case("a", "medium", 5), _case("b", "medium", 0)
    assert main.priority_score(old_medium) > main.priority_score(new_medium)
    old_medium_long = _case("c", "medium", 100)
    assert main.priority_score(old_medium_long) < main.priority_score(_case("d", "high", 0))


def test_extraction_hindi_and_english_duration_and_severity():
    from backend.pipeline import extraction as ex
    def get(text, name):
        return {f.field_name: f.value for f in ex.extract_symptom_timeline(text)}.get(name)
    assert get("mujhe 3 din se halka bukhar hai", "duration_mentioned") == "3 din"
    assert get("mujhe 3 din se halka bukhar hai", "severity") == "halka"
    assert get("mera sar dard since yesterday", "duration_mentioned") == "since yesterday"
    assert get("fever for two days", "duration_mentioned") == "two days"
    assert get("severe chest pain for 2 hours", "severity") == "severe"
    assert get("मुझे 3 दिन से बुखार है", "duration_mentioned") == "3 दिन"
    assert get("headache", "duration_mentioned") is None


def _run_case(input_type, **kw):
    from backend.models.schemas import PatientInput
    from backend.pipeline import intake, extraction
    pi = PatientInput(case_id="t-" + input_type, patient_consent_given=True, input_type=input_type, **kw)
    with patch.object(main.audit, "log_action"), \
         patch.object(triage, "call_slm_for_triage", return_value={"urgency": "medium", "reason": "ok"}):
        return pi, main._process_case(pi)


def test_voice_transcript_flows_through_pipeline():
    from backend.pipeline import intake
    with patch.object(intake, "transcribe_voice", return_value=("mujhe 3 din se bukhar hai", 0.8)):
        _, case = _run_case("voice", voice_file_path="x.wav")
    assert case.input.raw_text == "mujhe 3 din se bukhar hai"
    assert any(f.source == "voice_transcript" for f in case.extracted_fields)
    assert case.triage.urgency_level == "medium"


def test_voice_failure_goes_to_manual_review_not_a_guess():
    from backend.pipeline import intake
    with patch.object(intake, "transcribe_voice", return_value=("", 0.0)):
        _, case = _run_case("voice", voice_file_path="x.wav")
    assert any(f.flag_type == "unclear_input" and f.severity == "critical" for f in case.flags)
    assert case.triage.urgency_level in ("high", "critical")


def test_ocr_low_confidence_is_flagged():
    from backend.pipeline import extraction
    with patch.object(extraction, "ocr_report_image", return_value=("chest pain since morning", 0.4)):
        _, case = _run_case("report_image", report_image_path="x.png")
    assert case.triage.urgency_level == "critical"
    assert any(f.flag_type == "low_confidence" for f in case.flags)


def test_arabic_script_is_flagged_critical():
    from backend.pipeline import flagging
    flags = flagging.run_all_flags([], [], "\u0645\u062c\u06be\u06d2 \u0628\u062e\u0627\u0631 \u06c1\u06d2")
    assert any(f.flag_type == "unclear_input" and f.severity == "critical" for f in flags)
    assert not flagging.flag_unsupported_script("mujhe bukhar hai")
    assert not flagging.flag_unsupported_script("\u092e\u0941\u091d\u0947 \u092c\u0941\u0916\u093e\u0930 \u0939\u0948")


def test_asr_language_is_passed_to_whisper():
    from backend.pipeline import intake
    seen = {}
    def fake(path, language=None):
        seen["lang"] = language
        return "x", 0.9
    with patch.object(intake, "transcribe_voice", side_effect=fake):
        _run_case("voice", voice_file_path="x.wav", declared_language="mixed")
    assert seen["lang"] == "hi"


def test_low_confidence_voice_cannot_rank_low():
    from backend.pipeline import intake
    with patch.object(intake, "transcribe_voice", return_value=("din din se vukar hai", 0.5)), \
         patch.object(triage, "call_slm_for_triage", return_value={"urgency": "low", "reason": "vague"}):
        pi, case = None, None
        from backend.models.schemas import PatientInput
        pi = PatientInput(case_id="t-lowconf", patient_consent_given=True, input_type="voice", voice_file_path="x.wav")
        with patch.object(main.audit, "log_action"):
            case = main._process_case(pi)
    assert case.triage.urgency_level == "medium" and case.triage.needs_manual_review


def test_typed_text_is_not_gated():
    with patch.object(triage, "call_slm_for_triage", return_value={"urgency": "low", "reason": "ok"}):
        r = triage.score_case("c9", "mild headache", [])
    out = triage.apply_uncertainty_gate(r, "typed_text", 0.2)
    assert out is r and not out.needs_manual_review


def test_translation_placeholder_no_longer_prefixes_text():
    from backend.pipeline import language
    assert language.translate_to_english("मुझे बुखार है", "hi")[0] == "मुझे बुखार है"