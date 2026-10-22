"""
 Language detection & translation

"""

from backend.config import SUPPORTED_LANGUAGES, TRANSLATION_CONFIDENCE_THRESHOLD


def detect_language(text: str) -> tuple[str, float]:
    """
    TODO: wire up real language detection, e.g.:
        from langdetect import detect_langs
        guesses = detect_langs(text)
        return guesses[0].lang, guesses[0].prob

    For Hindi-English mixed text specifically, a simple heuristic
    (ratio of Devanagari characters to Latin characters) is often more
    reliable than generic language-detection libraries — worth trying
    before reaching for something heavier.
    """
    devanagari_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    latin_chars = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if devanagari_chars == 0:
        return "en", 0.9
    if latin_chars == 0:
        return "hi", 0.9
    return "mixed", 0.7


def translate_to_english(text: str, source_lang: str) -> tuple[str, float]:
    """
    TODO: wire up real translation. Options, roughly in order of how much
    setup they need for a hackathon timeline:
        1. SLM prompt (if you already have phi3/gemma running via Ollama,
           just ask it to translate — one less dependency to install)
        2. googletrans (quick, needs internet, rate-limited)
        3. IndicTrans2 (best quality for Indian languages, heavier install)

    Returns (translated_text, confidence). If source_lang == "en", this
    is a no-op passthrough.
    """
    if source_lang == "en":
        return text, 1.0

   
    return text, 1.0


def process_language(text: str) -> dict:
    """Convenience wrapper used by main.py — runs detect + translate together
    and flags low-confidence translations for the uncertainty stage."""
    lang, detect_conf = detect_language(text)
    translated_text, translate_conf = translate_to_english(text, lang)

    return {
        "original_text": text,
        "detected_language": lang,
        "detection_confidence": detect_conf,
        "translated_text": translated_text,
        "translation_confidence": translate_conf,
        "low_confidence": translate_conf < TRANSLATION_CONFIDENCE_THRESHOLD,
    }