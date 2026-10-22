"""
Central config.
"""


SLM_PROVIDER = "ollama"         
SLM_BASE_URL = "http://localhost:11434"
SLM_MODEL = "phi3"              

#  Triage urgency levels ---
URGENCY_LEVELS = ["low", "medium", "high", "critical"]


RED_FLAG_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "unconscious", "unresponsive", "stroke",
    "seizure", "suicidal", "severe allergic reaction", "anaphylaxis",
    # Romanised Hindi / Hinglish (starter set, needs clinician review)
    "seene mein dard", "seene me dard", "saans lene mein takleef", "saans lene me takleef",
    "saans phool", "behosh", "daura pada", "daura aaya", "bahut khoon", "khoon nahi ruk",
    "jaan ka khatra", "khudkushi",
    # Devanagari
    "सीने में दर्द", "सांस लेने में तकलीफ", "बेहोश", "दौरा", "बहुत खून", "खुदकुशी",
]

# Confidence thresholds 
# Below this, a field gets flagged as "low confidence" rather than trusted outright.
OCR_CONFIDENCE_THRESHOLD = 0.6
TRANSLATION_CONFIDENCE_THRESHOLD = 0.65

#  Supported languages for the multilingual layer 
SUPPORTED_LANGUAGES = ["en", "hi", "mixed"]  

#  Consent / audit 
REQUIRE_CONSENT_BEFORE_PROCESSING = True
AUDIT_LOG_PATH = "backend/data/audit_log.jsonl"

#  Voice / OCR / uploads 
WHISPER_MODEL_SIZE = "small"    
WHISPER_DEVICE = "cpu"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"            
OCR_LANGS = "eng"                
UPLOAD_DIR = "backend/data/uploads"
MAX_UPLOAD_MB = 25
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".opus"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Vocabulary hints make Whisper far more accurate on short medical phrases.
WHISPER_PROMPT = {
    "hi": "मुझे तीन दिन से बुखार है। सीने में दर्द, सांस लेने में तकलीफ, खांसी, सिर दर्द, उल्टी, चक्कर, कमज़ोरी।",
    "en": "I have had fever for three days. Chest pain, shortness of breath, cough, headache, vomiting, dizziness.",
}
INPUT_CONFIDENCE_FLOOR = 0.6   # voice/OCR read below this is not trusted to rank a case low