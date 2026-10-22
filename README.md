# CareFlow AI

Human-in-the-loop healthcare triage assistant. Built for the Confluentia hackathon (Tredence track) by team **The_Chaotic_123**: Anshuman, Animesh, Aakash.

CareFlow takes a patient's input (typed symptoms, a voice note or a report photo), structures it, flags gaps and conflicts, proposes an urgency, and puts the case in a ranked queue for a human reviewer. **The system never diagnoses and never makes the final call.** Every case ends with a clinician, and every flag and score shows its reason.

## Pipeline

```
Patient input (text / voice / report image)
        |
Consent gate            no consent, no processing, nothing stored
        |
Voice -> text (Whisper)  |  Report image -> text (Tesseract)
        |
Extraction              duration, severity, symptom timeline (English, Hindi, Hinglish)
        |
Flag engine             missing info, low confidence, conflicts, prompt-injection, unsupported script
        |
Triage                  urgency = max(rule floor, Phi-3 proposal)
        |
Uncertainty gate        low-confidence voice/OCR is raised to at least medium + manual review
        |
Aging priority queue    urgency + waiting time
        |
Human review            approve / edit / escalate, all written to an append-only audit log
```

## Design principles

- **Escalate-only fusion.** Red-flag rules set a floor before the model runs. The model can only raise urgency, never lower it. Critical rule hits skip the model.
- **Fail-safe, not guess.** If the model, speech-to-text or OCR fails or is unsure, the case goes to manual review at elevated priority.
- **Privacy.** Whisper, Tesseract and Phi-3 (via Ollama) all run locally. Withdrawing consent deletes the case and its uploaded files.
- **Aging queue.** +5 points per hour waiting (cap 25, +5 if manual review), so old medium cases do not get buried.

## Status

| Area | State |
|---|---|
| Consent gate, append-only audit log, consent withdrawal | Built |
| Text, voice (faster-whisper) and report-image (Tesseract) intake | Built |
| Red-flag rules (English, romanised Hindi, Devanagari), flag engine | Built |
| Fusion triage with Phi-3, failure fallback, uncertainty gate, aging queue | Built |
| Reviewer dashboard: queue, all cases, audit log, settings, about, new-case form | Built |
| Golden evaluation set and unit tests | Built |
| Hindi to English translation | Not wired in (rules read Hindi directly) |
| SLM-based extraction | In progress (extraction is regex-based today) |
| Auth and roles, PostgreSQL, async queue and workers | Planned |

## Project structure

```
careflow-ai/
├── backend/
│   ├── main.py              FastAPI app and routes
│   ├── config.py            thresholds, SLM, Whisper/Tesseract settings, red-flag keywords
│   ├── requirements.txt
│   ├── pipeline/
│   │   ├── intake.py        consent check, voice transcription (faster-whisper)
│   │   ├── language.py      language handling (translation is a passthrough for now)
│   │   ├── extraction.py    OCR (Tesseract) and symptom/timeline extraction
│   │   ├── flagging.py      missing / low-confidence / conflict / injection / script flags
│   │   ├── triage.py        rule floor, Phi-3 proposal, fusion, uncertainty gate, aging
│   │   └── audit.py         append-only JSONL audit log
│   ├── models/schemas.py    Pydantic models
│   └── data/sample_cases.json
├── frontend/dashboard.html  reviewer dashboard (single file, open in a browser)
├── eval/                    golden_cases.json (24 synthetic cases) and run_eval.py
└── tests/test_triage.py
```

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/cases/intake` | submit a typed case (JSON) |
| POST | `/cases/upload` | submit a voice note or report image (multipart) |
| GET | `/cases` | ranked queue |
| GET | `/cases/{id}` | one case |
| POST | `/cases/{id}/review` | approve / edit / escalate |
| GET | `/cases/{id}/audit-trail` | audit events for a case |
| DELETE | `/cases/{id}` | withdraw consent, delete case and files |
| GET | `/audit` | recent audit events |
| GET | `/config` | non-secret settings |

## Setup (Windows)

Requires Python 3.10+.

```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

**Phi-3 (triage model).** Install Ollama from https://ollama.com, then:

```bat
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" pull phi3
```

Warm it up once (`ollama run phi3`, say hello, `/bye`) so the first case does not time out. `gemma2:2b` is a lighter fallback.

**Tesseract (report images).** Install the Windows build from the UB Mannheim page. If it is not on PATH, set `TESSERACT_CMD` in `backend/config.py`, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`.

**Whisper (voice).** `faster-whisper` downloads the `small` model on first use. Change `WHISPER_MODEL_SIZE` in `backend/config.py` if you need a lighter one.

## Run

```bat
uvicorn backend.main:app --port 8000
```

Then open `frontend/dashboard.html` in a browser. It shows **LIVE** when connected to the API and falls back to labelled demo data (**OFFLINE**) when it is not. Use **+ New case** to submit text, voice or a report image. Run without `--reload` while demoing.

## Tests and evaluation

```bat
python -m pytest tests
python -m eval.run_eval --rules-only
python -m eval.run_eval
```

`eval/golden_cases.json` holds 24 synthetic cases (English 15, Hindi 7, Hinglish 2).

| Run | Critical recall | Under-triage | Over-triage |
|---|---|---|---|
| Rules only | 13/13 | 0 | 9 |
| Rules + Phi-3 | 13/13 | 0 | 1 |

These are **development results, not accuracy claims**. Labels were written by the team and are not clinician-reviewed. Hindi keywords were added after the first run missed all four Hindi critical cases, so the set is no longer held-out. A fresh unseen set is needed before quoting any number.

## Known limitations

- Red-flag rules are a starter keyword set. They do not understand negation ("no chest pain" still raises critical), which over-triages but does not under-triage. They need clinician review and mapping to standards such as ESI or IMNCI.
- Whisper can mishear Hindi. Low confidence sends the case to manual review. The original audio is not yet shown to the reviewer.
- Cases are held in memory and are lost on restart. The audit log is a JSONL file and is not yet hash-chained.
- No authentication or roles yet.
- Not a medical device. For research and demonstration only.
