import sys
import json
import requests
import traceback
sys.path.insert(0, ".")
from pipeline.diagnosis_inference import DiagnosisInference, OLLAMA_API_URL, OLLAMA_MODEL

clean_summary = """CHIEF COMPLAINT:
  - She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days [Symptoms: severe constipation, stool | Frequency: only once every 2–3 days]
  - She is having gas. [Symptoms: gas]

HISTORY OF PRESENT ILLNESS:
  - Flatulence and abdominal distension since 2024.
  - Constipation has been severe and persistent.

PAST MEDICAL HISTORY:
  - In 2018: rheumatoid arthritis; taking medications (folic acid, etc.).
  - In 2019: transient episode of heavy menstrual bleeding (resolved).
  - Gallbladder stone (cholelithiasis) detected on abdominal USG; advised non-surgical conservative monitoring."""

dx = DiagnosisInference()

# Testing prompt format
prompt = f"""You are an expert clinical diagnostician. Based on the following structured patient report, provide your clinical diagnostic assessment.

PATIENT REPORT:
{clean_summary}

Provide your differential diagnosis assessment in the exact format below:

Primary Diagnosis: [Primary diagnosis name]
Secondary Diagnoses: [Comma-separated secondary diagnoses or differentials]
Clinical Reasoning: [Detailed clinical explanation connecting symptoms to diagnosis]
Recommended Tests: [Recommended diagnostic tests]"""

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 1024
    }
}

try:
    print("Calling Ollama...", flush=True)
    r = requests.post(OLLAMA_API_URL, json=body, timeout=120)
    print("Status:", r.status_code, flush=True)
    raw = r.json().get("message", {}).get("content", "")
    print("Raw Content:\n", raw, flush=True)
    parsed = dx._parse_llm_response(raw)
    print("\nParsed Result:\n", json.dumps(parsed, indent=2), flush=True)
except Exception as e:
    print("Error:", e)

