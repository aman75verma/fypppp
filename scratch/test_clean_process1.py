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
  - She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days
  - Flatulence and abdominal distension since 2024.

PAST MEDICAL HISTORY:
  - In 2018: rheumatoid arthritis; taking medications (folic acid, etc.).
  - In 2019: transient episode of heavy menstrual bleeding (resolved).
  - Gallbladder stone (cholelithiasis) detected on abdominal USG; advised non-surgical conservative monitoring."""

dx = DiagnosisInference()

prompt = (
    f"Analyze the following patient findings and provide a differential diagnosis:\n\n"
    f"{clean_summary}\n\n"
    f"Note: Differentiate active presenting complaints from resolved past medical history.\n\n"
    f"Return your assessment with:\n"
    f"Primary Diagnosis: <diagnosis>\n"
    f"Secondary Diagnoses: <diagnoses>\n"
    f"Clinical Reasoning: <reasoning>\n"
    f"Recommended Tests: <tests>"
)

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience."},
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 1024
    }
}

try:
    print("Calling Ollama with clean summary and 1024 tokens...", flush=True)
    r = requests.post(OLLAMA_API_URL, json=body, timeout=120)
    print("Status Code:", r.status_code, flush=True)
    raw = r.json().get("message", {}).get("content", "")
    print("Raw Content:\n", raw, flush=True)
    parsed = dx._parse_llm_response(raw)
    print("\nParsed Result:", json.dumps(parsed, indent=2), flush=True)
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

