import sys
import json
import requests
import traceback
sys.path.insert(0, ".")
from pipeline.diagnosis_inference import DiagnosisInference, OLLAMA_API_URL, OLLAMA_MODEL

dx = DiagnosisInference()
report = json.load(open('results/process_1/medical_report.json', encoding='utf-8'))
summary = dx.format_report_for_llm(report)

print(f"Summary length: {len(summary)}")

prompt = (
    f"Analyze the following patient findings and provide a differential diagnosis:\n\n"
    f"{summary}\n\n"
    f"Return a JSON object with:\n"
    f"\"primary_diagnosis\": string,\n"
    f"\"secondary_diagnoses\": list of strings,\n"
    f"\"clinical_reasoning\": string,\n"
    f"\"recommended_tests\": list of strings"
)

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "system", "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience."},
        {"role": "user", "content": prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 512
    }
}

try:
    r = requests.post(OLLAMA_API_URL, json=body, timeout=120)
    print("Status Code:", r.status_code)
    raw = r.json().get("message", {}).get("content", "")
    print("Raw Content:\n", raw[:600])
    parsed = dx._parse_llm_response(raw)
    print("\nParsed Result:", parsed)
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
