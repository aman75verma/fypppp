import sys
import json
import requests
import traceback
sys.path.insert(0, ".")
from pipeline.diagnosis_inference import DiagnosisInference, OLLAMA_API_URL, OLLAMA_MODEL

dx = DiagnosisInference()
report = json.load(open('results/process_1/medical_report.json', encoding='utf-8'))
summary = dx.format_report_for_llm(report)

prompt = (
    f"Analyze the following patient findings and provide a differential diagnosis:\n\n"
    f"{summary}\n\n"
    f"Provide the assessment with:\n"
    f"Primary Diagnosis: <diagnosis>\n"
    f"Secondary Diagnoses: <diagnoses>\n"
    f"Clinical Reasoning: <reasoning>\n"
    f"Recommended Tests: <tests>"
)

body = {
    "model": dx.ollama_model,
    "messages": [
        {
            "role": "system",
            "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience."
        },
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 512
    }
}

try:
    print("Sending request to Ollama...")
    res = requests.post(OLLAMA_API_URL, json=body, timeout=120)
    print("Status code:", res.status_code)
    if res.status_code == 200:
        raw_text = res.json().get("message", {}).get("content", "")
        print("Raw text length:", len(raw_text))
        print("Raw text:\n", raw_text)
        parsed = dx._parse_llm_response(raw_text)
        print("\nParsed:", parsed)
        print("Parse error?", parsed.get("parse_error"))
    else:
        print("Error response:", res.text)
except Exception as e:
    print("Exception occurred:", e)
    traceback.print_exc()
