import sys
import json
import requests
sys.path.insert(0, ".")
from pipeline.diagnosis_inference import DiagnosisInference, OLLAMA_API_URL, OLLAMA_MODEL

dx = DiagnosisInference()
report = json.load(open('results/process_1/medical_report.json', encoding='utf-8'))
summary = dx.format_report_for_llm(report)
cleaned_summary = '\n'.join(list(dict.fromkeys(summary.splitlines())))

prompt = (
    f"Analyze the following patient findings and provide a differential diagnosis:\n\n"
    f"{cleaned_summary}\n\n"
    f"Return a JSON object with:\n"
    f"\"primary_diagnosis\": string,\n"
    f"\"secondary_diagnoses\": list of strings,\n"
    f"\"clinical_reasoning\": string,\n"
    f"\"recommended_tests\": list of strings"
)

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {
            "role": "system",
            "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience."
        },
        {"role": "user", "content": prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 512
    }
}

r = requests.post(OLLAMA_API_URL, json=body, timeout=120)
content = r.json().get("message", {}).get("content", "")
print("Content:\n", content)
parsed = dx._parse_llm_response(content)
print("\nParsed:\n", json.dumps(parsed, indent=2))
