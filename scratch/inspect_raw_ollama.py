import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"c:\Users\ammua\Desktop\final_prompt")))
sys.path.insert(0, str(Path(r"c:\Users\ammua\Desktop\final_prompt\pipeline")))
import requests
import json
from diagnosis_inference import DiagnosisInference

with open("results/v3_benchmark/process_6/medical_report.json", "r", encoding="utf-8") as f:
    med = json.load(f)

dx = DiagnosisInference()
summary = dx.format_report_for_llm(med)
print("Summary:")
print(summary)

prompt = (
    f"Analyze the following patient findings and provide a differential diagnosis:\n\n"
    f"{summary}\n\n"
    f"Note: Differentiate active presenting complaints from resolved past medical history.\n\n"
    f"Provide your assessment in valid JSON matching this exact structure:\n"
    f"{{\n"
    f'  "primary_diagnosis": "canonical disease or clinical syndrome name",\n'
    f'  "secondary_diagnoses": ["differential 1", "differential 2"],\n'
    f'  "clinical_reasoning": "detailed clinical rationale linking symptoms to diagnosis",\n'
    f'  "recommended_tests": ["confirmatory test 1", "confirmatory test 2"]\n'
    f"}}\n\n"
    f"CRITICAL CLINICAL DIRECTIVES:\n"
    f"1. 'primary_diagnosis' MUST be a specific, concise medical disease or syndrome name (e.g. Choledocholithiasis, Obstructive Jaundice, Acute Cholecystitis).\n"
    f"2. Do NOT write a sentence or describe symptoms in 'primary_diagnosis'. It must be strictly the disease/syndrome name."
)

body = {
    "model": dx.ollama_model,
    "messages": [
        {"role": "system", "content": "You are an expert physician, clinical diagnostician, and gastroenterology specialist."},
        {"role": "user", "content": prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 1024}
}

res = requests.post("http://localhost:11434/api/chat", json=body, timeout=120)
raw = res.json().get("message", {}).get("content", "")
print("=== RAW OLLAMA OUTPUT FOR CASE 6 ===")
print(raw)

try:
    j = json.loads(raw)
    print("\nJSON parse SUCCESS:", j.keys())
    print("primary_diagnosis:", j.get("primary_diagnosis"))
    print("secondary_diagnoses:", j.get("secondary_diagnoses"))
except Exception as e:
    print("\nJSON parse FAILED:", e)
