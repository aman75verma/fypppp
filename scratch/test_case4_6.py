import json
import requests
from pathlib import Path
import sys

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
sys.path.insert(0, str(BASE_DIR))
from pipeline.diagnosis_inference import DiagnosisInference

dx = DiagnosisInference()

for cid in [4, 6]:
    with open(f"results/v3_benchmark/process_{cid}/medical_report.json", "r", encoding="utf-8") as f:
        med = json.load(f)
    summary = dx.format_report_for_llm(med)

    prompt = (
        f"You are OpenBioLLM, an expert clinical diagnostician. Based on the following structured patient findings, state your clinical diagnosis:\n\n"
        f"{summary}\n\n"
        f"Respond in EXACTLY the following format:\n"
        f"PRIMARY DIAGNOSIS: <Specific Medical Disease or Clinical Syndrome Name, max 5 words>\n"
        f"SECONDARY DIAGNOSES: <Differential condition 1, Differential condition 2>\n"
        f"CLINICAL REASONING: <1-3 sentences linking findings to diagnosis>\n"
        f"RECOMMENDED TESTS: <Confirmatory investigations>\n\n"
        f"IMPORTANT: PRIMARY DIAGNOSIS must be ONLY a medical disease name (e.g. Non-Ulcer Dyspepsia, Choledocholithiasis, Gastritis, GERD). Never write full sentences or describe symptoms in PRIMARY DIAGNOSIS."
    )

    body = {
        "model": dx.ollama_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300}
    }
    res = requests.post("http://127.0.0.1:11434/api/chat", json=body, timeout=120)
    content = res.json().get("message", {}).get("content", "")
    print(f"=== CASE {cid} ===")
    print(content)

