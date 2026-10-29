import json
import requests
from pathlib import Path
import sys

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
sys.path.insert(0, str(BASE_DIR))
from pipeline.diagnosis_inference import DiagnosisInference

dx = DiagnosisInference()

for cid in [4, 5, 6, 8]:
    with open(f"results/v3_benchmark/process_{cid}/medical_report.json", "r", encoding="utf-8") as f:
        med = json.load(f)
    summary = dx.format_report_for_llm(med)

    system_msg = (
        "You are OpenBioLLM-8B, an expert clinical diagnostic AI specializing in gastroenterology and internal medicine.\n"
        "You must output ONLY a valid JSON object adhering strictly to the schema below. Do NOT output markdown code blocks or conversational text.\n"
        "Schema:\n"
        "{\n"
        '  "primary_diagnosis": "<Concise canonical disease or clinical syndrome name (1-5 words)>",\n'
        '  "secondary_diagnoses": ["<Differential 1>", "<Differential 2>"],\n'
        '  "clinical_reasoning": "<1-3 sentences explaining pathophysiology and justification>",\n'
        '  "recommended_tests": ["<Investigation 1>", "<Investigation 2>"]\n'
        "}\n"
        "CRITICAL RULES:\n"
        '1. "primary_diagnosis" MUST be a formal medical diagnosis or syndrome (e.g. Peptic Ulcer Disease, Gastroesophageal Reflux Disease, Acute Gastritis, Cholelithiasis, Obstructive Jaundice, Irritable Bowel Syndrome). NEVER write symptom sentences, narrative descriptions, or patient history in primary_diagnosis.\n'
        "2. Differentiate active chief complaints from resolved past medical history."
    )

    body = {
        "model": dx.ollama_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Diagnose this patient case:\n\n{summary}"}
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512}
    }
    res = requests.post("http://127.0.0.1:11434/api/chat", json=body, timeout=120)
    content = res.json().get("message", {}).get("content", "")
    print(f"=== CASE {cid} ===")
    try:
        parsed = json.loads(content)
        print("  Primary:", parsed.get("primary_diagnosis"))
        print("  Secondary:", parsed.get("secondary_diagnoses"))
    except Exception as e:
        print("  Raw:", content[:200])

