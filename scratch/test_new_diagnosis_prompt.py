import json
import requests
import re
import sys

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "hf.co/bartowski/OpenBioLLM-Llama3-8B-GGUF:Q4_K_M"

test_summary_p1 = """CHIEF COMPLAINT:
  - She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days [Symptoms: severe constipation, stool | Frequency: only once every 2–3 days]
  - She is having gas and bloating. [Symptoms: gas, bloating]

HISTORY OF PRESENT ILLNESS:
  - Constipation has been severe and persistent since around 2024.
  - She is having significant flatulence and abdominal distension.

PAST MEDICAL HISTORY:
  - In 2018, diagnosed with rheumatoid arthritis; on long-term medication (folic acid, etc.).
  - In 2019, experienced heavy menstrual bleeding; investigated with ultrasound.
  - Abdominal ultrasound confirmed a gallbladder stone (cholelithiasis); doctor advised non-surgical management as there was no acute biliary colic."""

system_prompt = """You are OpenBioLLM-8B, an elite clinical decision-support specialist in internal medicine and gastroenterology.
Analyze the structured patient record carefully.
CLINICAL GUIDELINES:
1. Differentiate active presenting complaints from resolved past medical history or screening inquiries.
2. Consider biliary-associated gastrointestinal dysmotility: gallstones impair bile excretion, precipitating severe constipation and gas. Long-term medication for rheumatoid arthritis can further aggravate dysmotility.
3. Formulate step-by-step pathophysiological clinical reasoning FIRST before declaring the primary diagnosis.

CRITICAL: Output ONLY a valid JSON object matching this schema:
{
  "clinical_presentation_summary": "<Synthesis of active symptoms, onset, and timeline>",
  "pathophysiological_reasoning": "<Detailed step-by-step clinical chain-of-thought linking symptoms to underlying disease etiology>",
  "primary_diagnosis": "<Single most probable primary diagnosis>",
  "secondary_diagnoses": ["<Differential 1>", "<Differential 2>"],
  "recommended_tests": ["<Diagnostic Test 1>", "<Diagnostic Test 2>", "<Diagnostic Test 3>"]
}"""

user_prompt = f"Analyze the following patient clinical record and formulate your differential diagnosis:\n\n{test_summary_p1}\n\nRespond ONLY with the JSON object."

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 1024,
        "top_p": 0.1
    }
}

print("Calling Ollama with Forward CoT Prompt...")
res = requests.post(OLLAMA_API_URL, json=body, timeout=120)
print(f"Status: {res.status_code}")
if res.status_code == 200:
    content = res.json().get("message", {}).get("content", "")
    print("Response Content:\n", content)
else:
    print("Error:", res.text)

