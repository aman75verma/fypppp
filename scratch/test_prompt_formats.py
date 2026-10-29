import requests
import json
import re

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "hf.co/bartowski/OpenBioLLM-Llama3-8B-GGUF:Q4_K_M"

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

# Prompt with few-shot / strict format
user_prompt = f"""You are an expert medical AI specializing in internal medicine and gastroenterology.
Analyze the following structured patient clinical summary. Distinguish active presenting complaints from resolved past history.

PATIENT CLINICAL SUMMARY:
{clean_summary}

Respond strictly using the following 4 labeled sections:
Primary Diagnosis: <name of most likely primary condition>
Secondary Diagnoses: <comma-separated list of differential diagnoses>
Clinical Reasoning: <step-by-step clinical explanation connecting symptoms to diagnosis>
Recommended Tests: <comma-separated list of confirmatory tests>

Do not include any conversational greeting or dialogue. Begin immediately with "Primary Diagnosis:"."""

body = {
    "model": OLLAMA_MODEL,
    "messages": [
        {"role": "system", "content": "You are a clinical diagnostic AI. You provide immediate structured diagnostic assessments without conversational filler."},
        {"role": "user", "content": user_prompt}
    ],
    "stream": False,
    "options": {
        "temperature": 0.0,
        "num_predict": 1024,
        "stop": ["User:", "Patient:"]
    }
}

print("Testing prompt format with stop tokens and strict instruction...", flush=True)
r = requests.post(OLLAMA_API_URL, json=body, timeout=120)
print("Status:", r.status_code)
content = r.json().get("message", {}).get("content", "")
print("Output:\n", content)

