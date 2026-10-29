import requests
import json

prompt = """Analyze the following patient findings and provide a differential diagnosis:

CHIEF COMPLAINT:
  - She has severe constipation--meaning she isn't able to pass stool, and it happens only once every 2-3 days
  - She is having gas.

PAST MEDICAL HISTORY:
  - In 2018, she was diagnosed with rheumatoid arthritis on medication.
  - An ultrasound (USG) previously found a gallstone in the gallbladder (GB), but no acute pain.

Provide your output in valid JSON matching this exact structure:
{
  "primary_diagnosis": "<specific formal disease or clinical syndrome name only>",
  "secondary_diagnoses": ["differential 1", "differential 2"],
  "clinical_reasoning": "<detailed clinical rationale>",
  "recommended_tests": ["test 1", "test 2"]
}
CRITICAL REQUIREMENT: 'primary_diagnosis' MUST be a concise, standard medical disease or syndrome name (for example: 'Cholelithiasis with Functional Constipation' or 'Irritable Bowel Syndrome - Constipation Predominant'). Do NOT write a sentence or describe symptoms in primary_diagnosis."""

body = {
    "model": "hf.co/bartowski/OpenBioLLM-Llama3-8B-GGUF:Q4_K_M",
    "messages": [
        {"role": "system", "content": "You are an expert physician and clinical gastroenterology specialist."},
        {"role": "user", "content": prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 1024}
}

res = requests.post("http://localhost:11434/api/chat", json=body, timeout=120)
print("Status:", res.status_code)
content = res.json().get("message", {}).get("content", "")
print("Response content:")
print(content)

