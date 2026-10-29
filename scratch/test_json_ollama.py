import json
import requests

prompt = """Analyze the following patient findings and return a JSON differential diagnosis:

CHIEF COMPLAINT:
- She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days
- She is having gas.

PAST MEDICAL HISTORY:
- In 2018, rheumatoid arthritis on medication
- Stone in gallbladder (GB) found on USG

Return a JSON object in this exact structure:
{
  "primary_diagnosis": "<condition name>",
  "secondary_diagnoses": ["<differential 1>", "<differential 2>"],
  "clinical_reasoning": "<clinical rationale>",
  "recommended_tests": ["<test 1>", "<test 2>"]
}"""

body = {
    "model": "hf.co/bartowski/OpenBioLLM-Llama3-8B-GGUF:Q4_K_M",
    "messages": [
        {"role": "system", "content": "You are an expert clinical diagnostic physician. Respond ONLY with a valid JSON object matching the requested schema. No conversational filler."},
        {"role": "user", "content": prompt}
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 512}
}

res = requests.post("http://localhost:11434/api/chat", json=body, timeout=60)
print("status:", res.status_code)
content = res.json().get("message", {}).get("content", "")
print("content:")
print(content)

try:
    d = json.loads(content)
    print("Parsed JSON successfully!")
    print("primary_diagnosis:", d.get("primary_diagnosis"))
except Exception as e:
    print("JSON parse error:", e)

