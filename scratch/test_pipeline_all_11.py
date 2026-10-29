import json
from pathlib import Path
import sys

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
PIPELINE_DIR = BASE_DIR / "pipeline"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from diagnosis_inference import DiagnosisInference

dx = DiagnosisInference()

print("=" * 75)
print("TESTING FULL DIAGNOSIS INFERENCE PIPELINE ACROSS ALL 11 CASES")
print("=" * 75)

with open(BASE_DIR / "results" / "v3_benchmark" / "benchmark_summary.json", "r", encoding="utf-8") as f:
    bench = json.load(f)

for item in bench:
    pname = item["process"]
    cid = int(pname.split("_")[1])
    with open(BASE_DIR / "results" / "v3_benchmark" / pname / "medical_report.json", "r", encoding="utf-8") as mf:
        med = json.load(mf)
    
    res = dx.predict(med, pname)
    pred = res.get("prediction", {})
    prim = pred.get("primary_diagnosis", "N/A")
    sec = pred.get("secondary_diagnoses", [])
    model_str = res.get("model", "N/A")

    print(f"[{cid:02d}/11] {pname}")
    print(f"  Expected: {item['expected']}")
    print(f"  Observed: {prim}  [{model_str}]")
    print(f"  Differentials: {sec[:2]}")
