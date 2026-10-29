"""
Batch AI Diagnosis Runner with Local OpenBioLLM-8B
==================================================
Runs real local OpenBioLLM diagnosis inference for all 11 processes.
"""

import os
import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
PIPELINE_DIR = BASE_DIR / "pipeline"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from diagnosis_inference import DiagnosisInference

def main():
    results_base = BASE_DIR / "results"
    dx = DiagnosisInference()
    
    print("=" * 80)
    print("RUNNING LIVE OPENBIOLLM-8B DIAGNOSIS FOR ALL 11 PROCESSES")
    print("=" * 80)

    summary = []

    for i in range(1, 12):
        folder_name = f"process_{i}"
        folder_path = results_base / folder_name
        report_file = folder_path / "medical_report.json"
        out_file = folder_path / "diagnosis_prediction.json"

        if not report_file.exists():
            print(f"[{i}/11] Skipping {folder_name} (no medical_report.json)")
            continue

        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        print(f"\n[{i}/11] Diagnosing {folder_name} with OpenBioLLM-8B...")
        t0 = time.time()
        
        try:
            res = dx.predict(report_data, process_name=folder_name)
            dx.save_prediction(res, str(out_file))
            elapsed = round(time.time() - t0, 1)

            pred = res.get("prediction", {})
            primary = pred.get("primary_diagnosis", "N/A")
            api_used = res.get("api_used", False)
            model_name = res.get("model", "N/A")

            print(f"  ✓ {folder_name} completed in {elapsed}s")
            print(f"  🩺 Primary Diagnosis: {primary}")
            print(f"  🤖 Model: {model_name} | Live LLM: {api_used}")

            summary.append({
                "process": folder_name,
                "elapsed": elapsed,
                "primary": primary,
                "api_used": api_used,
                "model": model_name
            })
        except Exception as e:
            print(f"  ✗ Error on {folder_name}: {e}")
            summary.append({
                "process": folder_name,
                "elapsed": round(time.time() - t0, 1),
                "primary": f"ERROR: {e}",
                "api_used": False,
                "model": "Failed"
            })

    print("\n" + "=" * 80)
    print("DIAGNOSIS INFERENCE BATCH SUMMARY")
    print("=" * 80)
    print(f"{'Folder':<12} | {'Time (s)':<8} | {'Live LLM':<8} | {'Primary Diagnosis'}")
    print("-" * 80)
    for s in summary:
        print(f"{s['process']:<12} | {s['elapsed']:<8} | {str(s['api_used']):<8} | {s['primary']}")
    print("-" * 80)

if __name__ == "__main__":
    main()
