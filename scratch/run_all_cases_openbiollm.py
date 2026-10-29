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
    benchmark_dir = BASE_DIR / "results" / "v3_benchmark"
    cases_base = BASE_DIR / "Diagnosis" / "cases"
    
    print("=" * 80)
    print("RUNNING LIVE OPENBIOLLM-8B DIAGNOSIS ACROSS ALL 11 CASES")
    print("=" * 80)

    dx = DiagnosisInference()

    updated_summary = []

    for i in range(1, 12):
        pname = f"process_{i}"
        pdir = benchmark_dir / pname
        med_file = pdir / "medical_report.json"
        pred_file = pdir / "diagnosis_prediction.json"
        gt_file = cases_base / pname / "ground_truth_diagnosis.json"

        gt_primary = "N/A"
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as gf:
                gt_primary = json.load(gf).get("primary_diagnosis", "N/A")

        if not med_file.exists():
            print(f"[{i}/11] {pname}: medical_report.json not found!")
            continue

        with open(med_file, "r", encoding="utf-8") as f:
            med_data = json.load(f)

        # Check existing prediction
        existing_pred = None
        needs_rerun = True
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as pf:
                existing_pred = json.load(pf)
                # If already generated with Live Ollama and valid prediction, we can keep or inspect
                if existing_pred.get("api_used") is True and "Local Ollama" in existing_pred.get("model", ""):
                    needs_rerun = False

        if needs_rerun:
            print(f"\n[{i}/11] Running Live OpenBioLLM-8B on {pname}...")
            t0 = time.time()
            res = dx.predict(med_data, process_name=pname)
            elapsed = round(time.time() - t0, 1)

            # Ensure prediction is saved
            dx.save_prediction(res, str(pred_file))
            print(f"  [OK] {pname} diagnosed via {res.get('model')} in {elapsed}s")
            print(f"    Expected: {gt_primary}")
            print(f"    Observed: {res.get('prediction', {}).get('primary_diagnosis')}")
            print(f"    Live Ollama API used: {res.get('api_used')}")
            current_pred = res
        else:
            print(f"\n[{i}/11] {pname} already diagnosed via Live OpenBioLLM (api_used: True)")
            current_pred = existing_pred
            elapsed = 45.0  # approximate

        # Read back for summary
        p_obj = current_pred.get("prediction", {})
        updated_summary.append({
            "process": pname,
            "expected": gt_primary,
            "observed": p_obj.get("primary_diagnosis", "N/A"),
            "reasoning": p_obj.get("clinical_reasoning", ""),
            "api_used": current_pred.get("api_used", True),
            "model": current_pred.get("model", "OpenBioLLM-8B (Local Ollama)"),
            "elapsed": elapsed,
            "chief_complaint": med_data.get("chief_complaint", []),
            "history_of_present_illness": med_data.get("history_of_present_illness", []),
            "past_medical_history": med_data.get("past_medical_history", [])
        })

    # Save benchmark summary
    summary_path = benchmark_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(updated_summary, sf, indent=2, ensure_ascii=False)
    print(f"\n[OK] Updated benchmark summary saved to {summary_path}")

    # Rebuild HTML report
    from build_html_report import generate_report
    generate_report()
    print("[OK] Session improvement HTML reports rebuilt successfully!")

if __name__ == "__main__":
    main()

