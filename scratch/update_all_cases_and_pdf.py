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
from scratch.build_doctor_dossier import format_annotated_report, CASE_METADATA
from scratch.generate_custom_pdf import generate_pdf_report

def main():
    benchmark_dir = BASE_DIR / "results" / "v3_benchmark"
    dossier_dir = BASE_DIR / "doctor_clinical_evaluation_dossier"
    
    print("=" * 85)
    print("RE-RUNNING STRICT OPENBIOLLM-8B DIAGNOSES ACROSS ALL 11 CASES")
    print("=" * 85)

    dx = DiagnosisInference()

    updated_summary = []

    for meta in CASE_METADATA:
        cid = meta["id"]
        pname = f"process_{cid}"
        cfolder = dossier_dir / meta["folder_name"]
        
        pdir = benchmark_dir / pname
        med_file = pdir / "medical_report.json"
        pred_file = pdir / "diagnosis_prediction.json"
        
        if not med_file.exists():
            print(f"[{cid}/11] {pname}: medical_report.json not found!")
            continue

        with open(med_file, "r", encoding="utf-8") as f:
            med_data = json.load(f)

        print(f"\n[{cid}/11] Diagnosing {meta['title']} via OpenBioLLM-8B...", flush=True)
        t0 = time.time()
        res = dx.predict(med_data, process_name=pname)
        elapsed = round(time.time() - t0, 1)

        prim = res.get("prediction", {}).get("primary_diagnosis", "N/A")
        sec = res.get("prediction", {}).get("secondary_diagnoses", [])
        
        print(f"  [OK] Done in {elapsed}s", flush=True)
        print(f"    Expected: {meta['expected_dx']}", flush=True)
        print(f"    Observed: {prim}", flush=True)
        print(f"    Differentials: {sec[:3]}", flush=True)

        # Save to benchmark folder
        dx.save_prediction(res, str(pred_file))

        # Save to doctor dossier folder
        if cfolder.exists():
            dossier_pred_file = cfolder / "openbiollm_diagnosis.json"
            dx.save_prediction(res, str(dossier_pred_file))
            
            # Update annotated report
            ann_file = cfolder / "annotated_report.txt"
            ann_text = format_annotated_report(med_data, res, meta)
            with open(ann_file, "w", encoding="utf-8") as af:
                af.write(ann_text)

        # Append to summary
        updated_summary.append({
            "process": pname,
            "expected": meta["expected_dx"],
            "observed": prim,
            "reasoning": res.get("prediction", {}).get("clinical_reasoning", ""),
            "api_used": res.get("api_used", True),
            "model": res.get("model", "OpenBioLLM-8B (Local Ollama)"),
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

    # Re-generate the custom PDF
    print("\nRe-compiling updated PDF report...")
    generate_pdf_report()
    print("[OK] PDF Report successfully regenerated and copied to root and dossier!")

if __name__ == "__main__":
    main()

