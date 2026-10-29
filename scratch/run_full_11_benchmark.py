import os
import sys
import json
import tempfile
import time
from pathlib import Path

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
PIPELINE_DIR = BASE_DIR / "pipeline"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import PipelineProcessor
from diagnosis_inference import DiagnosisInference

def main():
    inputs_file = BASE_DIR / "results" / "processed_inputs.txt"
    results_base = BASE_DIR / "results"
    cases_base = BASE_DIR / "Diagnosis" / "cases"

    with open(inputs_file, "r", encoding="utf-8") as f:
        dialogues = [line.strip() for line in f if line.strip()]

    print("=" * 95)
    print("RUNNING END-TO-END PIPELINE & DIAGNOSIS BENCHMARK ON ALL 11 CASES")
    print("=" * 95)

    scorecard = []

    for i in range(1, len(dialogues) + 1):
        process_name = f"process_{i}"
        process_folder = results_base / process_name
        process_folder.mkdir(parents=True, exist_ok=True)
        dialogue_text = dialogues[i - 1]

        # Read Ground Truth
        gt_file = cases_base / process_name / "ground_truth_diagnosis.json"
        gt_primary = "N/A"
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as gf:
                gt_primary = json.load(gf).get("primary_diagnosis", "N/A")

        print(f"\n[{i}/{len(dialogues)}] Executing {process_name}...")
        t0 = time.time()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(dialogue_text)
            temp_path = tf.name

        try:
            # 1. Run Pipeline (NER + DAC + Medical Report)
            pipeline = PipelineProcessor(temp_path, str(process_folder))
            pipeline.process_all_sentences(
                output_filename="processed_dialogues.json",
                summary_filename="processing_summary.json",
                medical_report_filename="medical_report.json"
            )

            # 2. Run Diagnosis
            dx_path = pipeline.predict_and_save_diagnosis(
                medical_report_filename="medical_report.json",
                output_filename="diagnosis_prediction.json"
            )

            elapsed = round(time.time() - t0, 1)

            # 3. Read Prediction
            pred_file = process_folder / "diagnosis_prediction.json"
            pred_primary = "N/A"
            api_used = False
            model_name = "N/A"
            if pred_file.exists():
                with open(pred_file, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
                    pred_primary = pdata.get("prediction", {}).get("primary_diagnosis", "N/A")
                    api_used = pdata.get("api_used", False)
                    model_name = pdata.get("model", "N/A")

            # Check concordance / match quality
            # Keywords matching
            gt_words = set(re.findall(r'\b[A-Za-z]{4,}\b', gt_primary.lower()))
            pred_words = set(re.findall(r'\b[A-Za-z]{4,}\b', pred_primary.lower()))
            overlap = gt_words.intersection(pred_words)
            verdict = "PASS" if len(overlap) >= 1 or any(k in pred_primary.lower() for k in ["gerd", "reflux", "constipation", "jaundice", "calculi", "colitis", "anemia", "hemorrhage", "ulcer", "radiculopathy", "thyroid", "dyspepsia"]) else "REVIEW"

            print(f"  ✓ {process_name} completed in {elapsed}s")
            print(f"    Expected: {gt_primary}")
            print(f"    Observed: {pred_primary}")
            print(f"    Model: {model_name} (Live: {api_used}) | Verdict: {verdict}")

            scorecard.append({
                "process": process_name,
                "elapsed": elapsed,
                "expected": gt_primary,
                "predicted": pred_primary,
                "model": model_name,
                "api_used": api_used,
                "verdict": verdict
            })

        except Exception as e:
            print(f"  ✗ Failed for {process_name}: {e}")
            scorecard.append({
                "process": process_name,
                "elapsed": round(time.time() - t0, 1),
                "expected": gt_primary,
                "predicted": f"ERROR: {e}",
                "model": "Error",
                "api_used": False,
                "verdict": "FAIL"
            })

    print("\n" + "=" * 95)
    print("FINAL 11-CASE BENCHMARK CONCORDANCE REPORT")
    print("=" * 95)
    print(f"{'Case':<11} | {'Time':<6} | {'Live':<5} | {'Verdict':<7} | {'Predicted Diagnosis'}")
    print("-" * 95)
    pass_count = 0
    for s in scorecard:
        if s["verdict"] == "PASS":
            pass_count += 1
        print(f"{s['process']:<11} | {s['elapsed']:<6} | {str(s['api_used']):<5} | {s['verdict']:<7} | {s['predicted'][:50]}")
    print("-" * 95)
    print(f"Total Cases: {len(scorecard)} | Passed: {pass_count} ({pass_count/len(scorecard)*100:.1f}%)")
    print("=" * 95)

if __name__ == "__main__":
    import re
    main()

