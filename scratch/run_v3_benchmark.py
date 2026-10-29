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
    cases_base = BASE_DIR / "Diagnosis" / "cases"
    benchmark_out = BASE_DIR / "results" / "v3_benchmark"
    benchmark_out.mkdir(parents=True, exist_ok=True)

    with open(inputs_file, "r", encoding="utf-8") as f:
        dialogues = [line.strip() for line in f if line.strip()]

    print("=" * 95)
    print("RUNNING END-TO-END PIPELINE & DIAGNOSIS BENCHMARK (DAC v3 + NER v2/v3 + OpenBioLLM-8B)")
    print("=" * 95)

    scorecard = []

    for i in range(1, len(dialogues) + 1):
        process_name = f"process_{i}"
        dialogue_text = dialogues[i - 1]
        process_folder = benchmark_out / process_name
        process_folder.mkdir(parents=True, exist_ok=True)

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
            # 1. Run Pipeline (NER v2 + DAC v3 + Medical Report)
            pipeline = PipelineProcessor(temp_path, str(process_folder))
            pipeline.process_all_sentences(
                output_filename="processed_dialogues.json",
                summary_filename="processing_summary.json",
                medical_report_filename="medical_report.json"
            )

            # 2. Run Diagnosis (OpenBioLLM-8B)
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
            reasoning = ""
            if pred_file.exists():
                with open(pred_file, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
                    pred_primary = pdata.get("prediction", {}).get("primary_diagnosis", "N/A")
                    reasoning = pdata.get("prediction", {}).get("clinical_reasoning", "")
                    api_used = pdata.get("metadata", {}).get("api_used", False)
                    model_name = pdata.get("metadata", {}).get("model", "N/A")

            # Check medical report
            rep_file = process_folder / "medical_report.json"
            rep_data = {}
            if rep_file.exists():
                with open(rep_file, "r", encoding="utf-8") as rf:
                    rep_data = json.load(rf)

            print(f"  [OK] {process_name} completed in {elapsed}s")
            print(f"    Expected: {gt_primary}")
            print(f"    Observed: {pred_primary}")
            print(f"    Model: {model_name} (Live: {api_used})")

            scorecard.append({
                "process": process_name,
                "expected": gt_primary,
                "observed": pred_primary,
                "reasoning": reasoning,
                "api_used": api_used,
                "model": model_name,
                "elapsed": elapsed,
                "chief_complaint": rep_data.get("chief_complaint", []),
                "history_of_present_illness": rep_data.get("history_of_present_illness", []),
                "past_medical_history": rep_data.get("past_medical_history", [])
            })

        except Exception as e:
            print(f"  [ERR] {process_name} failed: {e}")
            import traceback
            traceback.print_exc()
            scorecard.append({
                "process": process_name,
                "expected": gt_primary,
                "observed": f"ERROR: {e}",
                "elapsed": round(time.time() - t0, 1)
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Save detailed benchmark results
    with open(benchmark_out / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print("\n" + "=" * 95)
    print("BENCHMARK EXECUTION COMPLETE! Saved to results/v3_benchmark/benchmark_summary.json")
    print("=" * 95)

if __name__ == "__main__":
    main()

