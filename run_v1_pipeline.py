"""
Batch Runner for V1 Pipeline + AI Diagnosis
===========================================
Processes all 11 dialogues from results/processed_inputs.txt
using V1 NER, V1 DAC, and local OpenBioLLM-8B for AI Diagnosis.
Saves outputs directly to results/process_1 through results/process_11.
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path

# Add project root and pipeline directory to path
BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
PIPELINE_DIR = BASE_DIR / "pipeline"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import PipelineProcessor
from diagnosis_inference import DiagnosisInference

def main():
    inputs_file = BASE_DIR / "results" / "processed_inputs.txt"
    results_base = BASE_DIR / "results"

    if not inputs_file.exists():
        print(f"Error: Could not find inputs file: {inputs_file}")
        return

    with open(inputs_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} dialogues from {inputs_file}\n")
    print("=" * 80)
    print("STARTING V1 PIPELINE + OPENBIOLLM DIAGNOSIS RUN")
    print("=" * 80)

    summary = []

    for idx, dialogue_text in enumerate(lines, 1):
        process_folder = results_base / f"process_{idx}"
        process_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idx}/{len(lines)}] Processing dialogue -> {process_folder.name}/")
        start_time = time.time()

        # Write temp dialogue file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(dialogue_text)
            temp_path = tf.name

        try:
            # 1. Run V1 Pipeline (NER + DAC + Medical Report)
            pipeline = PipelineProcessor(temp_path, str(process_folder))
            outputs = pipeline.run(
                output_filename="processed_dialogues.json",
                summary_filename="processing_summary.json",
                medical_report_filename="medical_report.json",
                diagnosis_filename="diagnosis_prediction.json"
            )

            # Check diagnosis prediction
            dx_file = process_folder / "diagnosis_prediction.json"
            primary_dx = "N/A"
            api_used = False
            model_used = "N/A"
            if dx_file.exists():
                with open(dx_file, "r", encoding="utf-8") as df:
                    dx_data = json.load(df)
                    primary_dx = dx_data.get("prediction", {}).get("primary_diagnosis", "N/A")
                    api_used = dx_data.get("api_used", False)
                    model_used = dx_data.get("model", "N/A")

            elapsed = round(time.time() - start_time, 1)
            print(f"  ✓ Processed in {elapsed}s")
            print(f"  🩺 Primary Diagnosis: {primary_dx}")
            print(f"  🤖 Model: {model_used} | Live Inference: {api_used}")

            summary.append({
                "process": f"process_{idx}",
                "status": "SUCCESS",
                "elapsed_s": elapsed,
                "primary_diagnosis": primary_dx,
                "model": model_used,
                "api_used": api_used
            })

        except Exception as e:
            print(f"  ✗ FAILED for process_{idx}: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                "process": f"process_{idx}",
                "status": f"ERROR: {str(e)}",
                "elapsed_s": round(time.time() - start_time, 1),
                "primary_diagnosis": "ERROR",
                "model": "None",
                "api_used": False
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print("\n" + "=" * 80)
    print("ALL 11 PROCESSES COMPLETED!")
    print("=" * 80)
    print(f"{'Folder':<12} | {'Status':<8} | {'Time (s)':<8} | {'Live LLM':<8} | {'Primary Diagnosis'}")
    print("-" * 80)
    for s in summary:
        print(f"{s['process']:<12} | {s['status']:<8} | {s['elapsed_s']:<8} | {str(s['api_used']):<8} | {s['primary_diagnosis']}")
    print("-" * 80)

if __name__ == "__main__":
    main()
