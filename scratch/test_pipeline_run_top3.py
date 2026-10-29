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

inputs_file = BASE_DIR / "results" / "processed_inputs.txt"
results_base = BASE_DIR / "results"

with open(inputs_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(lines)} dialogues. Testing Process 1, 2, and 3...", flush=True)

for idx in [1, 2, 3]:
    dialogue_text = lines[idx - 1]
    process_folder = results_base / f"process_{idx}"
    process_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n==========================================")
    print(f"Running Process {idx} -> {process_folder}")
    print(f"==========================================")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(dialogue_text)
        temp_path = tf.name

    try:
        t0 = time.time()
        pipeline = PipelineProcessor(temp_path, str(process_folder))
        outputs = pipeline.run(
            output_filename="processed_dialogues.json",
            summary_filename="processing_summary.json",
            medical_report_filename="medical_report.json",
            diagnosis_filename="diagnosis_prediction.json"
        )
        elapsed = round(time.time() - t0, 1)

        # Check medical report
        med_path = process_folder / "medical_report.json"
        with open(med_path, "r", encoding="utf-8") as mf:
            med_data = json.load(mf)
        print(f"\n[Process {idx} Medical Report]")
        print("  Chief Complaint entries:", len(med_data.get("chief_complaint", [])))
        for cc in med_data.get("chief_complaint", []):
            print("   *", cc.get("sentence"))
        print("  Past Medical History entries:", len(med_data.get("past_medical_history", [])))
        for pmh in med_data.get("past_medical_history", []):
            print("   *", pmh.get("sentence"))

        # Check diagnosis
        dx_path = process_folder / "diagnosis_prediction.json"
        with open(dx_path, "r", encoding="utf-8") as df:
            dx_data = json.load(df)
        pred = dx_data.get("prediction", {})
        print(f"\n[Process {idx} Diagnosis]")
        print(f"  Primary Diagnosis: {pred.get('primary_diagnosis')}")
        print(f"  Secondary Diagnoses: {pred.get('secondary_diagnoses')}")
        print(f"  Clinical Reasoning: {str(pred.get('clinical_reasoning'))[:200]}...")
        print(f"  Recommended Tests: {pred.get('recommended_tests')}")
        print(f"  Model Used: {dx_data.get('model')}")
        print(f"  Time Elapsed: {elapsed}s")
        
    except Exception as e:
        print(f"Error on Process {idx}: {e}")
        import traceback
        traceback.print_exc()

