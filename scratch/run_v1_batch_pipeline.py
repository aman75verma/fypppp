"""
Batch Pipeline Runner for V1 Models
====================================
1. Cleans up any fragmented folders in results/
2. Sets up the 11 true benchmark clinical dialogues from processed_inputs.txt
3. Copies ground-truth annotated_dialogues.json for Manual vs Model evaluation
4. Runs full end-to-end pipeline with V1 models (PubMedBERT NER v1 + PubMedBERT DAC v1)
5. Predicts and saves AI Clinical Diagnosis (OpenBioLLM-8B) for each dialogue
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path("c:/Users/ammua/Desktop/final_prompt")
MAIN_FYP_RESULTS = Path("C:/Users/ammua/Desktop/main_fyp/TRAINED-PIPELINE/results")
RESULTS_BASE = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from pipeline import PipelineProcessor
from sentence_processor import load_dialogues
from diagnosis_inference import DiagnosisInference

def setup_environment():
    print("=" * 80)
    print("STEP 1: Setting up 11 Benchmarking Dialogues & Cleaning Up Stale Folders")
    print("=" * 80)
    
    os.makedirs(RESULTS_BASE, exist_ok=True)
    
    # 1. Clean up old process_ folders (especially process_12..25 and any fragments)
    for item in os.listdir(RESULTS_BASE):
        item_path = RESULTS_BASE / item
        if item_path.is_dir() and item.startswith("process_"):
            shutil.rmtree(item_path)
            print(f"  Removed old folder: {item}")
            
    # 2. Copy processed_inputs.txt
    src_inputs = MAIN_FYP_RESULTS / "processed_inputs.txt"
    dest_inputs = RESULTS_BASE / "processed_inputs.txt"
    if src_inputs.exists():
        shutil.copy2(src_inputs, dest_inputs)
        print(f"✓ Copied {src_inputs} -> {dest_inputs}")
    else:
        print(f"Error: Could not find {src_inputs}")
        sys.exit(1)

def run_v1_pipeline():
    print("\n" + "=" * 80)
    print("STEP 2: Running Pipeline on All 11 Dialogues with V1 Models")
    print("=" * 80)
    
    input_file = RESULTS_BASE / "processed_inputs.txt"
    with open(input_file, "r", encoding="utf-8") as f:
        dialogues = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(dialogues)} dialogues to process.\n")
    
    dx_engine = DiagnosisInference()
    summary_report = []
    
    for idx, dialogue_text in enumerate(dialogues, 1):
        folder_name = f"process_{idx}"
        output_dir = RESULTS_BASE / folder_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy ground-truth annotations if available
        src_ann = MAIN_FYP_RESULTS / folder_name / "annotated_dialogues.json"
        dest_ann = output_dir / "annotated_dialogues.json"
        if src_ann.exists():
            shutil.copy2(src_ann, dest_ann)
            
        print(f"\n[{idx}/{len(dialogues)}] Processing Dialogue -> results/{folder_name}/")
        print("-" * 60)
        
        # Temp file for dialogue text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
            tf.write(dialogue_text)
            temp_path = tf.name
            
        try:
            pipeline_proc = PipelineProcessor(temp_path, str(output_dir))
            sentences = load_dialogues(temp_path)
            num_sents = len(sentences)
            print(f"  Loaded {num_sents} sentences")
            
            sentence_history = []
            success_count = 0
            
            for s_idx, s_data in enumerate(sentences, 1):
                result = pipeline_proc.process_sentence(s_data, history=sentence_history)
                sentence_history.append(s_data['sentence'])
                
                if result['status'] == 'success':
                    pipeline_proc.save_to_json(result)
                    pipeline_proc.update_summary(result)
                    pipeline_proc.append_to_medical_report(result)
                    success_count += 1
                else:
                    pipeline_proc.update_summary(result)
                    
            # Save medical report
            med_rep_file = pipeline_proc.save_medical_report()
            print(f"  ✓ Processed: {success_count}/{num_sents} sentences")
            
            # Run AI Diagnosis Prediction
            print(f"  🩺 Generating AI Diagnosis with OpenBioLLM...")
            dx_file = pipeline_proc.predict_and_save_diagnosis()
            
            # Load diagnosis result
            primary_dx = "N/A"
            if dx_file and os.path.exists(dx_file):
                with open(dx_file, 'r', encoding='utf-8') as f:
                    dx_obj = json.load(f)
                    primary_dx = dx_obj.get('prediction', {}).get('primary_diagnosis', 'N/A')
                    
            summary_report.append({
                "process": folder_name,
                "sentences": num_sents,
                "success": success_count,
                "primary_diagnosis": primary_dx
            })
            print(f"  ✓ Diagnosis: {primary_dx}")
            
        except Exception as e:
            print(f"  ✗ Error in {folder_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    print("\n" + "=" * 80)
    print("V1 PIPELINE RUN & DIAGNOSIS PREDICTION COMPLETED!")
    print("=" * 80)
    for r in summary_report:
        print(f"  {r['process']}: {r['success']}/{r['sentences']} sentences -> Diagnosis: {r['primary_diagnosis']}")

if __name__ == "__main__":
    setup_environment()
    run_v1_pipeline()
