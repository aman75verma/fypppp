"""
Organize Diagnosis Folder:
Populate the Diagnosis directory with:
1. 11 doctor-patient transcripts (formatted, raw, and json)
2. Medical reports for all 11 transcripts
3. Ground truth diagnosis for all 11 transcripts
4. OpenLLM diagnosis (OpenBioLLM-8B) for all 11 transcripts
5. Per-case organized bundles (Diagnosis/cases/process_X/)
6. Consolidated files and comprehensive documentation (README.md)
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = Path("c:/Users/ammua/Desktop/final_prompt")
DIAGNOSIS_DIR = BASE_DIR / "Diagnosis"
RESULTS_DIR = BASE_DIR / "results"

def format_transcript_turns(raw_text: str) -> str:
    """Format single-line dialogue into readable speaker turns."""
    turns = re.findall(r'(Doctor:|Patient:)(.*?)(?=(?:Doctor:|Patient:|$))', raw_text, re.DOTALL)
    if not turns:
        return raw_text.strip()
    formatted = []
    for speaker, content in turns:
        formatted.append(f"{speaker} {content.strip()}")
    return "\n\n".join(formatted)

def main():
    print(f"Organizing Diagnosis directory: {DIAGNOSIS_DIR}")
    
    # Target folders
    transcripts_dir = DIAGNOSIS_DIR / "transcripts"
    med_reports_dir = DIAGNOSIS_DIR / "medical_reports"
    gt_dir = DIAGNOSIS_DIR / "ground_truth_diagnosis"
    llm_dir = DIAGNOSIS_DIR / "openllm_diagnosis"
    cases_dir = DIAGNOSIS_DIR / "cases"
    
    for d in [transcripts_dir, med_reports_dir, gt_dir, llm_dir, cases_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Transcripts
    inputs_file = RESULTS_DIR / "processed_inputs.txt"
    with open(inputs_file, "r", encoding="utf-8") as f:
        raw_transcripts = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(raw_transcripts)} transcripts from {inputs_file}")
    
    # 2. Load Existing Ground Truth
    gt_file = DIAGNOSIS_DIR / "ground_truth_diagnosis.json"
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            all_gt_raw = json.load(f)
    else:
        all_gt_raw = {}
        
    # 3. Load Existing Comparison if available
    comp_file = DIAGNOSIS_DIR / "diagnosis_comparison.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            all_comp = json.load(f)
    else:
        all_comp = {}
        
    consolidated_transcripts = {}
    consolidated_reports = {}
    consolidated_gt = {}
    consolidated_llm = {}
    consolidated_cases = {}
    
    for idx in range(1, 12):
        proc_name = f"process_{idx}"
        case_dir = cases_dir / proc_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # --- A. TRANSCRIPT ---
        raw_text = raw_transcripts[idx - 1] if idx - 1 < len(raw_transcripts) else ""
        formatted_text = format_transcript_turns(raw_text)
        
        # Save per-transcript text file in transcripts/
        t_file_path = transcripts_dir / f"transcript_{idx}.txt"
        with open(t_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Process {idx} Transcript\n")
            f.write(f"# Source: results/processed_inputs.txt (Line {idx})\n\n")
            f.write(formatted_text)
            f.write("\n")
            
        # Save in case folder
        with open(case_dir / "transcript.txt", "w", encoding="utf-8") as f:
            f.write(f"# Process {idx} Transcript\n\n")
            f.write(formatted_text)
            f.write("\n")
            
        consolidated_transcripts[proc_name] = {
            "process_id": proc_name,
            "dialogue_index": idx,
            "raw_text": raw_text,
            "formatted_text": formatted_text
        }
        
        # --- B. MEDICAL REPORT ---
        src_report = RESULTS_DIR / proc_name / "medical_report.json"
        if src_report.exists():
            with open(src_report, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
        else:
            rep_data = {}
            
        dest_report = med_reports_dir / f"medical_report_{idx}.json"
        with open(dest_report, "w", encoding="utf-8") as f:
            json.dump(rep_data, f, indent=2, ensure_ascii=False)
            
        with open(case_dir / "medical_report.json", "w", encoding="utf-8") as f:
            json.dump(rep_data, f, indent=2, ensure_ascii=False)
            
        consolidated_reports[proc_name] = rep_data
        
        # --- C. GROUND TRUTH DIAGNOSIS ---
        gt_obj = None
        if proc_name in all_gt_raw:
            gt_obj = all_gt_raw[proc_name].get("ground_truth_diagnosis", all_gt_raw[proc_name])
        elif proc_name in all_comp and "ground_truth" in all_comp[proc_name]:
            gt_obj = all_comp[proc_name]["ground_truth"]
            
        if not gt_obj:
            gt_obj = {"primary_diagnosis": "N/A", "secondary_diagnoses": [], "clinical_reasoning": "", "recommended_tests": []}
            
        dest_gt = gt_dir / f"ground_truth_{idx}.json"
        with open(dest_gt, "w", encoding="utf-8") as f:
            json.dump({
                "process_id": proc_name,
                "source": f"results/{proc_name}/medical_report.json",
                "ground_truth_diagnosis": gt_obj
            }, f, indent=2, ensure_ascii=False)
            
        with open(case_dir / "ground_truth_diagnosis.json", "w", encoding="utf-8") as f:
            json.dump(gt_obj, f, indent=2, ensure_ascii=False)
            
        consolidated_gt[proc_name] = {
            "source": f"results/{proc_name}/medical_report.json",
            "ground_truth_diagnosis": gt_obj
        }
        
        # --- D. OPENLLM DIAGNOSIS ---
        src_pred = RESULTS_DIR / proc_name / "diagnosis_prediction.json"
        if src_pred.exists():
            with open(src_pred, "r", encoding="utf-8") as f:
                pred_file_data = json.load(f)
        else:
            pred_file_data = {}
            
        dest_llm = llm_dir / f"openllm_diagnosis_{idx}.json"
        with open(dest_llm, "w", encoding="utf-8") as f:
            json.dump(pred_file_data, f, indent=2, ensure_ascii=False)
            
        with open(case_dir / "openllm_diagnosis.json", "w", encoding="utf-8") as f:
            json.dump(pred_file_data, f, indent=2, ensure_ascii=False)
            
        consolidated_llm[proc_name] = pred_file_data
        
        # --- E. CASE SUMMARY (All in One) ---
        case_summary = {
            "process_id": proc_name,
            "dialogue_index": idx,
            "transcript_summary": {
                "num_characters": len(raw_text),
                "preview": raw_text[:120] + "..." if len(raw_text) > 120 else raw_text
            },
            "ground_truth": gt_obj,
            "openllm_prediction": pred_file_data.get("prediction", {}),
            "model_metadata": {
                "model": pred_file_data.get("model", "OpenBioLLM-8B"),
                "api_used": pred_file_data.get("api_used", False)
            }
        }
        with open(case_dir / "case_summary.json", "w", encoding="utf-8") as f:
            json.dump(case_summary, f, indent=2, ensure_ascii=False)
            
        consolidated_cases[proc_name] = case_summary
        print(f"✓ Processed case {proc_name}: transcript, report, ground truth, openllm diagnosis copied.")

    # Save Consolidated Files in Category Folders
    with open(transcripts_dir / "all_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_transcripts, f, indent=2, ensure_ascii=False)
        
    with open(transcripts_dir / "all_transcripts.txt", "w", encoding="utf-8") as f:
        for idx in range(1, 12):
            p = f"process_{idx}"
            f.write("=" * 80 + "\n")
            f.write(f"PROCESS {idx} TRANSCRIPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(consolidated_transcripts[p]["formatted_text"])
            f.write("\n\n\n")
            
    with open(med_reports_dir / "all_medical_reports.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_reports, f, indent=2, ensure_ascii=False)
        
    with open(gt_dir / "all_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_gt, f, indent=2, ensure_ascii=False)
        
    with open(llm_dir / "all_openllm_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_llm, f, indent=2, ensure_ascii=False)
        
    # Also save Root Level Convenient Files in Diagnosis/
    with open(DIAGNOSIS_DIR / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_transcripts, f, indent=2, ensure_ascii=False)
        
    with open(DIAGNOSIS_DIR / "medical_reports.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_reports, f, indent=2, ensure_ascii=False)
        
    with open(DIAGNOSIS_DIR / "openllm_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(consolidated_llm, f, indent=2, ensure_ascii=False)
        
    # Update llm_predicted_diagnosis.json to also have all 11 for backwards-compatibility
    with open(DIAGNOSIS_DIR / "llm_predicted_diagnosis.json", "w", encoding="utf-8") as f:
        # Formatted like previous llm_predicted_diagnosis.json
        legacy_llm = {}
        for p, d in consolidated_llm.items():
            legacy_llm[p] = {
                "clinical_summary_sent_to_llm": d.get("clinical_summary", ""),
                "llm_prediction": d.get("prediction", {}),
                "model": d.get("model", "OpenBioLLM-8B")
            }
        json.dump(legacy_llm, f, indent=2, ensure_ascii=False)
        
    # Generate README.md in Diagnosis/
    readme_path = DIAGNOSIS_DIR / "README.md"
    generate_readme(readme_path, consolidated_cases)
    print(f"✓ Generated README documentation at {readme_path}")
    print("\nORGANIZATION COMPLETE! All 11 cases fully consolidated.")

def generate_readme(readme_path: Path, cases: dict):
    md = []
    md.append("# Diagnosis Module: 11 Clinical Benchmarks\n")
    md.append("This directory organizes all clinical datasets, pipeline-extracted medical reports, expert physician ground truth diagnoses, and OpenBioLLM-8B predicted diagnoses across all **11 benchmark patient dialogues**.\n")
    md.append("## 📂 Folder Structure\n")
    md.append("```\nDiagnosis/\n├── cases/                          # Individual patient cases (process_1 to process_11)\n│   ├── process_1/\n│   │   ├── transcript.txt          # Formatted doctor-patient dialogue\n│   │   ├── medical_report.json     # Extracted clinical report from pipeline\n│   │   ├── ground_truth_diagnosis.json\n│   │   ├── openllm_diagnosis.json  # OpenBioLLM-8B clinical prediction\n│   │   └── case_summary.json       # Combined case bundle with comparison\n│   └── ... (process_2 to process_11)\n├── transcripts/                    # All 11 raw & formatted transcripts\n│   ├── transcript_1.txt ... transcript_11.txt\n│   ├── all_transcripts.txt\n│   └── all_transcripts.json\n├── medical_reports/                # All 11 structured medical reports\n│   ├── medical_report_1.json ... medical_report_11.json\n│   └── all_medical_reports.json\n├── ground_truth_diagnosis/         # Expert clinical ground truth diagnoses\n│   ├── ground_truth_1.json ... ground_truth_11.json\n│   └── all_ground_truth.json\n├── openllm_diagnosis/              # OpenBioLLM-8B predicted diagnoses\n│   ├── openllm_diagnosis_1.json ... openllm_diagnosis_11.json\n│   └── all_openllm_diagnosis.json\n├── ground_truth_diagnosis.json     # Master Ground Truth dataset\n├── openllm_diagnosis.json          # Master OpenLLM prediction dataset\n├── medical_reports.json            # Master extracted medical reports\n├── transcripts.json                # Master dialogue transcripts\n├── diagnosis_comparison.json       # Side-by-side benchmark comparison data\n├── diagnosis_comparison.html       # Interactive visual benchmark dashboard\n└── README.md                       # This documentation\n```\n")
    md.append("## 📊 Summary Comparison Table (All 11 Cases)\n\n")
    md.append("| Case | Process ID | Ground Truth Primary Diagnosis | OpenBioLLM-8B Predicted Diagnosis |\n")
    md.append("| :--- | :--- | :--- | :--- |\n")
    
    for idx in range(1, 12):
        p = f"process_{idx}"
        case = cases.get(p, {})
        gt_dx = case.get("ground_truth", {}).get("primary_diagnosis", "N/A")
        llm_dx = case.get("openllm_prediction", {}).get("primary_diagnosis", "N/A")
        # shorten if too long
        if len(llm_dx) > 75:
            llm_dx = llm_dx[:72] + "..."
        md.append(f"| **Case {idx:02d}** | `{p}` | {gt_dx} | {llm_dx} |\n")
        
    md.append("\n## 🌐 Visual Dashboard\n")
    md.append("Open `diagnosis_comparison.html` in your web browser to interactively compare ground truth and LLM predictions, clinical reasoning, secondary diagnoses, and recommended tests.\n")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(md)

if __name__ == "__main__":
    main()
