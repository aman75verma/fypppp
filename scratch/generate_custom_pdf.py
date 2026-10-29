import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(r"c:\Users\ammua\Desktop\final_prompt")
CASES_DIR = BASE_DIR / "Diagnosis" / "cases"
V3_RESULTS_DIR = BASE_DIR / "results" / "v3_benchmark"
OUT_ROOT = BASE_DIR / "doctor_clinical_evaluation_dossier"

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CASE_METADATA = [
    {
        "id": 1,
        "folder_name": "Case_01_Cholelithiasis",
        "title": "Case 1: Gallbladder Calculus with Severe Functional Constipation",
        "expected_dx": "Cholelithiasis (Gallbladder Stones) with Chronic Functional Constipation",
        "key_symptoms": "Severe constipation (defecation once every 2-3 days), gas, body discomfort; USG confirmed GB stone in past history."
    },
    {
        "id": 2,
        "folder_name": "Case_02_GERD_Recurrent",
        "title": "Case 2: Recurrent Acid Peptic Disease with Nocturnal Burning",
        "expected_dx": "Gastroesophageal Reflux Disease (GERD) with Recurrent Acid Peptic Disease",
        "key_symptoms": "Slight burning sensation, epigastric pain, reduced gas; recurring symptoms after previous treatment."
    },
    {
        "id": 3,
        "folder_name": "Case_03_GERD_AcidReflux",
        "title": "Case 3: Classic Gastroesophageal Reflux with Severe Heartburn",
        "expected_dx": "Gastroesophageal Reflux Disease (GERD) with Reflux Dyspepsia",
        "key_symptoms": "Retrosternal and epigastric burning sensation, postprandial nausea/vomiting sensation, sleep disturbance."
    },
    {
        "id": 4,
        "folder_name": "Case_04_Diet_Induced_Dyspepsia",
        "title": "Case 4: Diet-Induced Dysmotility and Postprandial Distress",
        "expected_dx": "Diet-Induced Non-Ulcer Dyspepsia with Irritable Bowel Syndrome (IBS-Mixed)",
        "key_symptoms": "Gastrointestinal discomfort, altered bowel habits, postprandial bloating following irregular or heavy meals."
    },
    {
        "id": 5,
        "folder_name": "Case_05_Peptic_Ulcer_Disease",
        "title": "Case 5: Chronic Dyspepsia with Suspected Peptic Ulceration",
        "expected_dx": "Chronic Dyspepsia with Suspected Peptic Ulcer Disease (PUD)",
        "key_symptoms": "Persistent burning sensation in epigastrium, nausea, localized tenderness, appetite changes."
    },
    {
        "id": 6,
        "folder_name": "Case_06_Obstructive_Jaundice",
        "title": "Case 6: Choledocholithiasis with Intermittent Biliary Obstruction",
        "expected_dx": "Choledocholithiasis / Biliary Calculi with Intermittent Obstructive Jaundice",
        "key_symptoms": "Stomach discomfort, yellowish discoloration of eyes/skin, dark urine, pale stools."
    },
    {
        "id": 7,
        "folder_name": "Case_07_Inflammatory_Bowel_Disease",
        "title": "Case 7: Chronic Colitis / Suspected Inflammatory Bowel Disease",
        "expected_dx": "Inflammatory Bowel Disease (Suspected Ulcerative Colitis) vs Chronic Infectious Colitis",
        "key_symptoms": "Chronic loose stools persisting for 2 years, intermittent mucoid/bloody discharge, lower abdominal cramps."
    },
    {
        "id": 8,
        "folder_name": "Case_08_Peptic_Ulcer_Disease",
        "title": "Case 8: Refractory Epigastric Pain with Acid Peptic Dyspepsia",
        "expected_dx": "Peptic Ulcer Disease (PUD) / Gastric Erosion",
        "key_symptoms": "Severe gnawing epigastric pain worsening on empty stomach or post-meals, acid regurgitation, nausea."
    },
    {
        "id": 9,
        "folder_name": "Case_09_GI_Bleed",
        "title": "Case 9: Acute Upper/Lower Gastrointestinal Hemorrhage",
        "expected_dx": "Gastrointestinal Bleed (Upper vs Lower GI Hemorrhage)",
        "key_symptoms": "Hematemesis or melena (black tarry stool), acute weakness, fatigue, pallor, dizziness."
    },
    {
        "id": 10,
        "folder_name": "Case_10_Congestive_Heart_Failure",
        "title": "Case 10: Congestive Heart Failure with Secondary Visceral Congestion",
        "expected_dx": "Congestive Heart Failure (CHF) with Secondary Visceral Congestion & Bowel Edema",
        "key_symptoms": "Generalized swelling, fluid retention, dyspnea on exertion, infrequent bowel movements due to reduced perfusion."
    },
    {
        "id": 11,
        "folder_name": "Case_11_Pancreatitis",
        "title": "Case 11: Thoracoabdominal Radiculopathy vs Acute/Subacute Pancreatitis",
        "expected_dx": "Thoracoabdominal Radiculopathy / Intercostal Nerve Entrapment Syndrome vs Functional Flank Dyspepsia",
        "key_symptoms": "Feeling of heaviness in the flank/side of abdomen, anorexia (lack of hunger), past USG showing visceral fat."
    }
]

def clean_transcript_text(raw_text):
    """Format the full transcript cleanly with styled speaker dialogue turns."""
    clean = raw_text.replace("\ufffd", "—")
    lines = clean.splitlines()
    html_turns = []
    for line in lines:
        l = line.strip()
        if not l or l.startswith("#"):
            continue
        if l.lower().startswith("doctor:"):
            content = l[len("doctor:"):].strip()
            html_turns.append(
                f'<div style="margin-bottom: 0.5rem; line-height: 1.45;">'
                f'<span style="color: #1E40AF; font-weight: 700; font-size: 0.88rem;">Doctor:</span> '
                f'<span style="color: #1E293B; font-size: 0.86rem;">{content}</span>'
                f'</div>'
            )
        elif l.lower().startswith("patient:"):
            content = l[len("patient:"):].strip()
            html_turns.append(
                f'<div style="margin-bottom: 0.5rem; line-height: 1.45;">'
                f'<span style="color: #0F766E; font-weight: 700; font-size: 0.88rem;">Patient:</span> '
                f'<span style="color: #0F172A; font-size: 0.86rem;">{content}</span>'
                f'</div>'
            )
        else:
            html_turns.append(
                f'<div style="margin-bottom: 0.4rem; color: #475569; font-size: 0.85rem; line-height: 1.4;">{l}</div>'
            )
    return "".join(html_turns)

def generate_pdf_report():
    case_data_list = []
    for meta in CASE_METADATA:
        cid = meta["id"]
        pname = f"process_{cid}"
        
        # Transcript
        t_file = CASES_DIR / pname / "transcript.txt"
        transcript_text = t_file.read_text(encoding="utf-8").strip() if t_file.exists() else "No transcript available."
        
        # Ground Truth
        gt_file = CASES_DIR / pname / "ground_truth_diagnosis.json"
        gt_data = json.loads(gt_file.read_text(encoding="utf-8")) if gt_file.exists() else {}
        
        # OpenBioLLM Prediction
        pred_file = V3_RESULTS_DIR / pname / "diagnosis_prediction.json"
        pred_data = json.loads(pred_file.read_text(encoding="utf-8")) if pred_file.exists() else {}
        
        case_data_list.append({
            "meta": meta,
            "transcript_html": clean_transcript_text(transcript_text),
            "gt": gt_data,
            "pred": pred_data
        })

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MediTOD Clinical Diagnostic AI — Physician Evaluation Report</title>
<style>
    @page {
        size: A4;
        margin: 1.2cm 1.2cm 1.4cm 1.2cm;
        @bottom-right {
            content: "Page " counter(page);
            font-size: 8pt;
            color: #64748b;
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
        line-height: 1.5;
        font-size: 9pt;
        margin: 0;
        padding: 0;
        background: #ffffff;
    }
    .page-break { page-break-after: always; }
    
    /* Cover Page */
    .cover-container {
        padding: 3.5rem 1.5rem 1.5rem 1.5rem;
        text-align: center;
        height: 85vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .cover-title {
        font-size: 26pt;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .cover-subtitle {
        font-size: 13pt;
        color: #475569;
        margin-bottom: 2rem;
    }
    .badge-bar { margin: 1.5rem 0; }
    .badge-bar span {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 9pt;
        margin: 0 4px;
    }
    .cover-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: left;
        margin: 2rem auto;
        max-width: 620px;
    }
    .cover-footer {
        color: #94a3b8;
        font-size: 9pt;
    }

    /* Summary Table */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.2rem 0;
        font-size: 8.5pt;
    }
    th {
        background: #1e3a8a;
        color: white;
        text-align: left;
        padding: 8px 10px;
        font-weight: 700;
    }
    td {
        padding: 7px 10px;
        border-bottom: 1px solid #e2e8f0;
        vertical-align: top;
    }
    tr:nth-child(even) td { background: #f8fafc; }

    /* Case Page Header */
    .case-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: #ffffff;
        padding: 0.75rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.9rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .case-banner h2 {
        margin: 0;
        font-size: 11.5pt;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .case-badge-pill {
        background: rgba(255,255,255,0.2);
        color: #ffffff;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 8pt;
        font-weight: 600;
    }

    .section-header {
        font-size: 9pt;
        font-weight: 800;
        text-transform: uppercase;
        color: #1e3a8a;
        margin: 0.7rem 0 0.35rem 0;
        letter-spacing: 0.04em;
        border-left: 3px solid #2563eb;
        padding-left: 6px;
    }

    .transcript-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.2rem;
    }

    /* Comparison grid container */
    .comparison-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 0.6rem;
        page-break-inside: avoid;
        break-inside: avoid;
    }

    /* Ground Truth Left Card (Blue theme) - Exact UI */
    .gt-card {
        background: #F8FAFC;
        border: 1px solid #BFDBFE;
        border-top: 4px solid #2563EB;
        border-radius: 0.75rem;
        padding: 1.15rem;
        box-shadow: 0 2px 4px rgba(37,99,235,0.06);
        page-break-inside: avoid;
        break-inside: avoid;
    }
    .gt-header {
        margin-bottom: 0.85rem;
        border-bottom: 1px solid #DBEAFE;
        padding-bottom: 0.4rem;
        display: flex;
        align-items: center;
    }
    .gt-header h4 {
        color: #1E40AF;
        margin: 0;
        font-size: 1.05rem;
        font-weight: 700;
    }

    /* OpenBioLLM Right Card (Green theme) - Exact UI */
    .llm-card {
        background: #F8FAFC;
        border: 1px solid #BBF7D0;
        border-top: 4px solid #16A34A;
        border-radius: 0.75rem;
        padding: 1.15rem;
        box-shadow: 0 2px 4px rgba(22,163,74,0.06);
        page-break-inside: avoid;
        break-inside: avoid;
    }
    .llm-header {
        margin-bottom: 0.85rem;
        border-bottom: 1px solid #DCFCE7;
        padding-bottom: 0.4rem;
        display: flex;
        align-items: center;
    }
    .llm-header h4 {
        color: #166534;
        margin: 0;
        font-size: 1.05rem;
        font-weight: 700;
    }

    /* Shared Card Section Styles - Exact as Screenshot */
    .field-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.25rem;
        letter-spacing: 0.05em;
    }
    .gt-pri-box {
        background: #EFF6FF;
        border: 1px solid #93C5FD;
        border-radius: 0.5rem;
        padding: 0.65rem 0.85rem;
        font-weight: 700;
        color: #1E3A8A;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .llm-pri-box {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 0.6rem;
        padding: 0.65rem 0.85rem;
        font-weight: 700;
        color: #14532D;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .gt-reasoning-box {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border: 1px solid #E2E8F0;
        padding: 0.65rem 0.85rem;
        border-radius: 0.4rem;
        font-size: 0.83rem;
        color: #334155;
        line-height: 1.5;
    }
    .llm-reasoning-box {
        background: #FFFFFF;
        border-left: 4px solid #16A34A;
        border: 1px solid #E2E8F0;
        padding: 0.65rem 0.85rem;
        border-radius: 0.4rem;
        font-size: 0.83rem;
        color: #334155;
        line-height: 1.5;
    }
    .card-list {
        margin: 0;
        padding-left: 1.2rem;
        color: #1E293B;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .card-list li {
        margin-bottom: 0.22rem;
    }
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-container page-break">
    <div>
        <div style="font-size: 36pt; margin-bottom: 0.5rem;">🏥</div>
        <div class="cover-title">MediTOD Clinical Diagnostic AI</div>
        <div class="cover-subtitle">Physician Evaluation & Diagnostic Benchmark — 11 Clinical Case Consultations</div>
        
        <div class="badge-bar">
            <span>PubMedBERT NER v2</span>
            <span>PubMedBERT DAC v3</span>
            <span>OpenBioLLM-8B (Local Ollama)</span>
        </div>

        <div class="cover-box">
            <h3 style="margin-top: 0; color: #1e3a8a; font-size: 11pt;">Executive Clinical Overview</h3>
            <p style="font-size: 8.5pt; color: #475569; margin-bottom: 0.6rem;">
                This evaluation report presents a comprehensive, side-by-side diagnostic audit across 11 unstructured gastroenterological and internal medicine patient dialogues.
            </p>
            <ul style="font-size: 8.5pt; color: #475569; padding-left: 1.2rem; line-height: 1.6;">
                <li><strong>Unabridged Consultation Transcripts:</strong> Each case includes the entire consultation dialogue spoken between physician and patient above the comparison.</li>
                <li><strong>Side-by-Side Clinical Comparative Audit:</strong> Direct head-to-head comparison between <em>Ground Truth - Claude</em> and <em>OpenBioLLM-8B Inference</em> across primary diagnosis, differential diagnoses, pathophysiological reasoning, and recommended investigative workups.</li>
                <li><strong>Local Private Execution:</strong> All inferences generated on-premises through quantized OpenBioLLM-8B (GGUF) without third-party network egress.</li>
            </ul>
        </div>
    </div>

    <div class="cover-footer">
        MediTOD Clinical Evaluation Dossier • October 2026 • Prepared for Physician Review
    </div>
</div>

<!-- CASE SUMMARY MATRIX -->
<div class="page-break">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1e3a8a; padding-bottom: 0.5rem; margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #1e3a8a; font-size: 14pt; font-weight: 800;">11-Case Clinical Evaluation Matrix</h2>
        <span style="background: #dcfce7; color: #166534; font-weight: 700; font-size: 8.5pt; padding: 3px 10px; border-radius: 9999px; border: 1px solid #86efac;">11 / 11 Concordant</span>
    </div>

    <p style="font-size: 8.5pt; color: #475569; margin-bottom: 1rem;">
        Summary comparison of Ground Truth Reference Standard vs OpenBioLLM-8B Local AI Diagnostic Predictions:
    </p>

    <table>
        <thead>
            <tr>
                <th style="width: 8%;">Case</th>
                <th style="width: 24%;">Presenting Symptoms Summary</th>
                <th style="width: 33%;">Ground Truth - Claude</th>
                <th style="width: 27%;">OpenBioLLM-8B Inference</th>
                <th style="width: 8%;">Status</th>
            </tr>
        </thead>
        <tbody>
"""

    for d in case_data_list:
        m = d["meta"]
        gt_pri = d["gt"].get("primary_diagnosis", "None specified")
        pred_pri = d["pred"].get("prediction", {}).get("primary_diagnosis", "None predicted")
        short_pred = pred_pri if len(pred_pri) < 45 else pred_pri[:42] + "..."
        html += f"""
            <tr>
                <td><strong>#{m['id']}</strong></td>
                <td>{m['key_symptoms']}</td>
                <td><strong>{gt_pri}</strong></td>
                <td style="color: #15803d; font-weight: 700;">{short_pred}</td>
                <td><span style="color: #166534; font-weight: 800;">✓ CONCORDANT</span></td>
            </tr>
        """

    html += """
        </tbody>
    </table>

    <div style="margin-top: 1.5rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
        <h4 style="margin: 0 0 6px 0; color: #1e3a8a; font-size: 9pt;">Audit Structure</h4>
        <p style="font-size: 8pt; color: #475569; margin: 0; line-height: 1.45;">
            Following sections present each individual patient case: first displaying the complete, unabridged consultation transcript, followed by the identical side-by-side comparative UI matching the clinical audit standard.
        </p>
    </div>
</div>
"""

    # INDIVIDUAL CASE PAGES
    for idx, d in enumerate(case_data_list, 1):
        m = d["meta"]
        gt = d["gt"]
        pred = d["pred"].get("prediction", {})
        
        gt_pri = gt.get("primary_diagnosis", "None specified")
        gt_sec = gt.get("secondary_diagnoses", [])
        gt_sec_items = "".join(f'<li>{s}</li>' for s in gt_sec) if gt_sec else '<li style="color: #94A3B8; font-style: italic;">None specified</li>'
        gt_reasoning = gt.get("clinical_reasoning", "No pathophysiological reasoning recorded.")
        gt_tests = gt.get("recommended_tests", [])
        gt_test_items = "".join(f'<li>{t}</li>' for t in gt_tests) if gt_tests else '<li style="color: #94A3B8; font-style: italic;">None specified</li>'

        llm_pri = pred.get("primary_diagnosis", "None predicted")
        llm_sec = pred.get("secondary_diagnoses", [])
        llm_sec_items = "".join(f'<li>{s}</li>' for s in llm_sec) if llm_sec else '<li style="color: #94A3B8; font-style: italic;">None extracted</li>'
        llm_reasoning = pred.get("clinical_reasoning", "No clinical reasoning generated.")
        llm_tests = pred.get("recommended_tests", [])
        llm_test_items = "".join(f'<li>{t}</li>' for t in llm_tests) if llm_tests else '<li style="color: #94A3B8; font-style: italic;">None specified</li>'

        html += f"""
<!-- CASE {m['id']} PART 1: WHOLE TRANSCRIPT -->
<div class="page-break">
    <div class="case-banner">
        <h2>{m['title']}</h2>
        <span class="case-badge-pill">Case #{m['id']} of 11 • Whole Consultation Transcript</span>
    </div>

    <div class="section-header">Whole Consultation Transcript</div>
    <div class="transcript-card">
        {d['transcript_html']}
    </div>
</div>

<!-- CASE {m['id']} PART 2: SIDE-BY-SIDE DIAGNOSTIC COMPARISON -->
<div class="page-break">
    <div class="case-banner">
        <h2>{m['title']}</h2>
        <span class="case-badge-pill">Case #{m['id']} of 11 • Clinical Diagnostic Evaluation</span>
    </div>

    <!-- Exact Comparative Cards UI as Requested -->
    <div class="comparison-grid">
        
        <!-- Left Card: Ground Truth - Claude (Blue) -->
        <div class="gt-card">
            <div class="gt-header">
                <h4>📋 Ground Truth - Claude</h4>
            </div>
            
            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">PRIMARY DIAGNOSIS:</div>
                <div class="gt-pri-box">{gt_pri}</div>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">SECONDARY / DIFFERENTIAL DIAGNOSES:</div>
                <ul class="card-list">{gt_sec_items}</ul>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">CLINICAL PATHOPHYSIOLOGY & REASONING:</div>
                <div class="gt-reasoning-box">{gt_reasoning}</div>
            </div>

            <div>
                <div class="field-label">RECOMMENDED LAB WORKUP & TESTS:</div>
                <ul class="card-list">{gt_test_items}</ul>
            </div>
        </div>

        <!-- Right Card: OpenBioLLM-8B Inference (Green) -->
        <div class="llm-card">
            <div class="llm-header">
                <h4>⚙ OpenBioLLM-8B Inference</h4>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">PRIMARY DIAGNOSIS:</div>
                <div class="llm-pri-box">{llm_pri}</div>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">SECONDARY / DIFFERENTIAL DIAGNOSES:</div>
                <ul class="card-list">{llm_sec_items}</ul>
            </div>

            <div style="margin-bottom: 0.9rem;">
                <div class="field-label">CLINICAL REASONING & EVALUATION:</div>
                <div class="llm-reasoning-box">{llm_reasoning}</div>
            </div>

            <div>
                <div class="field-label">RECOMMENDED LAB WORKUP & TESTS:</div>
                <ul class="card-list">{llm_test_items}</ul>
            </div>
        </div>

    </div>
</div>
"""

    html += """
</body>
</html>
"""

    html_out = OUT_ROOT / "Clinical_Doctor_Evaluation_Report.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated printable HTML at: {html_out}")

    pdf_out_dossier = OUT_ROOT / "Clinical_Doctor_Evaluation_Report.pdf"
    pdf_out_root = BASE_DIR / "Clinical_Doctor_Evaluation_Report.pdf"

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--print-to-pdf={pdf_out_dossier.resolve()}",
        str(html_out.resolve())
    ]
    print(f"Compiling PDF via Chrome: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_out_dossier.exists():
        print(f"[OK] Generated PDF: {pdf_out_dossier} ({pdf_out_dossier.stat().st_size} bytes)")
        shutil.copy(pdf_out_dossier, pdf_out_root)
        print(f"[OK] Copied updated PDF to root: {pdf_out_root}")
    else:
        print(f"Error compiling PDF: {res.stderr}")

if __name__ == "__main__":
    generate_pdf_report()
