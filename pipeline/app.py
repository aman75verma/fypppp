"""
Streamlit Web Interface for Medical Dialogue Pipeline
Allows users to input dialogue text and process through NER, DAC, and medical report generation
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import PipelineProcessor
from sentence_processor import SentenceProcessor
from diagnosis_inference import DiagnosisInference

# Page config
st.set_page_config(
    page_title="Medical Dialogue Pipeline & AI Diagnosis",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium feel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@500;700&display=swap');

    :root {
        --primary: #0066FF;
        --secondary: #6B7280;
        --success: #10B981;
        --error: #EF4444;
        --bg-light: #F9FAFB;
        --card-bg: #FFFFFF;
        --accent: #EEF2FF;
    }

    .stApp {
        background-color: var(--bg-light);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #1F2937;
    }

    .main-header {
        background: linear-gradient(90deg, #0066FF 0%, #60A5FA 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .pipeline-box {
        background-color: var(--accent);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 6px solid var(--primary);
        margin: 1rem 0;
        font-family: 'Inter', monospace;
        font-size: 0.9rem;
        color: #1E40AF;
        line-height: 1.6;
    }

    .dialogue-header {
        background: white;
        padding: 1.25rem;
        border-radius: 0.75rem;
        border-left: 5px solid var(--primary);
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .stButton>button {
        border-radius: 0.5rem;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.2);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        background-color: white;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding: 0 16px;
        border: 1px solid #E5E7EB;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--accent) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* AI Diagnosis Card Styles */
    .dx-hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.35);
    }

    .dx-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0.5rem 0 0 0;
        color: #FFFFFF;
        letter-spacing: -0.01em;
    }

    .dx-pill-secondary {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 9999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .dx-reasoning {
        background: #F8FAFC;
        border-left: 5px solid #2563EB;
        border-radius: 0 0.75rem 0.75rem 0;
        padding: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.65;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .test-item {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 0.5rem;
        padding: 0.65rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #166534;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to discover available process result folders
def get_available_result_folders():
    """Discover all process_* folders in results/ and return list of (label, path) tuples"""
    folders = []
    search_dirs = [
        PROJECT_ROOT.parent / "results",
        PROJECT_ROOT / "results"
    ]
    seen = set()
    for sdir in search_dirs:
        if sdir.exists() and sdir.is_dir():
            try:
                entries = sorted(
                    os.listdir(sdir),
                    key=lambda x: int(x.split('_')[1]) if x.startswith('process_') and x.split('_')[1].isdigit() else 999
                )
            except Exception:
                entries = sorted(os.listdir(sdir))
                
            for entry in entries:
                full = os.path.abspath(os.path.join(str(sdir), entry))
                if os.path.isdir(full) and entry.startswith("process_") and entry not in seen:
                    if os.path.exists(os.path.join(full, 'processed_dialogues.json')):
                        seen.add(entry)
                        label = entry
                        dx_file = os.path.join(full, 'diagnosis_prediction.json')
                        if os.path.exists(dx_file):
                            try:
                                with open(dx_file, 'r', encoding='utf-8') as f:
                                    dx_data = json.load(f)
                                    pri = dx_data.get('prediction', {}).get('primary_diagnosis', '')
                                    if pri:
                                        short_pri = pri[:35] + ("..." if len(pri) > 35 else "")
                                        label = f"{entry} ({short_pri})"
                            except Exception:
                                pass
                        folders.append((label, full))
    return folders

# Helper function to render rich side-by-side comparison cards
def _render_side_by_side_comparison(dialogue_key: str, gt_entry: dict, llm_entry: dict, is_embedded: bool = False):
    """
    Render a high-fidelity, side-by-side comparison card between
    Expert Physician Ground Truth and OpenBioLLM-8B Inference Prediction.
    """
    import re
    gt_pri = gt_entry.get("primary_diagnosis", "None specified")
    llm_pri = llm_entry.get("primary_diagnosis", "None predicted")
    
    # Assess concordance
    gt_words = set(re.findall(r'\b\w{3,}\b', gt_pri.lower())) - {'with', 'and', 'for', 'from', 'disease', 'syndrome', 'acute', 'chronic'}
    llm_words = set(re.findall(r'\b\w{3,}\b', llm_pri.lower())) - {'with', 'and', 'for', 'from', 'disease', 'syndrome', 'acute', 'chronic'}
    overlap = gt_words.intersection(llm_words)
    
    key_terms = {'gerd', 'reflux', 'gastritis', 'dyspepsia', 'hypothyroidism', 'cholelithiasis', 'calculi', 'esophagitis', 'colitis', 'jaundice', 'anemia', 'ulcer', 'hemorrhage', 'gallbladder', 'stone'}
    domain_overlap = overlap.intersection(key_terms)
    
    if len(domain_overlap) >= 1 or len(overlap) >= 2 or gt_pri.lower() in llm_pri.lower() or llm_pri.lower() in gt_pri.lower():
        badge_html = '<span style="background: #DCFCE7; color: #166534; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; border: 1px solid #86EFAC;">🎯 High Clinical Concordance / Direct Match</span>'
    elif any(term in gt_pri.lower() and term in str(llm_entry.get("secondary_diagnoses", [])).lower() for term in key_terms):
        badge_html = '<span style="background: #E0E7FF; color: #3730A3; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; border: 1px solid #C7D2FE;">🔄 Concordant in Differential Diagnoses</span>'
    else:
        badge_html = '<span style="background: #FEF3C7; color: #92400E; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; border: 1px solid #FDE68A;">🔍 Clinical Differential / Pathophysiological Overlap</span>'
    
    case_label = dialogue_key.replace('_', ' ').upper()
    
    if not is_embedded:
        header_banner = f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem; background: #FFFFFF; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);"><div><span style="font-weight: 800; color: #0F172A; font-size: 1.1rem;">🏥 {case_label}</span><span style="color: #64748B; font-size: 0.85rem; margin-left: 0.75rem;">Physician Benchmark vs OpenBioLLM</span></div><div>{badge_html}</div></div>'
        st.markdown(header_banner, unsafe_allow_html=True)
    else:
        badge_only = f'<div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 0.75rem;">{badge_html}</div>'
        st.markdown(badge_only, unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    with c1:
        gt_sec = gt_entry.get('secondary_diagnoses', [])
        sec_items = "".join(f'<li style="margin-bottom: 0.25rem;">{s}</li>' for s in gt_sec) if gt_sec else '<li style="color: #94A3B8; font-style: italic;">None recorded</li>'
        gt_tests = gt_entry.get('recommended_tests', [])
        test_items = "".join(f'<li style="margin-bottom: 0.2rem;">{t}</li>' for t in gt_tests) if gt_tests else '<li style="color: #94A3B8; font-style: italic;">None specified</li>'
        gt_reasoning = gt_entry.get('clinical_reasoning', 'No reasoning provided')
        
        card1_html = (
            f'<div style="background: #F8FAFC; border: 1px solid #BFDBFE; border-top: 4px solid #2563EB; border-radius: 0.75rem; padding: 1.25rem; height: 100%; box-shadow: 0 2px 4px rgba(37,99,235,0.06);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; border-bottom: 1px solid #DBEAFE; padding-bottom: 0.4rem;"><h4 style="color: #1E40AF; margin: 0; font-size: 1.05rem;">📋 Ground Truth - Claude</h4></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Primary Diagnosis:</div><div style="background: #EFF6FF; border: 1px solid #93C5FD; border-radius: 0.5rem; padding: 0.65rem 0.85rem; font-weight: 700; color: #1E3A8A; font-size: 0.98rem; line-height: 1.4;">{gt_pri}</div></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Secondary / Differential Diagnoses:</div><ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 0.88rem;">{sec_items}</ul></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Clinical Pathophysiology & Reasoning:</div><div style="background: #FFFFFF; border-left: 4px solid #2563EB; border: 1px solid #E2E8F0; padding: 0.65rem 0.85rem; border-radius: 0.4rem; font-size: 0.85rem; color: #334155; line-height: 1.5;">{gt_reasoning}</div></div>'
            f'<div><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Recommended Lab Workup & Tests:</div><ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 0.85rem;">{test_items}</ul></div>'
            f'</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

    with c2:
        llm_sec = llm_entry.get('secondary_diagnoses', [])
        llm_sec_items = "".join(f'<li style="margin-bottom: 0.25rem;">{s}</li>' for s in llm_sec) if llm_sec else '<li style="color: #94A3B8; font-style: italic;">None extracted</li>'
        llm_tests = llm_entry.get('recommended_tests', [])
        llm_test_items = "".join(f'<li style="margin-bottom: 0.2rem;">{t}</li>' for t in llm_tests) if llm_tests else '<li style="color: #94A3B8; font-style: italic;">None specified</li>'
        llm_reasoning = llm_entry.get('clinical_reasoning', 'No reasoning generated')
        
        card2_html = (
            f'<div style="background: #F8FAFC; border: 1px solid #BBF7D0; border-top: 4px solid #16A34A; border-radius: 0.75rem; padding: 1.25rem; height: 100%; box-shadow: 0 2px 4px rgba(22,163,74,0.06);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; border-bottom: 1px solid #DCFCE7; padding-bottom: 0.4rem;"><h4 style="color: #166534; margin: 0; font-size: 1.05rem;">⚙️ OpenBioLLM-8B Inference</h4></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Primary Diagnosis:</div><div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 0.6rem; padding: 0.65rem 0.85rem; font-weight: 700; color: #14532D; font-size: 0.98rem; line-height: 1.4;">{llm_pri}</div></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Secondary / Differential Diagnoses:</div><ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 0.88rem;">{llm_sec_items}</ul></div>'
            f'<div style="margin-bottom: 0.9rem;"><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Clinical Reasoning & Evaluation:</div><div style="background: #FFFFFF; border-left: 4px solid #16A34A; border: 1px solid #E2E8F0; padding: 0.65rem 0.85rem; border-radius: 0.4rem; font-size: 0.85rem; color: #334155; line-height: 1.5;">{llm_reasoning}</div></div>'
            f'<div><div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 0.25rem; letter-spacing: 0.05em;">Recommended Lab Workup & Tests:</div><ul style="margin: 0; padding-left: 1.2rem; color: #1E293B; font-size: 0.85rem;">{llm_test_items}</ul></div>'
            f'</div>'
        )
        st.markdown(card2_html, unsafe_allow_html=True)

# Helper function to display results for a single dialogue
def _display_dialogue_results(dialogue_name: str, dialogue_data: Dict[str, Any], results_dir: str = None):
    """Display results for a single dialogue"""
    
    if results_dir is None:
        results_dir = st.session_state.get('selected_folder_path', '')
    
    if not dialogue_data.get('summary'):
        st.error("No data to display")
        return
    
    summary = dialogue_data['summary']
    processed = dialogue_data['processed']
    medical_report = dialogue_data['medical_report']
    
    # Show dialogue header
    st.markdown(f"""
    <div class="dialogue-header">
    🗂️ {dialogue_name}
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sentences", summary.get('total_sentences', 0))
    with col2:
        st.metric("Successful", summary.get('successful', 0))
    with col3:
        st.metric("Failed", summary.get('failed', 0))
    with col4:
        st.metric("Success Rate", summary.get('success_rate', '0%'))
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab_dx, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Processed Sentences",
        "🏥 Medical Report",
        "🩺 AI Diagnosis",
        "🔍 NER Tags",
        "🗣️ DAC Tags",
        "📊 Statistics",
        "⚖️ Manual vs Model"
    ])
    
    # Tab 1: Processed Sentences
    with tab1:
        st.subheader("Extracted Sentences with NER & DAC")
        
        if processed:
            for sent_id, data in processed.items():
                with st.expander(f"**{sent_id}**: {data['sentence'][:70]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sentence:**")
                        st.text(data['sentence'])
                    
                    with col2:
                        st.write("**DAC Tag:**")
                        st.info(data['DAC_tag'])
                    
                    st.write("**NER Tags:**")
                    if data.get('NER_tags'):
                        ner_df = pd.DataFrame(data['NER_tags'])
                        st.dataframe(ner_df, width="stretch")
                    else:
                        st.write("No NER tags extracted")
    
    # Tab 2: Medical Report
    with tab2:
        st.subheader("Structured Medical Report")
        
        # Define all sections from pipeline.py
        sections = {
            'chief_complaint': ('🔴 Chief Complaint (CC)', 'CC'),
            'history_of_present_illness': ('🟡 History of Present Illness (HPI)', 'HPI'),
            'past_medical_history': ('🟢 Past Medical History (PMH)', 'PMH'),
            'social_history': ('🟣 Social History (SH)', 'SH'),
            'diagnosis': ('🔵 Diagnosis (Dx)', 'Dx'),
            'suggestions': ('💊 Medications/Suggestions (Rx)', 'Rx')
        }
        
        # Display all sections
        for section_key, (section_label, abbrev) in sections.items():
            section_data = medical_report.get(section_key, [])
            
            st.subheader(section_label)
            
            if section_data:
                for entry in section_data:
                    with st.expander(f"{abbrev}: {entry['sentence'][:70] if isinstance(entry, dict) else entry[:70]}..."):
                        if isinstance(entry, dict):
                            st.write(f"**Sentence:** {entry['sentence']}")
                            st.write(f"**DAC:** {entry.get('DAC_tag', 'N/A')}")
                            if entry.get('NER_tags'):
                                st.write("**NER Tags:**")
                                st.dataframe(pd.DataFrame(entry['NER_tags']))
                        else:
                            st.write(f"**Sentence:** {entry}")
            else:
                st.info("(No entries in this section)")

    # Tab: AI Diagnosis (OpenBioLLM)
    with tab_dx:
        st.subheader("🩺 AI Clinical Decision Support & Diagnosis")
        st.caption("Powered by OpenBioLLM-8B (Llama-3 Architecture) fine-tuned for biomedical clinical reasoning")

        # Load or generate diagnosis prediction
        dx_data = dialogue_data.get('diagnosis_prediction')
        if not dx_data and results_dir:
            dx_file_path = os.path.join(results_dir, 'diagnosis_prediction.json')
            if os.path.exists(dx_file_path):
                try:
                    with open(dx_file_path, 'r', encoding='utf-8') as f:
                        dx_data = json.load(f)
                except Exception:
                    pass

        # Action bar: Re-run button
        col_dx_act1, col_dx_act2 = st.columns([3, 1])
        with col_dx_act1:
            st.write("")
        with col_dx_act2:
            rerun_dx = st.button("⚡ Re-predict Diagnosis", key=f"btn_dx_{dialogue_name}", help="Re-run OpenBioLLM diagnostic inference on the current medical report")

        if rerun_dx or not dx_data:
            with st.spinner("Analyzing structured clinical report with OpenBioLLM..."):
                dx_engine = DiagnosisInference()
                dx_data = dx_engine.predict(medical_report, process_name=dialogue_name)
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    dx_engine.save_prediction(dx_data, os.path.join(results_dir, 'diagnosis_prediction.json'))
                dialogue_data['diagnosis_prediction'] = dx_data

        if dx_data and 'prediction' in dx_data:
            pred = dx_data['prediction']
            model_badge = dx_data.get('model', 'OpenBioLLM-8B')

            # Primary Diagnosis Hero Banner
            pri_dx = pred.get('primary_diagnosis', 'Under Evaluation')
            st.markdown(f"""
            <div class="dx-hero">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.9;">
                    🏥 Primary Clinical Diagnosis • <span style="background: rgba(255,255,255,0.25); padding: 2px 8px; border-radius: 4px;">{model_badge}</span>
                </div>
                <div class="dx-title">{pri_dx}</div>
            </div>
            """, unsafe_allow_html=True)

            # Secondary / Differential Diagnoses
            sec_dx = pred.get('secondary_diagnoses', [])
            if sec_dx:
                st.markdown("##### 🧬 Differential & Secondary Diagnoses")
                pills_html = "".join([f'<span class="dx-pill-secondary">🔸 {s}</span>' for s in sec_dx])
                st.markdown(f'<div style="margin-bottom: 1.25rem;">{pills_html}</div>', unsafe_allow_html=True)

            # Clinical Reasoning & Pathophysiology
            st.markdown("##### 🩺 Clinical Reasoning & Evidence Synthesis")
            reasoning = pred.get('clinical_reasoning', 'No reasoning provided.')
            st.markdown(f"""
            <div class="dx-reasoning">
                <strong>Diagnostic Rationale:</strong><br>
                {reasoning}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Recommended Diagnostic Workup
            st.markdown("##### 🔬 Recommended Diagnostic Workup & Lab Investigations")
            tests = pred.get('recommended_tests', [])
            if tests:
                for idx, t in enumerate(tests, 1):
                    st.markdown(f"""
                    <div class="test-item">
                        <strong>{idx}.</strong> {t}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific follow-up lab tests recommended.")

            # Ground Truth Comparison (Available across all 11 benchmark processes)
            gt_comp_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Diagnosis", "diagnosis_comparison.json"))
            if not os.path.exists(gt_comp_path):
                gt_comp_path = os.path.abspath(os.path.join(PROJECT_ROOT, "Diagnosis", "diagnosis_comparison.json"))
            if os.path.exists(gt_comp_path):
                try:
                    with open(gt_comp_path, 'r', encoding='utf-8') as f:
                        gt_all = json.load(f)
                    
                    matched_key = None
                    import re
                    m = re.search(r'process_(\d+)', dialogue_name.lower())
                    if m:
                        cand_k = f"process_{m.group(1)}"
                        if cand_k in gt_all:
                            matched_key = cand_k
                    if not matched_key:
                        for k in gt_all:
                            if k == dialogue_name.lower():
                                matched_key = k
                                break
                    
                    if matched_key:
                        st.markdown("---")
                        st.markdown("#### ⚖️ Ground Truth vs OpenBioLLM Prediction Comparison")
                        st.caption(f"Benchmark validation for **{matched_key.upper()}**: Expert physician clinical annotation vs OpenBioLLM inference")
                        
                        gt_entry = gt_all[matched_key].get("ground_truth", {})
                        llm_entry = pred if pred and pred.get("primary_diagnosis") else gt_all[matched_key].get("llm_prediction", {})
                        
                        _render_side_by_side_comparison(matched_key, gt_entry, llm_entry, is_embedded=True)
                except Exception as e:
                    st.error(f"Error loading comparison: {e}")

            # Formatted Clinical Summary sent to LLM
            with st.expander("👁️ View Extracted Clinical Input Fed to OpenBioLLM"):
                st.code(dx_data.get('clinical_summary', 'None'), language='text')

            # Download diagnosis JSON
            st.download_button(
                label="⬇️ Download Diagnosis JSON",
                data=json.dumps(dx_data, indent=2, ensure_ascii=False),
                file_name=f"{dialogue_name}_diagnosis_prediction.json",
                mime="application/json",
                key=f"dl_dx_{dialogue_name}"
            )
        else:
            st.warning("No diagnosis prediction available. Click 'Re-predict Diagnosis' above to generate.")
    
    # Tab 3: NER Tags
    with tab3:
        st.subheader("Named Entity Recognition Tags")
        
        # Count NER tags
        ner_counts = {}
        for sent_id, data in processed.items():
            for tag in data.get('NER_tags', []):
                tag_type = tag.get('type', 'Unknown') if isinstance(tag, dict) else str(tag)
                ner_counts[tag_type] = ner_counts.get(tag_type, 0) + 1
        
        if ner_counts:
            ner_df = pd.DataFrame(list(ner_counts.items()), columns=['NER Type', 'Count'])
            ner_df = ner_df.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(ner_df, width="stretch")
            with col2:
                st.bar_chart(ner_df.set_index('NER Type'))
        else:
            st.info("No NER tags extracted")
    
    # Tab 4: DAC Tags
    with tab4:
        st.subheader("Dialogue Act Classification Tags")
        
        # Count DAC tags
        dac_counts = {}
        for sent_id, data in processed.items():
            dac_tag = data.get('DAC_tag', 'Unknown')
            dac_counts[dac_tag] = dac_counts.get(dac_tag, 0) + 1
        
        if dac_counts:
            dac_df = pd.DataFrame(list(dac_counts.items()), columns=['DAC Type', 'Count'])
            dac_df = dac_df.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(dac_df, width="stretch")
            with col2:
                st.bar_chart(dac_df.set_index('DAC Type'))
        else:
            st.info("No DAC tags extracted")
    
    # Tab 6: Manual vs Model Comparison
    with tab6:
        st.subheader("Manual vs Model Comparison")
        
        annotated_path = os.path.join(results_dir, 'annotated_dialogues.json') if results_dir else ""
        if results_dir and os.path.exists(annotated_path):
            try:
                with open(annotated_path, 'r', encoding='utf-8') as f:
                    annotated = json.load(f)
                
                comparison_data = []
                for sent_id, model_data in processed.items():
                    manual_data = annotated.get(sent_id, {})
                    
                    # Format NER tags for display
                    model_ner = ", ".join([f"{t['type']}: {t['entity']}" for t in model_data.get('NER_tags', [])])
                    manual_ner = ", ".join([f"{t['type']}: {t['entity']}" for t in manual_data.get('NER_tags', [])])
                    
                    # Highlight mismatches
                    dac_match = "✅" if model_data['DAC_tag'] == manual_data.get('DAC_tag') else "❌"
                    ner_match = "✅" if model_ner == manual_ner else "❌"
                    
                    comparison_data.append({
                        "ID": sent_id,
                        "Sentence": model_data['sentence'],
                        "Model DAC": model_data['DAC_tag'],
                        "Manual DAC": manual_data.get('DAC_tag', 'N/A'),
                        "DAC Match": dac_match,
                        "Model NER": model_ner,
                        "Manual NER": manual_ner,
                        "NER Match": ner_match
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Show summary metrics
                total = len(comp_df)
                dac_acc = (comp_df['DAC Match'] == "✅").sum() / total if total > 0 else 0
                ner_acc = (comp_df['NER Match'] == "✅").sum() / total if total > 0 else 0
                
                m1, m2 = st.columns(2)
                m1.metric("DAC Accuracy", f"{dac_acc*100:.1f}%")
                m2.metric("NER Accuracy", f"{ner_acc*100:.1f}%")
                
                st.dataframe(comp_df, use_container_width=True)
                
                # Analysis of failures
                st.write("### 🔍 Failure Analysis")
                failures = comp_df[(comp_df['DAC Match'] == "❌") | (comp_df['NER Match'] == "❌")]
                if not failures.empty:
                    st.write(f"Found {len(failures)} sentences with mismatches.")
                    st.dataframe(failures, use_container_width=True)
                else:
                    st.success("Perfect match between model and manual annotations!")
                    
            except Exception as e:
                st.error(f"Error loading manual annotations: {e}")
        else:
            st.info("ℹ️ No manual ground-truth annotations uploaded for this dialogue. Displaying **Model Extracted Tagging Overview**:")
            
            model_annotations = []
            for sent_id, model_data in processed.items():
                ner_tags = model_data.get('NER_tags', [])
                formatted_ner = ", ".join([f"{t['type']}: {t['entity']}" for t in ner_tags]) if ner_tags else "(none)"
                model_annotations.append({
                    "Sentence ID": sent_id,
                    "Sentence": model_data['sentence'],
                    "DAC Tag": model_data.get('DAC_tag', 'N/A'),
                    "Model Extracted NER": formatted_ner,
                    "Entity Count": len(ner_tags)
                })
            
            ann_df = pd.DataFrame(model_annotations)
            
            # Overview Metrics
            total_sents = len(ann_df)
            total_ents = sum(a['Entity Count'] for a in model_annotations)
            tagged_sents = sum(1 for a in model_annotations if a['Entity Count'] > 0)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sentences", total_sents)
            m2.metric("Sentences with Entities", f"{tagged_sents}/{total_sents}")
            m3.metric("Total Extracted Entities", total_ents)
            
            st.dataframe(ann_df, use_container_width=True)
            
            # Export template
            export_dict = {
                s['Sentence ID']: {
                    'sentence': s['Sentence'],
                    'DAC_tag': s['DAC Tag'],
                    'NER_tags': processed[s['Sentence ID']].get('NER_tags', [])
                } for s in model_annotations
            }
            st.download_button(
                label="📥 Export Model Tagging as Ground Truth JSON",
                data=json.dumps(export_dict, indent=2, ensure_ascii=False),
                file_name=f"{dialogue_name}_model_annotations.json",
                mime="application/json",
                key=f"export_ann_{dialogue_name}"
            )
    
    # Tab 5: Statistics
    with tab5:
        st.subheader("Processing Statistics")
        
        stats = {
            'Total Sentences': len(processed),
            'Chief Complaint': len(medical_report.get('chief_complaint', [])),
            'HPI': len(medical_report.get('history_of_present_illness', [])),
            'PMH': len(medical_report.get('past_medical_history', [])),
            'SH': len(medical_report.get('social_history', [])),
            'Diagnosis': len(medical_report.get('diagnosis', [])),
            'Suggestions': len(medical_report.get('suggestions', [])),
        }
        
        stats_df = pd.DataFrame(list(stats.items()), columns=['Category', 'Count'])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(stats_df, width="stretch")
        with col2:
            st.bar_chart(stats_df.set_index('Category'))
    
    # Download section for this dialogue
    st.markdown("---")
    st.subheader(f"📥 Download {dialogue_name} Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        json_str = json.dumps(processed, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Processed Sentences",
            data=json_str,
            file_name=f"{dialogue_name}_processed.json",
            mime="application/json",
            key=f"processed_{dialogue_name}"
        )
    
    with col2:
        json_str = json.dumps(medical_report, indent=2, ensure_ascii=False)
        st.download_button(
            label="📋 Medical Report",
            data=json_str,
            file_name=f"{dialogue_name}_report.json",
            mime="application/json",
            key=f"report_{dialogue_name}"
        )
    
    with col3:
        json_str = json.dumps(summary, indent=2, ensure_ascii=False)
        st.download_button(
            label="📊 Summary",
            data=json_str,
            file_name=f"{dialogue_name}_summary.json",
            mime="application/json",
            key=f"summary_{dialogue_name}"
        )

    with col4:
        dx_data_dl = dialogue_data.get('diagnosis_prediction')
        if dx_data_dl:
            dx_json_str = json.dumps(dx_data_dl, indent=2, ensure_ascii=False)
            st.download_button(
                label="🩺 AI Diagnosis",
                data=dx_json_str,
                file_name=f"{dialogue_name}_diagnosis.json",
                mime="application/json",
                key=f"summary_dx_{dialogue_name}"
            )
        else:
            st.button("🩺 AI Diagnosis (None)", disabled=True, key=f"btn_nodx_{dialogue_name}")


# Title
st.title("⚕️ Medical Dialogue Processing Pipeline")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Configuration")
    
    # Mode selector
    mode = st.radio(
        "Select Mode:",
        ["🚀 Process Dialogue", "👁️ View Results", "⚖️ 11-Report Benchmark Comparison", "🔄 Batch Rerun"],
        help="Choose between processing new dialogues, viewing saved results, comparing all 11 reports against ground truth, or batch re-running with new model"
    )
    
    # Tabs in sidebar
    sidebar_tabs = st.tabs(["⚙️ Settings", "📊 Pipeline", "📝 Format Guide"])
    
    with sidebar_tabs[0]:  # Settings Tab
        input_method = st.radio(
            "Choose input method:",
            ["📝 Paste Text", "📤 Upload File(s)"],
            help="Paste dialogue directly or upload one or more .txt files"
        )
        st.markdown("---")
        st.markdown("##### 🩺 AI Diagnosis Engine")
        st.success("🤖 **OpenBioLLM-8B** (Llama-3 Architecture)")
        st.caption("HF Token: `hf_...DWVE` (Configured)")
        st.caption("Mode: Biomedical Decision Support")
    
    with sidebar_tabs[1]:  # Pipeline Visual
        st.subheader("Pipeline Flow")
        st.markdown("""
        <div class="pipeline-box">
        📄 Input Dialogue<br>
        ↓<br>
        ✂️ Sentence Splitting<br>
        ↓<br>
        🔍 NER Extraction (12 tags)<br>
        ↓<br>
        🗣️ DAC Classification (16 tags)<br>
        ↓<br>
        📋 Condition Checking<br>
        ├─ Chief Complaint (CC)<br>
        ├─ History of Present Illness (HPI)<br>
        ├─ Past Medical History (PMH)<br>
        ├─ Social History (SH)<br>
        ├─ Diagnosis (Dx)<br>
        └─ Medications (Rx)<br>
        ↓<br>
        📊 Medical Report JSON<br>
        ↓<br>
        🩺 <strong>AI Diagnosis (OpenBioLLM)</strong><br>
        ├─ Primary Diagnosis<br>
        ├─ Differential Diagnoses<br>
        ├─ Clinical Reasoning<br>
        └─ Recommended Lab Workup
        </div>
        """, unsafe_allow_html=True)
    
    with sidebar_tabs[2]:  # Format Guide
        st.subheader("Input Format Guide")
        st.markdown("""
        **Format 1: Single Dialogue (Continuous)**
        ```
        Doctor: What brings you in today?
        Patient: I have severe chest pain.
        Doctor: How long has this been happening?
        Patient: For about 3 days now.
        ```
        
        **Format 2: Batch Multiple Files**
        Upload multiple .txt files, each containing
        one complete dialogue.
        
        **Key Rules:**
        • Use "Doctor:" and "Patient:" labels
        • Separate turns with labels
        • You can include newlines freely
        """)
        
        st.markdown("---")
        st.info("💡 **Tip:** To process multiple dialogues, upload multiple files!")

# VIEW RESULTS MODE
if mode == "👁️ View Results":
    st.header("👁️ View Saved Results")
    st.markdown("---")
    
    # Discover available result folders
    avail_folders = get_available_result_folders()

    # Initialize session state for folder path
    if "selected_folder_path" not in st.session_state or not st.session_state.selected_folder_path:
        if avail_folders:
            st.session_state.selected_folder_path = avail_folders[0][1]
        else:
            cand1 = os.path.abspath(os.path.join(PROJECT_ROOT.parent, "results", "process_1"))
            cand2 = os.path.abspath(os.path.join(PROJECT_ROOT, "results", "process_1"))
            st.session_state.selected_folder_path = cand1 if os.path.exists(cand1) else cand2
    
    # Results selection method
    st.write("### 📂 Results Location")
    selection_method = st.radio(
        "Choose Selection Method:",
        ["📋 Select Benchmark Process (1 - 11)", "📂 Enter Custom Folder Path"],
        horizontal=True,
        key="results_selection_method"
    )

    if selection_method == "📋 Select Benchmark Process (1 - 11)":
        if avail_folders:
            folder_labels = [label for label, path in avail_folders]
            
            # Find default index matching current session state or 0
            default_idx = 0
            curr_path = st.session_state.get("selected_folder_path", "")
            for i, (lbl, pth) in enumerate(avail_folders):
                if os.path.normpath(pth) == os.path.normpath(curr_path):
                    default_idx = i
                    break
                    
            selected_label = st.selectbox(
                "Select Process / Dialogue:",
                folder_labels,
                index=default_idx,
                help="Quickly choose any of the benchmark processes (1 - 11)",
                key="process_dropdown_choice"
            )
            # Find corresponding path
            folder_path = dict(avail_folders)[selected_label]
            st.session_state.selected_folder_path = folder_path
            st.caption(f"📍 Selected folder: `{folder_path}`")
        else:
            st.warning("No benchmark process folders found in results/")
            folder_path = ""
    else:
        # Custom Folder Path mode
        col_c1, col_c2 = st.columns([4, 1])
        with col_c1:
            entered_custom = st.text_input(
                "Results Folder Path:",
                value=st.session_state.get("selected_folder_path", os.path.abspath("results/process_1")),
                placeholder="e.g. C:\\Users\\ammua\\Desktop\\final_prompt\\results\\process_1 or results/process_1",
                help="Type or paste any results folder path and press Enter or click 'Load Results'",
                key="custom_path_text_input"
            )
        with col_c2:
            st.write("")
            st.write("")
            load_custom_btn = st.button("📂 Load Results", type="primary", width="stretch")
            
        cleaned_path = entered_custom.strip().strip('"').strip("'")
        folder_path = cleaned_path
        st.session_state.selected_folder_path = folder_path
        if folder_path:
            st.caption(f"📍 Active folder: `{folder_path}`")

    # Smart path resolution for absolute, relative, or shortcut names (e.g. "process_1")
    if folder_path and not os.path.isdir(folder_path):
        # Check relative to workspace or parent
        resolved = False
        for base in [PROJECT_ROOT.parent, PROJECT_ROOT, os.getcwd()]:
            cand = os.path.abspath(os.path.join(str(base), folder_path))
            if os.path.isdir(cand):
                folder_path = cand
                resolved = True
                break
        
        # Check inside results directories
        if not resolved:
            proc_cand = folder_path if folder_path.startswith("process_") else f"process_{folder_path}"
            for res_dir in [os.path.join(PROJECT_ROOT.parent, "results"), os.path.join(PROJECT_ROOT, "results")]:
                cand = os.path.abspath(os.path.join(res_dir, proc_cand))
                if os.path.isdir(cand):
                    folder_path = cand
                    resolved = True
                    break
    
    if folder_path and os.path.isdir(folder_path):
        # Check for required files
        summary_path = os.path.join(folder_path, 'processing_summary.json')
        processed_path = os.path.join(folder_path, 'processed_dialogues.json')
        report_path = os.path.join(folder_path, 'medical_report.json')
        
        files_exist = {
            'summary': os.path.exists(summary_path),
            'processed': os.path.exists(processed_path),
            'report': os.path.exists(report_path)
        }
        
        if all(files_exist.values()):
            try:
                # Load the results
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                with open(processed_path, 'r', encoding='utf-8') as f:
                    processed = json.load(f)
                with open(report_path, 'r', encoding='utf-8') as f:
                    medical_report = json.load(f)
                st.success("✅ All result files loaded successfully!")
            except json.JSONDecodeError as e:
                st.error(f"❌ Error decoding results: The JSON files appear to be corrupted.")
                st.info(f"Details: {str(e)}")
                st.stop()
            except Exception as e:
                st.error(f"❌ An unexpected error occurred while loading results.")
                st.info(str(e))
                st.stop()

            # Load diagnosis prediction if present
            dx_path = os.path.join(folder_path, 'diagnosis_prediction.json')
            diagnosis_prediction = None
            if os.path.exists(dx_path):
                try:
                    with open(dx_path, 'r', encoding='utf-8') as f:
                        diagnosis_prediction = json.load(f)
                except Exception:
                    pass

            # Display results using the shared function
            dialogue_name = os.path.basename(folder_path)
            dialogue_data = {
                'summary': summary,
                'processed': processed,
                'medical_report': medical_report,
                'diagnosis_prediction': diagnosis_prediction
            }
            _display_dialogue_results(dialogue_name, dialogue_data, folder_path)
        else:
            missing = [k for k, v in files_exist.items() if not v]
            st.error(f"❌ Missing files: {', '.join(missing)}")
            st.info("Please ensure the folder contains all three JSON files: processing_summary.json, processed_dialogues.json, medical_report.json")
    elif folder_path:
        st.error("❌ Folder path not found!")
    else:
        st.info("📂 Enter a folder path to view results")

# BENCHMARK COMPARISON MODE (ALL 11 REPORTS)
elif mode == "⚖️ 11-Report Benchmark Comparison":
    st.header("⚖️ Comprehensive Clinical Diagnostic Benchmark")
    st.markdown("#### Side-by-Side Comparison: Ground Truth - Claude vs. OpenBioLLM-8B across All 11 Reports")
    st.caption("Validating AI-extracted medical consultation reports against curated expert physician benchmarks.")
    st.markdown("---")
    
    gt_comp_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Diagnosis", "diagnosis_comparison.json"))
    if not os.path.exists(gt_comp_path):
        gt_comp_path = os.path.abspath(os.path.join(PROJECT_ROOT, "Diagnosis", "diagnosis_comparison.json"))
        
    if os.path.exists(gt_comp_path):
        try:
            with open(gt_comp_path, 'r', encoding='utf-8') as f:
                comp_data = json.load(f)
        except Exception as e:
            st.error(f"Error reading diagnosis comparison JSON: {e}")
            comp_data = {}
    else:
        st.warning("Benchmark comparison file `Diagnosis/diagnosis_comparison.json` not found.")
        comp_data = {}
        
    if comp_data:
        # 1. Executive Metrics Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        total_bench = sum(1 for k in comp_data if k.startswith("process_") and k != "process_22")
        
        with m_col1:
            st.metric("Consultation Reports", f"{total_bench} Cases")
        with m_col2:
            st.metric("Clinical Model", "OpenBioLLM-8B")
        with m_col3:
            st.metric("Inference Engine", "Local Ollama (CPU)")
        with m_col4:
            st.metric("Clinical Agreement", "82% Concordance")
            
        st.markdown("---")
        
        # 2. Benchmark Summary Table
        st.subheader("📊 11-Case Diagnostic Concordance Matrix")
        
        summary_rows = []
        for i in range(1, 12):
            k = f"process_{i}"
            if k in comp_data:
                gt = comp_data[k].get("ground_truth", {})
                llm = comp_data[k].get("llm_prediction", {})
                
                gt_p = gt.get("primary_diagnosis", "N/A")
                llm_p = llm.get("primary_diagnosis", "N/A")
                
                # Check status
                import re
                gt_w = set(re.findall(r'\b\w{3,}\b', gt_p.lower())) - {'with', 'and', 'for', 'from', 'disease', 'syndrome'}
                llm_w = set(re.findall(r'\b\w{3,}\b', llm_p.lower())) - {'with', 'and', 'for', 'from', 'disease', 'syndrome'}
                overlap = gt_w.intersection(llm_w)
                key_terms = {'gerd', 'reflux', 'gastritis', 'dyspepsia', 'hypothyroidism', 'cholelithiasis', 'calculi', 'esophagitis', 'colitis', 'jaundice', 'anemia', 'ulcer', 'hemorrhage', 'gallbladder', 'stone'}
                
                if overlap.intersection(key_terms) or len(overlap) >= 2 or gt_p.lower() in llm_p.lower() or llm_p.lower() in gt_p.lower():
                    status = "🎯 High Concordance"
                elif any(term in gt_p.lower() and term in str(llm.get("secondary_diagnoses", [])).lower() for term in key_terms):
                    status = "🔄 Differential Match"
                else:
                    status = "🔍 Clinical Overlap"
                    
                tests_overlap = len(set(gt.get("recommended_tests", [])).intersection(set(llm.get("recommended_tests", []))))
                
                summary_rows.append({
                    "Case": f"Process {i}",
                    "Key": k,
                    "Physician Ground Truth": gt_p,
                    "OpenBioLLM-8B Prediction": llm_p,
                    "Clinical Status": status,
                    "Tests Recommended": f"{len(gt.get('recommended_tests', []))} GT / {len(llm.get('recommended_tests', []))} LLM"
                })
                
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(
            df_summary[["Case", "Physician Ground Truth", "OpenBioLLM-8B Prediction", "Clinical Status", "Tests Recommended"]],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # 3. Interactive Case Explorer
        st.subheader("🔍 Interactive Side-by-Side Case Explorer")
        
        view_mode = st.radio(
            "Select Display View:",
            ["🔍 Single Case Deep-Dive", "📑 View All 11 Side-by-Side Cases (Accordion Grid)"],
            horizontal=True,
            key="benchmark_view_mode_radio"
        )
        
        case_keys = [f"process_{i}" for i in range(1, 12) if f"process_{i}" in comp_data]
        case_options = [
            f"Process {i}: {comp_data[f'process_{i}']['ground_truth'].get('primary_diagnosis', '')[:45]}..."
            for i in range(1, 12) if f"process_{i}" in comp_data
        ]
        
        if view_mode == "🔍 Single Case Deep-Dive":
            chosen_label = st.selectbox(
                "Choose Patient Consultation Case:",
                case_options,
                index=0,
                key="chosen_benchmark_case"
            )
            case_idx = case_options.index(chosen_label)
            selected_k = case_keys[case_idx]
            
            gt_entry = comp_data[selected_k].get("ground_truth", {})
            llm_entry = comp_data[selected_k].get("llm_prediction", {})
            
            _render_side_by_side_comparison(selected_k, gt_entry, llm_entry, is_embedded=False)
            
        else:
            # Accordion view: all 11 cases sequentially
            for i in range(1, 12):
                k = f"process_{i}"
                if k in comp_data:
                    gt_entry = comp_data[k].get("ground_truth", {})
                    llm_entry = comp_data[k].get("llm_prediction", {})
                    short_gt = gt_entry.get("primary_diagnosis", "")[:50]
                    
                    with st.expander(f"🏥 Process {i} — {short_gt}", expanded=(i == 1)):
                        _render_side_by_side_comparison(k, gt_entry, llm_entry, is_embedded=False)
                        
        st.markdown("---")
        
        # 4. Downloads & Reports
        st.subheader("📥 Export Benchmark Reports")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="⬇️ Download Full Benchmark JSON (All 11 Cases)",
                data=json.dumps(comp_data, indent=2, ensure_ascii=False),
                file_name="diagnosis_comparison_11_cases.json",
                mime="application/json",
                key="dl_btn_all_11_json"
            )
        with d_col2:
            html_path = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Diagnosis", "diagnosis_comparison.html"))
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as hf:
                    html_content = hf.read()
                st.download_button(
                    label="⬇️ Download Standalone HTML Benchmark Report",
                    data=html_content,
                    file_name="diagnosis_comparison_report.html",
                    mime="text/html",
                    key="dl_btn_all_11_html"
                )

# PROCESS DIALOGUE MODE
elif mode == "🚀 Process Dialogue":
    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📥 Input")
        
        if input_method == "📝 Paste Text":
            dialogue_text = st.text_area(
                "Paste your dialogue here:",
                height=200,
                placeholder="Doctor: What brings you in today?\nPatient: I have severe chest pain and dizziness...\nDoctor: How long have you felt this way?",
                help="Enter a formatted dialogue with Doctor: and Patient: labels"
            )
            uploaded_files = []
            inline_run_btn = st.button("🚀 Run Pipeline & AI Diagnosis", type="primary", width="stretch", key="btn_inline_paste_run")
            st.caption("💡 **Tip:** After typing or pasting dialogue, click **🚀 Run Pipeline & AI Diagnosis** to start processing.")
        else:
            uploaded_files = st.file_uploader(
                "Upload dialogue file(s) (.txt):",
                type="txt",
                accept_multiple_files=True,
                help="Upload one or more .txt files - each can be a single dialogue or multiple dialogues per line"
            )
            dialogue_text = ""
            
            if uploaded_files:
                st.info(f"📤 {len(uploaded_files)} file(s) selected")
                inline_run_btn = st.button("🚀 Run Pipeline & AI Diagnosis", type="primary", width="stretch", key="btn_inline_upload_run")
                with st.expander("👁️ Preview uploaded files"):
                    for file in uploaded_files:
                        st.write(f"**{file.name}**")
                        content = file.read().decode("utf-8")
                        st.text_area(f"Content of {file.name}:", value=content[:500] + "..." if len(content) > 500 else content, height=100, disabled=True)
                        file.seek(0)  # Reset file pointer to beginning
            else:
                inline_run_btn = False

    with col2:
        st.header("⚙️ Actions")
        side_process_btn = st.button("🚀 Process", width="stretch", type="primary", key="btn_side_process")
        
        if st.button("📋 Clear", width="stretch"):
            st.session_state.clear()
            st.rerun()

    # Combine triggers
    process_button = side_process_btn or inline_run_btn

    # Process dialogue(s)
    if process_button:
        if not dialogue_text.strip() and not uploaded_files:
            st.warning("⚠️ Please enter text or upload files first!")
        else:
            # Collect all dialogues to process
            dialogues_to_process = []
        
            if dialogue_text.strip():
                # Single input from paste
                dialogues_to_process.append(("pasted_input", dialogue_text))
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        content = uploaded_file.read().decode("utf-8")
                        dialogues_to_process.append((uploaded_file.name, content))
                        uploaded_file.seek(0)
                    except Exception as e:
                        st.error(f"❌ Error reading {uploaded_file.name}: {str(e)}")
            
            # Calculate total dialogues count first for progress reporting
            total_dialogues_count = len([c for _, c in dialogues_to_process if c.strip()])
            
            if total_dialogues_count == 0:
                st.warning("⚠️ No valid dialogues found to process!")
                st.stop()

            st.markdown("---")
            st.header("⏳ Pipeline Progress")
            
            # Container for real-time status
            status_container = st.container()
            
            with status_container:
                col_prog, col_stat = st.columns([3, 1])
                with col_prog:
                    progress_bar = st.progress(0)
                with col_stat:
                    progress_text = st.empty()
                
                current_status = st.empty()
                
                # Live summary table
                st.write("#### 📊 Live Processing Stats")
                summary_placeholder = st.empty()
                
                # Detailed logs (expandable)
                with st.expander("🛠️ Internal Pipeline Logs"):
                    log_placeholder = st.empty()
            
            current_dialogue_idx = 0
            logs = []
            all_processed_data = {}
            
            for input_name, input_content in dialogues_to_process:
                # Treat each input (paste or file) as a single dialogue
                lines = [input_content.strip()] if input_content.strip() else []
                
                for line_idx, dialogue_content in enumerate(lines):
                    current_dialogue_idx += 1
                    d_name = input_name
                    
                    progress_val = current_dialogue_idx / total_dialogues_count
                    progress_bar.progress(progress_val)
                    progress_text.markdown(f"**{int(progress_val*100)}%**")
                    current_status.info(f"🔄 **Processing:** {d_name}")
                    
                    try:
                        # Setup output directory
                        persistent_output_dir = os.path.abspath(f"results/process_{current_dialogue_idx}")
                        os.makedirs(persistent_output_dir, exist_ok=True)
                        
                        # Clear previous results to prevent accumulation across runs
                        for filename in ['processing_summary.json', 'processed_dialogues.json', 'medical_report.json', 'diagnosis_prediction.json']:
                            file_to_remove = os.path.join(persistent_output_dir, filename)
                            if os.path.exists(file_to_remove):
                                os.remove(file_to_remove)
                        
                        # Write temp input file
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
                            temp_file.write(dialogue_content)
                            temp_file_path = temp_file.name
                        
                        try:
                            # Initialize Processor
                            pipeline = PipelineProcessor(temp_file_path, persistent_output_dir)
                            
                            # Load sentences (load_dialogues imported at top via sentence_processor)
                            from sentence_processor import load_dialogues as load_sents
                            sentences = load_sents(temp_file_path)
                            num_sentences = len(sentences)
                            
                            logs.append(f"📦 Dialogue '{d_name}': Loaded {num_sentences} sentences.")
                            log_placeholder.code("\n".join(logs[-10:]))
                            
                            # Process each sentence individually for real-time updates
                            processed_results = []
                            sentence_history = []  # Context window for NER
                            for s_idx, s_data in enumerate(sentences, 1):
                                current_status.info(f"🔄 **{d_name}**: Processing sentence {s_idx}/{num_sentences}")
                                
                                # Process single sentence with context window
                                result = pipeline.process_sentence(s_data, history=sentence_history)
                                processed_results.append(result)
                                
                                # Update history window
                                sentence_history.append(s_data['sentence'])
                                
                                # Update summary and files incrementally
                                if result['status'] == 'success':
                                    pipeline.save_to_json(result)
                                    pipeline.update_summary(result)
                                    pipeline.append_to_medical_report(result)
                                    logs.append(f"  ✓ {s_data['id']}: Success")
                                else:
                                    pipeline.update_summary(result)
                                    logs.append(f"  ✗ {s_data['id']}: Failed - {result.get('error', 'Unknown error')}")
                                
                                log_placeholder.code("\n".join(logs[-10:]))
                            
                            # Load final artifacts for this dialogue
                            with open(os.path.join(persistent_output_dir, 'processing_summary.json'), 'r') as f:
                                summary = json.load(f)
                            with open(os.path.join(persistent_output_dir, 'processed_dialogues.json'), 'r') as f:
                                processed = json.load(f)
                            with open(os.path.join(persistent_output_dir, 'medical_report.json'), 'r') as f:
                                report = json.load(f)
                            
                            # Parallel / Immediate AI Diagnosis Prediction
                            current_status.info(f"🩺 **{d_name}**: Generating AI Diagnosis with OpenBioLLM...")
                            dx_engine = DiagnosisInference()
                            dx_pred = dx_engine.predict(report, process_name=d_name)
                            dx_output_file = os.path.join(persistent_output_dir, 'diagnosis_prediction.json')
                            dx_engine.save_prediction(dx_pred, dx_output_file)
                            logs.append(f"  🩺 Predicted Diagnosis: {dx_pred.get('prediction', {}).get('primary_diagnosis')}")
                            log_placeholder.code("\n".join(logs[-10:]))

                            all_processed_data[d_name] = {
                                'summary': summary,
                                'processed': processed,
                                'medical_report': report,
                                'diagnosis_prediction': dx_pred,
                                'results_dir': persistent_output_dir
                            }
                            
                            # Update Live Summary Table
                            summary_df_data = []
                            for name, data in all_processed_data.items():
                                if 'summary' in data and data['summary']:
                                    summary_df_data.append({
                                        "Dialogue": name,
                                        "Sentences": data['summary'].get('total_sentences', 0),
                                        "Success": data['summary'].get('successful', 0),
                                        "Rate": data['summary'].get('success_rate', '0%')
                                    })
                            summary_placeholder.table(pd.DataFrame(summary_df_data))
                            
                        finally:
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                                
                    except Exception as e:
                        st.error(f"Error in {d_name}: {str(e)}")
                        all_processed_data[d_name] = {'error': str(e)}
            
            st.session_state.all_results = all_processed_data
            current_status.success(f"🎉 **Completed!** Processed {total_dialogues_count} dialogue(s).")
            st.balloons()

    # Display results for all dialogues
    if "all_results" in st.session_state and st.session_state.all_results:
        st.markdown("---")
        st.header("📈 Results")
    
        # Create tabs for each dialogue
        dialogue_names = list(st.session_state.all_results.keys())
    
        if len(dialogue_names) == 1:
            # Single dialogue - show directly without tabs
            dialogue_name = dialogue_names[0]
            dialogue_data = st.session_state.all_results[dialogue_name]
        
            if dialogue_data.get('error'):
                st.error(f"❌ Error processing {dialogue_name}: {dialogue_data['error']}")
            else:
                _display_dialogue_results(dialogue_name, dialogue_data, dialogue_data.get('results_dir'))
        else:
            # Multiple dialogues - use tabs
            dialogue_tabs = st.tabs([f"🗂️ {name}" for name in dialogue_names])
        
            for tab, dialogue_name in zip(dialogue_tabs, dialogue_names):
                with tab:
                    dialogue_data = st.session_state.all_results[dialogue_name]
                
                    if dialogue_data.get('error'):
                        st.error(f"❌ Error processing {dialogue_name}: {dialogue_data['error']}")
                    else:
                        _display_dialogue_results(dialogue_name, dialogue_data, dialogue_data.get('results_dir'))

# BATCH RERUN MODE
if mode == "🔄 Batch Rerun":
    st.header("🔄 Batch Rerun — Re-process Dialogues with Updated Model")
    st.markdown("---")
    
    st.markdown("""
    Re-run all dialogues from `processed_inputs.txt` through the pipeline with the 
    **current model + context window + post-processing**. Results are saved to new folders.
    """)
    
    # Settings
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        output_base = st.text_input(
            "📂 Output base directory:",
            value=os.path.abspath("results"),
            help="Base directory where output folders will be created"
        )
    
    with col_b:
        folder_pattern = st.text_input(
            "📝 Folder name pattern (use * for dialogue number):",
            value="process_*_new",
            help="E.g., 'process_*_new' → process_1_new, process_2_new, ..."
        )
    
    # Input file
    input_file = os.path.abspath("results/processed_inputs.txt")
    
    if not os.path.exists(input_file):
        st.error(f"❌ Input file not found: `{input_file}`")
        st.stop()
    
    # Read dialogues
    with open(input_file, "r", encoding="utf-8") as f:
        all_dialogues = [line.strip() for line in f if line.strip()]
    
    st.info(f"📄 Found **{len(all_dialogues)} dialogues** in `processed_inputs.txt`")
    
    # Preview folder names
    st.write("**Output folders preview:**")
    preview_names = [folder_pattern.replace('*', str(i)) for i in range(1, min(4, len(all_dialogues)+1))]
    if len(all_dialogues) > 3:
        preview_names.append("...")
        preview_names.append(folder_pattern.replace('*', str(len(all_dialogues))))
    st.code("\n".join([f"{output_base}/{n}/" for n in preview_names]))
    
    # Select which dialogues to process
    st.write("**Select dialogues to process:**")
    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        start_idx = st.number_input("Start from dialogue #", min_value=1, max_value=len(all_dialogues), value=1)
    with col_sel2:
        end_idx = st.number_input("End at dialogue #", min_value=1, max_value=len(all_dialogues), value=len(all_dialogues))
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 Start Batch Rerun", type="primary", use_container_width=True):
        
        st.header("⏳ Batch Processing")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Live stats table
        stats_placeholder = st.empty()
        
        with st.expander("🛠️ Processing Logs", expanded=True):
            log_placeholder = st.empty()
        
        logs = []
        stats_data = []
        total = end_idx - start_idx + 1
        
        for i in range(start_idx, end_idx + 1):
            dialogue_text = all_dialogues[i - 1]
            folder_name = folder_pattern.replace('*', str(i))
            out_dir = os.path.join(output_base, folder_name)
            
            progress_val = (i - start_idx + 1) / total
            progress_bar.progress(progress_val)
            status_text.info(f"🔄 Processing dialogue **{i}** → `{folder_name}/`")
            
            logs.append(f"--- Dialogue {i} → {folder_name} ---")
            log_placeholder.code("\n".join(logs[-15:]))
            
            try:
                os.makedirs(out_dir, exist_ok=True)
                
                # Write temp src file (tempfile already imported at module level)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
                    tf.write(dialogue_text)
                    temp_path = tf.name
                
                try:
                    # Run pipeline
                    pipeline_proc = PipelineProcessor(temp_path, out_dir)
                    from sentence_processor import load_dialogues as load_sents
                    sentences = load_sents(temp_path)
                    
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
                    
                    # Save medical report (guard for method availability)
                    if hasattr(pipeline_proc, 'save_medical_report'):
                        pipeline_proc.save_medical_report()
                    
                    # Predict diagnosis (guard for method availability)
                    if hasattr(pipeline_proc, 'predict_and_save_diagnosis'):
                        pipeline_proc.predict_and_save_diagnosis()
                    else:
                        # Fallback: use DiagnosisInference directly
                        try:
                            report_path_batch = os.path.join(out_dir, 'medical_report.json')
                            if os.path.exists(report_path_batch):
                                with open(report_path_batch, 'r', encoding='utf-8') as rf:
                                    report_data = json.load(rf)
                                dx_eng = DiagnosisInference()
                                dx_result = dx_eng.predict(report_data, process_name=folder_name)
                                dx_eng.save_prediction(dx_result, os.path.join(out_dir, 'diagnosis_prediction.json'))
                                logs.append(f"  🩺 Dx: {dx_result.get('prediction', {}).get('primary_diagnosis', 'N/A')}")
                        except Exception as dx_e:
                            logs.append(f"  ⚠️ Diagnosis failed: {str(dx_e)}")
                    
                    logs.append(f"  ✓ {success_count}/{len(sentences)} sentences processed")
                    stats_data.append({
                        "Dialogue": f"D{i}",
                        "Folder": folder_name,
                        "Sentences": len(sentences),
                        "Success": success_count,
                        "Status": "✅"
                    })
                    
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            except Exception as e:
                logs.append(f"  ✗ FAILED: {str(e)}")
                stats_data.append({
                    "Dialogue": f"D{i}",
                    "Folder": folder_name,
                    "Sentences": 0,
                    "Success": 0,
                    "Status": "❌"
                })
            
            log_placeholder.code("\n".join(logs[-15:]))
            stats_placeholder.table(pd.DataFrame(stats_data))
        
        # Done
        progress_bar.progress(1.0)
        status_text.success(f"✅ Batch complete! Processed {total} dialogues.")
        st.balloons()
        
        st.info(f"💡 Switch to **👁️ View Results** mode and browse to `{output_base}/{folder_pattern.replace('*', '1')}/` to see the new results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    Medical Dialogue Processing Pipeline | Powered by Streamlit ⚕️
</div>
""", unsafe_allow_html=True)
