"""
Main Pipeline: Processes dialogues through NER, DAC, and SLI tasks
Combines outputs into JSON format
"""

import os
import sys
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_processor import load_dialogues
from ner_inference import NERInference

# ── Global NER model (lazy-loaded) ──────────────────────────
_GLOBAL_NER_MODEL = None

def get_ner_model():
    global _GLOBAL_NER_MODEL
    if _GLOBAL_NER_MODEL is None:
        model_dir = str(PROJECT_ROOT / 'models' / 'ner')
        _GLOBAL_NER_MODEL = NERInference(model_dir)
    return _GLOBAL_NER_MODEL

def infer_ner_custom(sentence: str, history: list = None) -> list:
    model = get_ner_model()
    return model.predict(sentence, history)

# from sli_inference import infer_sli  # Using actual DeBERTa SLI model (COMMENTED OUT)

# Context window: match training (HISTORY_TURNS=2 in train_ner.py)
NER_HISTORY_TURNS = 2


_GLOBAL_DAC_CONFIG = None
_GLOBAL_DAC_MODEL = None


def infer_dac(sentence: str, history: List[str] = None, speaker: str = None) -> str:
    """
    Call new HuggingFace DAC inference with context window.
    """
    global _GLOBAL_DAC_MODEL
    try:
        if _GLOBAL_DAC_MODEL is None:
            # Import our new HuggingFace DAC inference script
            from dac_inference import DACInference
            model_path = os.path.join(str(PROJECT_ROOT), 'models', 'dac')
            _GLOBAL_DAC_MODEL = DACInference(model_path)
            
        return _GLOBAL_DAC_MODEL.predict(sentence, history, speaker=speaker)
    except Exception as e:
        print(f"Warning: HF DAC inference fallback to 'other': {e}")
        import traceback
        traceback.print_exc()
        return 'other'


class PipelineProcessor:
    """Main pipeline for processing dialogues through multiple tasks"""
    
    def __init__(self, dialogue_file: str, output_dir: str = 'pipeline_output'):
        """
        Initialize pipeline
        
        Args:
            dialogue_file: Path to dialogue .src file
            output_dir: Directory to save outputs
        """
        dialogue_path = Path(dialogue_file)
        output_path = Path(output_dir)

        if not dialogue_path.is_absolute():
            dialogue_path = PROJECT_ROOT / dialogue_path
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        self.dialogue_file = str(dialogue_path)
        self.output_dir = str(output_path)
        self.results = {}
        self.medical_report_data = {
            'chief_complaint': [],
            'history_of_present_illness': [],
            'suggestions': [],
            'diagnosis': [],
            'past_medical_history': [],
                'social_history': []
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_sentence(self, sentence_data: Dict[str, str],
                          history: List[str] = None) -> Dict[str, Any]:
        """
        Process single sentence through all three tasks.
        
        Args:
            sentence_data: Dict with 'id', 'sentence', and optional 'speaker'
            history: List of previous sentences for NER context window
            
        Returns:
            Dict with results from all tasks
        """
        sent_id = sentence_data['id']
        sentence = sentence_data['sentence']
        speaker = sentence_data.get('speaker', 'Unknown')
        if history is None:
            history = []
        
        try:
            # Run NER task with context window (matching training format)
            try:
                ner_tags = infer_ner_custom(sentence, history)
                
            except Exception as e:
                print(f"NER extraction failed: {e}")
                raise
            
            # Run DAC task with context window
            try:
                dac_tag = infer_dac(sentence, history, speaker=speaker)
                if not isinstance(dac_tag, str):
                    print(f"Warning: DAC returned non-string type: {type(dac_tag)}")
                    dac_tag = str(dac_tag)
            except Exception as e:
                print(f"DAC inference failed: {e}")
                raise
            
            # Run SLI task (COMMENTED OUT)
            sli_tags = []  # Empty list when SLI is disabled
            
            result = {
                'sentence_id': sent_id,
                'speaker': speaker,
                'sentence': sentence,
                'NER_tags': ner_tags,
                'DAC_tag': dac_tag,
                'SLI_tags': [],  # SLI_tags disabled
                'status': 'success'
            }
            
            print(f"[OK] Processed {sent_id} ({speaker}): {sentence[:60]}...", flush=True)
            
        except Exception as e:
            result = {
                'sentence_id': sent_id,
                'speaker': speaker,
                'sentence': sentence,
                'NER_tags': [],
                'DAC_tag': 'error',
                'SLI_tags': [],
                'status': 'error',
                'error': str(e)
            }
            print(f"[FAIL] Error processing {sent_id}: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    

    def check_chief_complaint_condition(self, result: dict) -> bool:
        speaker = result.get('speaker', '').strip().lower()
        # Doctor inquiries/questions never form chief complaint
        if speaker == 'doctor':
            return False

        ner_required_types = ['SYMPTOM_VALUE', 'SYMPTOM_LOCATION']
        dac_target_tags = ['inform-symptom', 'inform-basic_info']
        
        ner_match = False
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type in ner_required_types:
                        ner_match = True
                        break
        
        dac_match = False
        dac_tag_normalized = result.get('DAC_tag', '').lower() if result.get('DAC_tag') else ''
        if any(target in dac_tag_normalized for target in dac_target_tags):
            dac_match = True
        elif speaker == 'patient' and ('request-symptom' in dac_tag_normalized or 'inform' in dac_tag_normalized):
            dac_match = True

        if not (ner_match and dac_match):
            return False

        # Temporal Leakage Filter: exclude distant resolved past history (e.g., 2018/2019)
        sentence_lower = result.get('sentence', '').lower()

        # Prior consultations/diagnoses/resolved events should never be chief complaint
        prior_consult_patterns = [
            r'\bafter\s+running\s+tests\b',
            r'\bafter\s+the\s+usg\b',
            r'\bshowed\s+(?:the|it\s+to\s+the)\s+(?:report|doctor)\b',
            r'\breferred\s+us\b',
            r'\bno\s+need\s+for\s+an\s+operation\b',
            r'\bmenstrua(?:l|tion)\b',
            r'\brheumatoid\s+arthritis\b',
            r'\bknee\s+pain\b'
        ]
        if any(re.search(pat, sentence_lower) for pat in prior_consult_patterns):
            return False

        past_patterns = [
            r'\bin\s+20\d\d\b',
            r'\b\d+\s+years?\s+ago\b',
            r'\b\d+\s+months?\s+ago\b',
            r'\bback\s+in\b',
            r'\byear\s+later\b',
            r'\bhistory\s+of\b',
            r'\bpreviously\b',
            r'\bat\s+that\s+time\b',
            r'\blong\s+time\s+ago\b'
        ]
        has_past_marker = any(re.search(pat, sentence_lower) for pat in past_patterns)

        recent_patterns = [
            r'\bnow\b',
            r'\bcurrently\b',
            r'\btoday\b',
            r'\byesterday\b',
            r'\bpast\s+few\s+days\b',
            r'\blast\s+few\s+days\b',
            r'\bfor\s+\d+\s+days\b',
            r'\bsince\s+yesterday\b',
            r'\bsince\s+2024\b',
            r'\bpresent\b',
            r'\bstill\b',
            r'\bpersists?\b'
        ]
        has_recent_marker = any(re.search(pat, sentence_lower) for pat in recent_patterns)

        if has_past_marker and not has_recent_marker:
            return False

        return True

    def check_history_of_present_illness_condition(self, result: dict) -> bool:
        speaker = result.get('speaker', '').strip().lower()
        sentence = result.get('sentence', '')
        # Doctor asking questions is screening, not HPI narrative
        if speaker == 'doctor' and '?' in sentence:
            return False

        ner_required_types = ['SYMPTOM_CHARACTERISTICS', 'SYMPTOM_FREQUENCY', 'SYMPTOM_LOCATION', 'TEMPORAL', 'SYMPTOM_VALUE']
        dac_target_tags = ['inform-basic_info', 'inform-symptom', 'inform-etiology', 'inform-existing_examination_and_treatment']
        
        ner_match = False
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type in ner_required_types:
                        ner_match = True
                        break
        
        dac_match = False
        dac_tag_normalized = result.get('DAC_tag', '').lower() if result.get('DAC_tag') else ''
        if any(target in dac_tag_normalized for target in dac_target_tags):
            dac_match = True
        elif speaker == 'patient' and ner_match:
            dac_match = True
                
        return ner_match and dac_match

    def check_past_medical_history_condition(self, result: dict) -> bool:
        speaker = result.get('speaker', '').strip().lower()
        sentence = result.get('sentence', '')
        dac_tag_normalized = result.get('DAC_tag', '').lower() if result.get('DAC_tag') else ''

        # Filter: doctor screening questions should NEVER be recorded as confirmed patient PMH
        if speaker == 'doctor' and ('?' in sentence or 'request' in dac_tag_normalized):
            return False

        dac_path_b = ['inform-existing_examination_and_treatment']
        dac_path_c = ['inform-basic_info', 'inform-medical_advice', 'inform-drug_recommendation', 'inform-precautions']
        
        has_medical_history_value = False
        has_medical_test_name = False
        has_any_entity = False
        
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type != 'O':
                        has_any_entity = True
                    if ner_type == 'MEDICAL_HISTORY_VALUE':
                        has_medical_history_value = True
                    if ner_type == 'MEDICAL_TEST_NAME':
                        has_medical_test_name = True

        sentence_lower = sentence.lower()
        past_patterns = [
            r'\bin\s+20\d\d\b',
            r'\b\d+\s+years?\s+ago\b',
            r'\b\d+\s+months?\s+ago\b',
            r'\bback\s+in\b',
            r'\byear\s+later\b',
            r'\bhistory\s+of\b',
            r'\bpreviously\b',
            r'\bat\s+that\s+time\b',
            r'\blong\s+time\s+ago\b'
        ]
        has_past_marker = any(re.search(pat, sentence_lower) for pat in past_patterns)

        # Path A: Explicit medical history or test name from non-doctor inquiry
        if has_medical_history_value or has_medical_test_name:
            return True
            
        # Path B: Existing examination / treatment with entity
        if any(target in dac_tag_normalized for target in dac_path_b):
            if has_any_entity:
                return True

        # Path C: Past resolved events reported by patient
        if has_past_marker and has_any_entity and speaker == 'patient':
            return True
                
        # Path D: Inform acts with medical history value
        if any(target in dac_tag_normalized for target in dac_path_c):
            if has_medical_history_value:
                return True
                
        return False

    def check_social_history_condition(self, result: dict) -> bool:
        ner_required_types = ['HABIT_VALUE', 'HABIT_FREQUENCY', 'EXPOSURE_VALUE']
        dac_target_tags = ['inform-basic_info', 'inform-symptom', 'inform-etiology', 'inform-existing_examination_and_treatment']
        
        has_sh = False
        has_medical_history = False
        
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type in ner_required_types:
                        has_sh = True
                    if ner_type == 'MEDICAL_HISTORY_VALUE':
                        has_medical_history = True
                        
        if has_medical_history:
            return False
            
        dac_match = False
        dac_tag_normalized = result['DAC_tag'].lower() if result.get('DAC_tag') else ''
        if any(target in dac_tag_normalized for target in dac_target_tags):
            dac_match = True
            
        return has_sh and dac_match

    def check_diagnosis_condition(self, result: dict) -> bool:
        dac_target_tags = ['diagnose', 'inform-medical_advice']
        ner_required_types = ['SYMPTOM_VALUE']
        
        dac_match = False
        dac_tag_normalized = result['DAC_tag'].lower() if result.get('DAC_tag') else ''
        if any(target in dac_tag_normalized for target in dac_target_tags):
            dac_match = True
            
        if not dac_match:
            return False
            
        has_disease = False
        has_medical_history = False
        
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type in ner_required_types:
                        has_disease = True
                    if ner_type == 'MEDICAL_HISTORY_VALUE':
                        has_medical_history = True
                        
        return has_disease and not has_medical_history

    def check_suggestion_condition(self, result: dict) -> bool:
        dac_target_tags = ['inform-precautions', 'inform-drug_recommendation', 'inform-medical_advice']
        ner_required_types = ['MEDICATION', 'MEDICAL_TEST_NAME']
        
        dac_match = False
        dac_tag_normalized = result['DAC_tag'].lower() if result.get('DAC_tag') else ''
        if any(target in dac_tag_normalized for target in dac_target_tags):
            dac_match = True
            
        if not dac_match:
            return False
            
        has_medication_or_test = False
        has_medical_history = False
        
        if result.get('NER_tags'):
            for ner_tag in result['NER_tags']:
                if isinstance(ner_tag, dict) and 'type' in ner_tag:
                    ner_type = ner_tag['type'].upper().replace('B-', '').replace('I-', '')
                    if ner_type in ner_required_types:
                        has_medication_or_test = True
                    if ner_type == 'MEDICAL_HISTORY_VALUE':
                        has_medical_history = True
                        
        # Fallback: if DAC is inform-precautions, accept even without NER
        if 'inform-precautions' in dac_tag_normalized and not has_medical_history:
            return True
            
        return has_medication_or_test and not has_medical_history
    
    def append_to_medical_report(self, result: dict, filename: str = 'medical_report.json'):
        import os, json
        output_path = os.path.join(self.output_dir, filename)
        
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                medical_report_data = json.load(f)
            if 'suggestions' not in medical_report_data:
                medical_report_data['suggestions'] = []
            if 'diagnosis' not in medical_report_data:
                medical_report_data['diagnosis'] = []
            if 'past_medical_history' not in medical_report_data:
                medical_report_data['past_medical_history'] = []
            if 'social_history' not in medical_report_data:
                medical_report_data['social_history'] = []
        else:
            medical_report_data = {
                'chief_complaint': [],
                'history_of_present_illness': [],
                'suggestions': [],
                'diagnosis': [],
                'past_medical_history': [],
                'social_history': []
            }

        self.medical_report_data = medical_report_data
        
        def create_entry(res):
            return {
                'sentence_id': res['sentence_id'],
                'speaker': res.get('speaker', 'Unknown'),
                'sentence': res['sentence'],
                'NER_tags': res['NER_tags'],
                'DAC_tag': res['DAC_tag'],
                'SLI_tags': res.get('SLI_tags', [])
            }

        def add_if_unique(section_name, res_entry):
            existing_ids = {item.get('sentence_id') for item in medical_report_data[section_name] if isinstance(item, dict)}
            if res_entry['sentence_id'] not in existing_ids:
                medical_report_data[section_name].append(res_entry)
                print(f"    [+] Added to {section_name}")

        entry = create_entry(result)

        if self.check_chief_complaint_condition(result):
            add_if_unique('chief_complaint', entry)
        
        if self.check_history_of_present_illness_condition(result):
            add_if_unique('history_of_present_illness', entry)

        if self.check_suggestion_condition(result):
            add_if_unique('suggestions', entry)
        
        if self.check_diagnosis_condition(result):
            add_if_unique('diagnosis', entry)
        
        if self.check_past_medical_history_condition(result):
            add_if_unique('past_medical_history', entry)
            
        if self.check_social_history_condition(result):
            add_if_unique('social_history', entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(medical_report_data, f, ensure_ascii=False, indent=2)
        
        return output_path

    def process_all_sentences(self, output_filename: str = 'processed_dialogues.json',
                              summary_filename: str = 'processing_summary.json',
                              medical_report_filename: str = 'medical_report.json') -> List[Dict[str, Any]]:
        """
        Process all sentences one at a time and append to files incrementally
        
        Args:
            output_filename: Name for main JSON output
            summary_filename: Name for summary file
            medical_report_filename: Name for medical report file
            
        Returns:
            List of all processing results
        """
        print(f"Loading dialogue from: {self.dialogue_file}")
        sentences = load_dialogues(self.dialogue_file)
        print(f"Loaded {len(sentences)} sentences\n")
        
        print(f"Processing sentences and appending results to files...")
        print("=" * 80)
        
        # Clean previous files to prevent accumulation across runs
        for fname in [output_filename, summary_filename, medical_report_filename, 'diagnosis_prediction.json']:
            target_f = os.path.join(self.output_dir, fname)
            if os.path.exists(target_f):
                try:
                    os.remove(target_f)
                except Exception:
                    pass

        all_results = []
        
        # Sliding history window for NER context (matches training HISTORY_TURNS=2)
        sentence_history = []
        
        # Process each sentence one at a time
        for idx, sentence_data in enumerate(sentences, 1):
            print(f"\n[{idx}/{len(sentences)}] Processing: {sentence_data['sentence'][:70]}...")
            
            # Process single sentence through all tasks (with context window)
            result = self.process_sentence(sentence_data, history=sentence_history)
            all_results.append(result)
            
            # Update history window (keep last NER_HISTORY_TURNS sentences)
            sentence_history.append(sentence_data['sentence'])
            
            if result['status'] == 'success':
                # Append to JSON file
                self.save_to_json(result, output_filename)
                print(f"  ✓ Appended to {output_filename}")
                
                # Update summary file
                self.update_summary(result, summary_filename)
                print(f"  ✓ Updated {summary_filename}")
                
                # Append to medical report file if conditions met
                self.append_to_medical_report(result, medical_report_filename)
                print(f"  ✓ Updated {medical_report_filename}")
            else:
                # Still update summary even for failed sentences
                self.update_summary(result, summary_filename)
                print(f"  ✗ Failed to process, updated summary")
        
        print("\n" + "=" * 80)
        print(f"Processing complete! {len(all_results)} sentences processed.\n")
        
        return all_results
    
    def save_to_json(self, result: Dict[str, Any], filename: str = 'processed_dialogues.json'):
        """
        Append single result to JSON file
        
        Args:
            result: Single processing result dict
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        sent_id = result['sentence_id']
        entry = {
            'sentence': result['sentence'],
            'NER_tags': result['NER_tags'],
            'DAC_tag': result['DAC_tag'],
            'SLI_tags': result['SLI_tags']
        }
        
        # Load existing data or create new
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        else:
            json_data = {}
        
        # Add/update entry
        json_data[sent_id] = entry
        
        # Write back
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def update_summary(self, current_result: Dict[str, Any], filename: str = 'processing_summary.json'):
        """
        Update processing summary incrementally
        
        Args:
            current_result: Single processing result
            filename: Output filename
        """
        summary_path = os.path.join(self.output_dir, filename)
        
        # Load existing summary or create new
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        else:
            summary = {
                'total_sentences': 0,
                'successful': 0,
                'failed': 0,
                'failed_sentences': []
            }
        
        # Update counters
        summary['total_sentences'] += 1
        if current_result['status'] == 'success':
            summary['successful'] += 1
        else:
            summary['failed'] += 1
            summary['failed_sentences'].append(current_result['sentence_id'])
        
        # Recalculate success rate
        summary['success_rate'] = f"{(summary['successful']/summary['total_sentences']*100):.2f}%"
        
        # Write back
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary_path
    
    def save_medical_report(self, filename: str = 'medical_report.json'):
        """
        Save medical report data (chief complaint + history of present illness) to JSON file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if not hasattr(self, 'medical_report_data') or self.medical_report_data is None:
            self.medical_report_data = {
                'chief_complaint': [],
                'history_of_present_illness': [],
                'suggestions': [],
                'diagnosis': [],
                'past_medical_history': [],
                'social_history': []
            }

        if not self.medical_report_data['chief_complaint'] and not self.medical_report_data['history_of_present_illness']:
            print(f"⚠ No sentences met medical report criteria")
            return None
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.medical_report_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved medical report to: {output_path}")
        print(f"  Chief complaint entries: {len(self.medical_report_data['chief_complaint'])}")
        print(f"  History of present illness entries: {len(self.medical_report_data['history_of_present_illness'])}")
        
        return output_path

    def predict_and_save_diagnosis(self, medical_report_filename: str = 'medical_report.json',
                                   output_filename: str = 'diagnosis_prediction.json') -> Optional[str]:
        """
        Generate clinical diagnosis prediction from the medical report using OpenBioLLM
        and save to diagnosis_prediction.json in the output directory.
        """
        try:
            from diagnosis_inference import DiagnosisInference
            medical_report_path = os.path.join(self.output_dir, medical_report_filename)
            if not os.path.exists(medical_report_path):
                if hasattr(self, 'medical_report_data') and self.medical_report_data:
                    report_data = self.medical_report_data
                else:
                    print("⚠ No medical report available for diagnosis prediction")
                    return None
            else:
                with open(medical_report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)

            process_name = os.path.basename(os.path.normpath(self.output_dir))
            print(f"\n🩺 Running AI Diagnosis Prediction (OpenBioLLM-8B) for {process_name}...")
            dx = DiagnosisInference()
            prediction_result = dx.predict(report_data, process_name=process_name)

            output_path = os.path.join(self.output_dir, output_filename)
            dx.save_prediction(prediction_result, output_path)
            print(f"✓ Saved diagnosis prediction to: {output_path}")
            print(f"  Primary Diagnosis: {prediction_result.get('prediction', {}).get('primary_diagnosis')}")
            return output_path
        except Exception as e:
            print(f"Warning: Diagnosis prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run(self, output_filename: str = 'processed_dialogues.json', 
            summary_filename: str = 'processing_summary.json',
            medical_report_filename: str = 'medical_report.json',
            diagnosis_filename: str = 'diagnosis_prediction.json') -> Dict[str, Any]:
        """
        Run complete pipeline with incremental file appending and diagnosis generation
        
        Args:
            output_filename: Name for main JSON output
            summary_filename: Name for summary file
            medical_report_filename: Name for medical report file
            diagnosis_filename: Name for diagnosis prediction file
            
        Returns:
            Dict with paths to output files
        """
        # Process all sentences and append to files incrementally
        results = self.process_all_sentences(output_filename, summary_filename, medical_report_filename)
        
        # Parallel / automatic diagnosis prediction after medical report is ready
        diagnosis_path = self.predict_and_save_diagnosis(medical_report_filename, diagnosis_filename)
        
        # Print final summary from the summary file
        summary_path = os.path.join(self.output_dir, summary_filename)
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print("=" * 80)
            print("FINAL PROCESSING SUMMARY")
            print("=" * 80)
            print(f"Total sentences: {summary['total_sentences']}")
            print(f"Successful: {summary['successful']}")
            print(f"Failed: {summary['failed']}")
            print(f"Success rate: {summary['success_rate']}")
            
            if summary['failed_sentences']:
                print(f"Failed sentences: {summary['failed_sentences']}")
        
        json_path = os.path.join(self.output_dir, output_filename)
        medical_report_path = os.path.join(self.output_dir, medical_report_filename)
        
        return {
            'json_file': json_path,
            'summary_file': summary_path,
            'medical_report_file': medical_report_path,
            'diagnosis_file': diagnosis_path,
            'results': results
        }


def split_multiple_dialogues(file_path: str) -> List[str]:
    """
    Split input file into separate dialogue texts
    Each dialogue is separated by patient introductions or line breaks
    
    Args:
        file_path: Path to dialogue file
        
    Returns:
        List of dialogue strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    # Mode 1: conversational files (e.g., Patient/Doctor transcripts)
    transcript_dialogues = [
        line for line in cleaned_lines
        if 'Doctor:' in line and 'Patient' in line
    ]
    if transcript_dialogues:
        return transcript_dialogues

    # Mode 2: generic text files (one case/sentence per non-empty line)
    return cleaned_lines


def save_dialogue_file(dialogue_text: str, output_path: str):
    """
    Save dialogue text to a .src file
    
    Args:
        dialogue_text: The dialogue content
        output_path: Where to save the file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dialogue_text)


def process_batch_dialogues(source_file: str, output_dir: str = 'pipeline_output_batch'):
    """
    Process multiple dialogues from a source file and create separate output folders for each
    
    Args:
        source_file: Path to dialogue source file
        output_dir: Base output directory for batch results
    """
    source_path = Path(source_file)
    if not source_path.exists():
        print(f"Error: {source_file} not found!")
        return
    
    print("\n" + "="*80)
    print("MULTIPLE DIALOGUES PIPELINE")
    print("="*80 + "\n")
    
    # Split the source file into dialogues
    print(f"Splitting {source_path.name} into separate dialogues...")
    dialogues = split_multiple_dialogues(source_file)
    
    # Check if expected count is in filename (e.g., "dialogues(7).src")
    import re
    expected_match = re.search(r'\((\d+)\)', source_path.name)
    if expected_match:
        expected_count = int(expected_match.group(1))
        if len(dialogues) != expected_count:
            print(f"Warning: Expected {expected_count} dialogues, found {len(dialogues)}")
    
    print(f"Found {len(dialogues)} dialogues\n")
    
    # Process each dialogue
    all_results = {}
    base_output_dir = Path(output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for idx, dialogue_text in enumerate(dialogues, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING DIALOGUE {idx}/{len(dialogues)}")
        print(f"{'='*80}\n")
        
        # Create temporary dialogue file
        temp_dialogue_file = PROJECT_ROOT / f'temp_dialogue_{idx}.src'
        save_dialogue_file(dialogue_text, str(temp_dialogue_file))
        
        # Create output folder for this dialogue
        output_folder = base_output_dir / f'dialogue_{idx}'
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            # Process dialogue through pipeline
            print(f"Processing dialogue {idx}...")
            pipeline = PipelineProcessor(str(temp_dialogue_file), str(output_folder))
            results = pipeline.process_all_sentences(
                output_filename='processed_dialogues.json',
                summary_filename='processing_summary.json',
                medical_report_filename='medical_report.json'
            )
            
            # Predict diagnosis
            dx_file = pipeline.predict_and_save_diagnosis(
                medical_report_filename='medical_report.json',
                output_filename='diagnosis_prediction.json'
            )
            
            # Store results
            all_results[f'dialogue_{idx}'] = {
                'status': 'success',
                'sentence_count': len(results),
                'output_folder': str(output_folder),
                'diagnosis_file': dx_file
            }
            
            # Show summary
            print(f"\n✓ Dialogue {idx} processing complete!")
            print(f"  Folder: {output_folder}")
            print(f"  Sentences processed: {len(results)}")
            
            # List output files
            if output_folder.exists():
                files = os.listdir(output_folder)
                print(f"  Output files: {files}")
            
        except Exception as e:
            print(f"\n✗ Error processing dialogue {idx}: {e}")
            import traceback
            traceback.print_exc()
            all_results[f'dialogue_{idx}'] = {
                'status': 'error',
                'error': str(e)
            }
        
        finally:
            # Clean up temporary file
            if temp_dialogue_file.exists():
                os.remove(temp_dialogue_file)
    
    # Save batch summary
    summary_path = base_output_dir / 'batch_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*80)
    print("BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {base_output_dir}")
    print(f"Structure: {len(dialogues)} folders with 3 files each")
    print(f"\nSummary saved to: {summary_path}\n")
    
    # Print final summary
    for dialogue_key, result in all_results.items():
        status = result['status']
        if status == 'success':
            print(f"✓ {dialogue_key}: {result['sentence_count']} sentences processed")
        else:
            print(f"✗ {dialogue_key}: {result['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline: Process dialogues through NER, DAC, and SLI tasks. Supports both single and multiple dialogues.'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='dialogue.src',
        help='Path to dialogue file (can contain single or multiple dialogues)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results. Auto-named based on input if not specified.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='processed_dialogues.json',
        help='Output JSON filename (default: processed_dialogues.json)'
    )
    parser.add_argument(
        '--summary-file',
        type=str,
        default='processing_summary.json',
        help='Summary filename (default: processing_summary.json)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Force batch mode (process each line as separate dialogue)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    
    # Detect if file contains multiple dialogues
    with open(input_path, 'r', encoding='utf-8') as f:
        dialogues = split_multiple_dialogues(str(input_path))
    
    # If multiple dialogues detected (or batch flag set) and more than 1 dialogue, use batch mode
    if args.batch or len(dialogues) > 1:
        # Auto-generate output dir name based on input filename
        if args.output_dir is None:
            base_name = input_path.stem.replace('.', '_')
            output_dir = PROJECT_ROOT / f'pipeline_output_{base_name}'
        else:
            output_dir = args.output_dir
            if not Path(output_dir).is_absolute():
                output_dir = PROJECT_ROOT / output_dir
        
        process_batch_dialogues(str(input_path), str(output_dir))
    
    else:
        # Single dialogue - use regular pipeline
        if args.output_dir is None:
            output_dir = PROJECT_ROOT / 'pipeline_output'
        else:
            output_dir = args.output_dir
            if not Path(output_dir).is_absolute():
                output_dir = PROJECT_ROOT / output_dir
        
        pipeline = PipelineProcessor(str(input_path), str(output_dir))
        outputs = pipeline.run(args.output_file, args.summary_file)
        
        print(f"\n✓ Pipeline complete!")
        print(f"  Main output: {outputs['json_file']}")
        print(f"  Summary: {outputs['summary_file']}")
        if outputs['medical_report_file']:
            print(f"  Medical report: {outputs['medical_report_file']}")


if __name__ == '__main__':
    main()
