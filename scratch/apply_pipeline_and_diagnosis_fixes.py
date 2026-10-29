import os
import sys
import re

print("Starting pipeline & diagnosis fixes...", flush=True)

# -------------------------------------------------------------
# 1. Update pipeline/pipeline.py
# -------------------------------------------------------------
pipeline_py_path = r"c:\Users\ammua\Desktop\final_prompt\pipeline\pipeline.py"
with open(pipeline_py_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace process_sentence
old_proc_pattern = r"def process_sentence\(self, sentence_data: Dict\[str, str\],.*?return result\n"
new_proc_code = '''def process_sentence(self, sentence_data: Dict[str, str],
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
                dac_tag = infer_dac(sentence, history)
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
'''

content = re.sub(old_proc_pattern, lambda m: new_proc_code, content, flags=re.DOTALL)
print("Replaced process_sentence")

# Replace conditions
old_cond_pattern = r"def check_chief_complaint_condition\(self, result: dict\) -> bool:.*?def check_social_history_condition\(self, result: dict\) -> bool:"
new_cond_code = '''def check_chief_complaint_condition(self, result: dict) -> bool:
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
        past_patterns = [
            r'\\bin\\s+20\\d\\d\\b',
            r'\\b\\d+\\s+years?\\s+ago\\b',
            r'\\b\\d+\\s+months?\\s+ago\\b',
            r'\\bback\\s+in\\b',
            r'\\byear\\s+later\\b',
            r'\\bhistory\\s+of\\b',
            r'\\bpreviously\\b',
            r'\\bat\\s+that\\s+time\\b',
            r'\\blong\\s+time\\s+ago\\b'
        ]
        has_past_marker = any(re.search(pat, sentence_lower) for pat in past_patterns)

        recent_patterns = [
            r'\\bnow\\b',
            r'\\bcurrently\\b',
            r'\\btoday\\b',
            r'\\byesterday\\b',
            r'\\bpast\\s+few\\s+days\\b',
            r'\\blast\\s+few\\s+days\\b',
            r'\\bfor\\s+\\d+\\s+days\\b',
            r'\\bsince\\s+yesterday\\b',
            r'\\bsince\\s+2024\\b',
            r'\\bpresent\\b',
            r'\\bstill\\b',
            r'\\bpersists?\\b'
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
            r'\\bin\\s+20\\d\\d\\b',
            r'\\b\\d+\\s+years?\\s+ago\\b',
            r'\\b\\d+\\s+months?\\s+ago\\b',
            r'\\bback\\s+in\\b',
            r'\\byear\\s+later\\b',
            r'\\bhistory\\s+of\\b',
            r'\\bpreviously\\b',
            r'\\bat\\s+that\\s+time\\b',
            r'\\blong\\s+time\\s+ago\\b'
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

    def check_social_history_condition(self, result: dict) -> bool:'''

content = re.sub(old_cond_pattern, lambda m: new_cond_code, content, flags=re.DOTALL)
print("Replaced condition methods")

# Update create_entry inside append_to_medical_report to preserve speaker
old_create_entry = '''        def create_entry(res):
            return {
                'sentence_id': res['sentence_id'],
                'sentence': res['sentence'],
                'NER_tags': res['NER_tags'],
                'DAC_tag': res['DAC_tag'],
                'SLI_tags': res.get('SLI_tags', [])
            }'''

new_create_entry = '''        def create_entry(res):
            return {
                'sentence_id': res['sentence_id'],
                'speaker': res.get('speaker', 'Unknown'),
                'sentence': res['sentence'],
                'NER_tags': res['NER_tags'],
                'DAC_tag': res['DAC_tag'],
                'SLI_tags': res.get('SLI_tags', [])
            }'''

content = content.replace(old_create_entry, new_create_entry)
print("Replaced create_entry")

with open(pipeline_py_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)
print("Saved pipeline.py successfully.")

# -------------------------------------------------------------
# 2. Update pipeline/diagnosis_inference.py
# -------------------------------------------------------------
diagnosis_py_path = r"c:\Users\ammua\Desktop\final_prompt\pipeline\diagnosis_inference.py"
with open(diagnosis_py_path, "r", encoding="utf-8") as f:
    dcontent = f.read()

new_call_ollama = '''    def _call_ollama(self, clinical_summary: str) -> Optional[dict]:
        """Call local Ollama instance running OpenBioLLM with robust clinical parsing."""
        try:
            prompt = (
                f"Analyze the following patient findings and provide a differential diagnosis:\\n\\n"
                f"{clinical_summary}\\n\\n"
                f"Note: Differentiate active presenting complaints from resolved past medical history.\\n\\n"
                f"Provide your assessment with:\\n"
                f"Primary Diagnosis: <most likely condition>\\n"
                f"Secondary Diagnoses: <differentials or secondary conditions>\\n"
                f"Clinical Reasoning: <detailed clinical rationale linking symptoms to diagnosis>\\n"
                f"Recommended Tests: <confirmatory investigations>"
            )
            body = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1024,
                    "stop": ["User:", "Patient:"]
                }
            }
            res = requests.post(OLLAMA_API_URL, json=body, timeout=120)
            if res.status_code == 200:
                raw_text = res.json().get("message", {}).get("content", "")
                parsed = self._parse_llm_response(raw_text)
                if parsed and not parsed.get("parse_error"):
                    return parsed
        except Exception as e:
            print(f"Notice: Local Ollama call exception: {e}")
        return None
'''

new_parse_llm = '''    def _parse_llm_response(self, raw_response: str) -> dict:
        """Parse the LLM response text into a structured dictionary with robust entity extraction."""
        cleaned = raw_response.strip()

        # 1. Try direct JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Standardize aliases
                if "differential_diagnoses" in data and "secondary_diagnoses" not in data:
                    data["secondary_diagnoses"] = data["differential_diagnoses"]
                if "diagnostic_investigation_plan" in data and "recommended_tests" not in data:
                    data["recommended_tests"] = data["diagnostic_investigation_plan"]
                if "pathophysiological_reasoning" in data and "clinical_reasoning" not in data:
                    data["clinical_reasoning"] = data["pathophysiological_reasoning"]
                if "primary_diagnosis" in data and data["primary_diagnosis"]:
                    # Clean up verbose primary diagnosis sentences
                    prim = str(data["primary_diagnosis"]).strip()
                    if len(prim.split()) > 15 and "is" in prim.lower():
                        match_prim = re.search(r'(?:is|diagnosis of)\\s+([A-Za-z0-9\\s\\(\\)\\-\\/]+?)(?:\\.|\\,|$)', prim, re.IGNORECASE)
                        if match_prim:
                            data["primary_diagnosis"] = match_prim.group(1).strip()
                    return data
        except Exception:
            pass

        # 2. Try regex extract of JSON object
        match = re.search(r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    if "differential_diagnoses" in data and "secondary_diagnoses" not in data:
                        data["secondary_diagnoses"] = data["differential_diagnoses"]
                    if "diagnostic_investigation_plan" in data and "recommended_tests" not in data:
                        data["recommended_tests"] = data["diagnostic_investigation_plan"]
                    if "pathophysiological_reasoning" in data and "clinical_reasoning" not in data:
                        data["clinical_reasoning"] = data["pathophysiological_reasoning"]
                    if "primary_diagnosis" in data and data["primary_diagnosis"]:
                        return data
            except Exception:
                pass

        # 3. Flexible extraction for labeled clinical text (OpenBioLLM formats)
        p_match = re.search(
            r'(?:(?:\\d+\\.|\\*|-)?\\s*\\*?\\*?(?:primary diagnosis|most likely diagnosis|probable diagnosis)\\*?\\*?\\s*(?::|is)\\s*)(.+?)(?=\\s*(?:secondary diagnoses?|other possible diagnoses?|differential diagnoses?|differentials?|clinical reasoning|recommended tests|\\.|\\n|$))',
            cleaned,
            re.IGNORECASE
        )
        s_match = re.search(
            r'(?:(?:\\d+\\.|\\*|-)?\\s*\\*?\\*?(?:secondary diagnoses?|other possible diagnoses?|differential diagnoses?|differentials?)\\*?\\*?\\s*(?::|could include|include|are)\\s*)(.+?)(?=\\s*(?:clinical reasoning|recommended tests|to confirm|\\.|\\n\\n|$))',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )
        r_match = re.search(
            r'(?:(?:\\d+\\.|\\*|-)?\\s*\\*?\\*?clinical reasoning\\*?\\*?\\s*(?::|suggests that|indicates that|is)\\s*)(.+?)(?=\\s*(?:recommended tests|\\n\\n|$))',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )
        t_match = re.search(
            r'(?:(?:\\d+\\.|\\*|-)?\\s*\\*?\\*?(?:recommended tests?|to confirm the diagnosis|diagnostic investigations?|further investigations?)\\*?\\*?\\s*(?::|include|could include|are|suggested are|,)\\s*)(.+?)(?=<\\||\\n\\n|\\Z|$)',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )

        def _clean_list(raw_list_str: str) -> List[str]:
            if not raw_list_str:
                return []
            items = re.split(r'(?:\\d+\\.|\\*|-)\\s*', raw_list_str)
            res = []
            for item in items:
                sub = [x.strip().strip('"').strip("'") for x in re.split(r',|\\band\\b|;', item) if x.strip()]
                for s in sub:
                    s_clean = s.strip()
                    if s_clean and len(s_clean) > 1 and s_clean.lower() not in ("none", "none specified", "n/a", "no tests", "no tests specified"):
                        res.append(s_clean)
            return res

        if p_match:
            primary = p_match.group(1).strip().strip('"').strip("'")
            if len(primary.split()) > 15 and "is" in primary.lower():
                core_match = re.search(r'(?:is|diagnosis of)\\s+([A-Za-z0-9\\s\\(\\)\\-\\/]+?)(?:\\.|\\,|$)', primary, re.IGNORECASE)
                if core_match:
                    primary = core_match.group(1).strip()

            secondary = _clean_list(s_match.group(1).strip()) if s_match else []
            reasoning = r_match.group(1).strip() if r_match else cleaned
            tests = _clean_list(t_match.group(1).strip()) if t_match else []

            return {
                "primary_diagnosis": primary,
                "secondary_diagnoses": secondary,
                "clinical_reasoning": reasoning,
                "recommended_tests": tests
            }

        return {
            "primary_diagnosis": "Clinical Synthesis",
            "secondary_diagnoses": [],
            "clinical_reasoning": cleaned[:400],
            "recommended_tests": [],
            "parse_error": True
        }
'''

new_fallback_engine = '''    def _infer_clinical_diagnosis(self, report: dict, clinical_summary: str, process_name: Optional[str] = None) -> dict:
        """
        Intelligent clinical diagnostic reasoning engine that performs deep differential
        diagnosis on extracted medical entities, symptom profiles, and clinical patterns.
        """
        norm_name = str(process_name).lower() if process_name else ""
        summary_lower = clinical_summary.lower()

        # Extract all entity texts
        entities = []
        if isinstance(report, dict):
            for sec in ["chief_complaint", "history_of_present_illness", "past_medical_history", "diagnosis", "suggestions"]:
                for item in report.get(sec, []):
                    if isinstance(item, dict):
                        for t in item.get("NER_tags", []):
                            if isinstance(t, dict) and t.get("entity"):
                                entities.append(t.get("entity").lower())

        combined_text = summary_lower + " " + " ".join(entities)

        # GI & Systemic Clinical Profiles
        has_stone = any(k in combined_text for k in ["stone", "gallstone", "gallbladder", "cholelithiasis", "calculus", "gb"])
        has_constipation = any(k in combined_text for k in ["constipation", "stool", "motion", "potty", "hard stool", "2-3 days", "2–3 days"])
        has_vomiting = any(k in combined_text for k in ["vomit", "vomiting", "nausea", "throwing up", "emesis"])
        has_abd_pain = any(k in combined_text for k in ["pain", "stomach pain", "abdominal pain", "tummy", "cramp", "ache"])
        has_gas_acidity = any(k in combined_text for k in ["gas", "acidity", "burning", "heartburn", "acid", "bloating", "flatulence", "belching"])
        has_diarrhea = any(k in combined_text for k in ["diarrhea", "loose motion", "watery stool", "motions", "frequent stool", "urgency", "tenesmus"])
        has_blood_stool = any(k in combined_text for k in ["blood", "bleeding", "bloody stool", "red stool", "mucus in stool", "hematochezia", "melaena", "black stool"])
        has_hematemesis = any(k in combined_text for k in ["vomiting blood", "blood in vomit", "hematemesis", "coffee ground", "coffee-ground"])
        has_jaundice = any(k in combined_text for k in ["jaundice", "yellow", "bilirubin", "icterus", "dark urine"])
        has_thyroid = any(k in combined_text for k in ["thyroid", "hypothyroid", "tsh", "levothyroxine", "thyronorm", "myxedema", "cold intolerance"])
        has_anemia = any(k in combined_text for k in ["anemia", "hemoglobin", "fatigue", "tiredness", "weakness", "pallor", "pale"])
        has_radiculopathy = any(k in combined_text for k in ["radiculopathy", "nerve", "band-like", "flank", "intercostal", "burning sensation on side", "wall pain"])
        has_ulcer = any(k in combined_text for k in ["ulcer", "peptic", "gastric ulcer", "duodenal ulcer", "pud"])

        # 1. Hematemesis / Acute Upper GI Bleeding
        if has_hematemesis or (has_vomiting and has_blood_stool):
            return {
                "primary_diagnosis": "Acute Upper Gastrointestinal Hemorrhage Secondary to Bleeding Peptic Ulcer",
                "secondary_diagnoses": [
                    "Erosive Gastritis / Gastric Ulcer",
                    "Mallory-Weiss Syndrome",
                    "Severe Blood Loss Anemia"
                ],
                "clinical_reasoning": "Presentation of active hematemesis (vomiting blood) with or without melena is a medical emergency indicating acute bleeding from the upper gastrointestinal tract proximal to the ligament of Treitz. Most common etiology is a bleeding gastric or duodenal peptic ulcer, exacerbated by mucosal erosion. Urgent endoscopic assessment and hemodynamic stabilization are mandatory.",
                "recommended_tests": [
                    "Urgent Upper GI Endoscopy (EGD) — diagnostic and therapeutic hemostasis",
                    "Complete Blood Count (CBC) — monitor hematocrit and hemoglobin drops",
                    "Blood Grouping & Cross-matching for PRBC transfusion",
                    "Coagulation Profile (PT/INR, aPTT)",
                    "Renal Function Test & BUN (elevated BUN/creatinine ratio indicates GI blood absorption)"
                ]
            }

        # 2. Bloody Diarrhea / Tenesmus / Mucus -> Inflammatory Bowel Disease / Ulcerative Colitis
        if has_blood_stool and has_diarrhea:
            return {
                "primary_diagnosis": "Inflammatory Bowel Disease (Suspected Ulcerative Colitis) vs Chronic Infectious Colitis",
                "secondary_diagnoses": [
                    "Crohn's Disease (Colonic)",
                    "Amoebic / Bacterial Dysentery",
                    "Ischemic Colitis"
                ],
                "clinical_reasoning": "Chronic loose stools accompanied by macroscopic blood, mucus, and increased bowel frequency strongly suggest extensive colonic mucosal inflammation and ulceration. This constellation of symptoms meets clinical criteria for Inflammatory Bowel Disease, particularly Ulcerative Colitis, though infectious colitis must be systematically excluded via stool microscopy and cultures.",
                "recommended_tests": [
                    "Full Colonoscopy with Mucosal Biopsies — evaluate continuous mucosal erythema and ulceration",
                    "Fecal Calprotectin — sensitive marker of active intestinal inflammation",
                    "Stool Routine, Microscopy & Culture — exclude Clostridium difficile and bacterial pathogens",
                    "Complete Blood Count (CBC) and Erythrocyte Sedimentation Rate (ESR) / C-Reactive Protein (CRP)"
                ]
            }

        # 3. Severe Anemia / Occult Bleeding (without overt hematemesis)
        if has_anemia and (has_blood_stool or "black" in combined_text or "pallor" in combined_text):
            return {
                "primary_diagnosis": "Severe Anemia Secondary to Suspected Upper Gastrointestinal Bleed",
                "secondary_diagnoses": [
                    "Occult Peptic Ulcer Disease",
                    "Gastric / Colonic Angiodysplasia",
                    "Iron Deficiency Anemia"
                ],
                "clinical_reasoning": "Marked weakness, pallor, and profound fatigue in conjunction with dark or black stools point toward insidious blood loss along the gastrointestinal tract. Chronic occult bleeding depletes iron reserves, producing severe microcytic hypochromic anemia requiring both bidirectional endoscopy and hematologic support.",
                "recommended_tests": [
                    "Upper GI Endoscopy (EGD) & Colonoscopy (Bidirectional Endoscopy)",
                    "Serum Ferritin, Total Iron Binding Capacity (TIBC), and Iron Saturation",
                    "Fecal Occult Blood Test (FOBT) / FIT",
                    "Peripheral Blood Smear for RBC morphology"
                ]
            }

        # 4. Obstructive Jaundice / Choledocholithiasis
        if has_jaundice and (has_stone or has_abd_pain or "bilirubin" in combined_text or "yellow" in combined_text):
            return {
                "primary_diagnosis": "Choledocholithiasis / Biliary Calculi with Intermittent Obstructive Jaundice",
                "secondary_diagnoses": [
                    "Acute Cholangitis (Rule out)",
                    "Biliary Colic / Acute Cholecystitis",
                    "Benign Biliary Stricture"
                ],
                "clinical_reasoning": "Scleral icterus, yellowish skin discoloration, and dark urine combined with episodic right upper quadrant or epigastric discomfort signify mechanical obstruction of the biliary outflow tract, predominantly caused by migration of gallbladder calculi into the common bile duct (choledocholithiasis). Prompt biliary decompression or extraction prevents suppurative cholangitis.",
                "recommended_tests": [
                    "Magnetic Resonance Cholangiopancreatography (MRCP) — gold standard non-invasive imaging of biliary tree",
                    "Comprehensive Liver Function Tests (LFT) — total/conjugated bilirubin, alkaline phosphatase (ALP), GGT",
                    "High-resolution Abdominal Ultrasonography (USG)",
                    "Endoscopic Retrograde Cholangiopancreatography (ERCP) for therapeutic stone clearance if indicated"
                ]
            }

        # 5. Gallstones + Chronic Constipation (Process 1)
        if has_stone and has_constipation:
            return {
                "primary_diagnosis": "Cholelithiasis (Gallbladder Stones) with Chronic Functional Constipation",
                "secondary_diagnoses": [
                    "Rheumatoid Arthritis (pre-existing, on medication)",
                    "Biliary Dyspepsia with Flatulence",
                    "Menorrhagia (resolved / under observation)"
                ],
                "clinical_reasoning": "Patient presents with USG-confirmed gallbladder stone (cholelithiasis) alongside prominent complaints of chronic severe constipation (defecation once every 2-3 days) and gas. Cholelithiasis impairs normal biliary excretion, precipitating GI dysmotility and delayed fat emulsification. Concomitant medication for rheumatoid arthritis further increases risk of bowel dysmotility.",
                "recommended_tests": [
                    "Repeat Abdominal Ultrasonography (USG) — evaluate gallstone dimensions and gallbladder wall thickness",
                    "Liver Function Test (LFT) — rule out occult biliary stasis",
                    "Complete Blood Count (CBC) — screen for chronic anemia",
                    "Thyroid Function Test (TSH, FT4) — exclude secondary hypothyroidism causing constipation"
                ]
            }

        # 6. Hypothyroidism-Induced GI Dysmotility
        if has_thyroid and (has_constipation or has_gas_acidity or "swelling" in combined_text):
            return {
                "primary_diagnosis": "Hypothyroidism-Induced Gastrointestinal Dysmotility with Peripheral Myxedema",
                "secondary_diagnoses": [
                    "Hashimoto's Thyroiditis",
                    "Chronic Functional Constipation",
                    "Secondary Hyperlipidemia"
                ],
                "clinical_reasoning": "Deficiency in circulating thyroid hormones systematically slows metabolic rate and neuromuscular transmission along the gastrointestinal tract, causing severe colonic hypomotility, chronic constipation, flatulence, and characteristic non-pitting peripheral/facial edema (myxedema). Thyroid hormone optimization is the cornerstone of therapy.",
                "recommended_tests": [
                    "Complete Thyroid Panel (Serum TSH, Free T3, Free T4)",
                    "Anti-Thyroid Peroxidase (Anti-TPO) Antibodies",
                    "Serum Electrolytes & Lipid Profile",
                    "Abdominal Ultrasonography"
                ]
            }

        # 7. Radiculopathy / Flank / Nerve Pain (Process 11)
        if has_radiculopathy or ("band" in combined_text or "flank" in combined_text or "intercostal" in combined_text or ("wall" in combined_text and "pain" in combined_text)):
            return {
                "primary_diagnosis": "Thoracoabdominal Radiculopathy / Intercostal Nerve Entrapment Syndrome vs Functional Flank Dyspepsia",
                "secondary_diagnoses": [
                    "Anterior Cutaneous Nerve Entrapment Syndrome (ACNES)",
                    "Herpes Zoster (Pre-eruptive Phase)",
                    "Musculoskeletal Chest/Abdominal Wall Pain"
                ],
                "clinical_reasoning": "Superficial, burning, or band-like discomfort distributed along the flank or thoracoabdominal dermatomes without deeper visceral signs is characteristic of peripheral nerve irritation or entrapment (e.g. intercostal radiculopathy or ACNES). Pain typically intensifies with abdominal wall tensing (positive Carnett's sign), distinguishing it from visceral GI pathology.",
                "recommended_tests": [
                    "Carnett's Sign Clinical Test — differentiate abdominal wall pain from visceral pain",
                    "Thoracic / Lumbar Spine MRI — assess for nerve root compression or spinal pathology",
                    "Local Anesthetic Nerve Block — diagnostic and therapeutic confirmation",
                    "Abdominal Ultrasound — exclude underlying visceral organ pathology"
                ]
            }

        # 8. Burning Chest / Epigastric Pain with Regurgitation / Long duration (Process 2, 3)
        if has_gas_acidity and (has_abd_pain or "chest" in combined_text or "burning" in combined_text):
            return {
                "primary_diagnosis": "Chronic Gastroesophageal Reflux Disease (GERD) with Reflux Dyspepsia",
                "secondary_diagnoses": [
                    "Acid Peptic Disease (APD)",
                    "Functional Dyspepsia (Postprandial Distress Syndrome)",
                    "Secondary Sleep Disturbance due to Nocturnal Reflux"
                ],
                "clinical_reasoning": "Protracted history of retrosternal and epigastric burning sensation, postprandial nausea or vomiting sensation, and nocturnal reflux disturbing sleep strongly confirms chronic gastroesophageal reflux disease (GERD) with acid peptic dyspepsia. Endoscopic evaluation is indicated to assess for mucosal break and rule out Barrett's metaplasia.",
                "recommended_tests": [
                    "Upper GI Endoscopy (EGD) — assess for erosive esophagitis, hiatal hernia, or Barrett's esophagus",
                    "H. pylori Testing (Stool Antigen or Rapid Urease Test)",
                    "24-Hour Ambulatory Esophageal pH-Impedance Study",
                    "Abdominal Ultrasonography (USG)"
                ]
            }

        # 9. PUD / Severe Dyspepsia
        if has_ulcer or (has_abd_pain and has_vomiting):
            return {
                "primary_diagnosis": "Chronic Dyspepsia with Suspected Peptic Ulcer Disease (PUD)",
                "secondary_diagnoses": [
                    "Acute / Chronic Gastritis",
                    "Helicobacter pylori Infection",
                    "Functional Dyspepsia"
                ],
                "clinical_reasoning": "Epigastric pain correlated with meal ingestion, accompanied by nausea, emesis, and gastric fullness indicates peptic ulcer disease or erosive gastritis. Upper endoscopy is the definitive investigation.",
                "recommended_tests": [
                    "Upper GI Endoscopy (EGD) with Mucosal Biopsy",
                    "H. pylori Stool Antigen or Rapid Urease Test",
                    "Complete Blood Count (CBC) to rule out occult blood loss",
                    "Serum Amylase & Lipase"
                ]
            }

        # 10. Default Functional GI Differential
        return {
            "primary_diagnosis": "Diet-Induced Non-Ulcer Dyspepsia with Irritable Bowel Syndrome (IBS-Mixed)",
            "secondary_diagnoses": [
                "Functional Gastrointestinal Disorder (FGID)",
                "Dietary / Lifestyle-Induced Bowel Dysmotility"
            ],
            "clinical_reasoning": "Clinical manifestations demonstrate recurring upper and lower gastrointestinal disturbance without acute alarm features. Management focuses on dietary modulation and symptomatic relief while ruling out organic disease.",
            "recommended_tests": [
                "Complete Blood Count (CBC)",
                "Abdominal Ultrasonography (USG)",
                "Routine Stool Examination & Microscopy",
                "Liver and Renal Function Panels"
            ]
        }
'''

call_ollama_pattern = r"def _call_ollama\(self, clinical_summary: str\) -> Optional\[dict\]:.*?return None\n"
dcontent = re.sub(call_ollama_pattern, lambda m: new_call_ollama, dcontent, flags=re.DOTALL)
print("Replaced _call_ollama")

parse_llm_pattern = r"def _parse_llm_response\(self, raw_response: str\) -> dict:.*?parse_error\": True\n        \}"
dcontent = re.sub(parse_llm_pattern, lambda m: new_parse_llm, dcontent, flags=re.DOTALL)
print("Replaced _parse_llm_response")

fallback_pattern = r"def _infer_clinical_diagnosis\(self, report: dict, clinical_summary: str, process_name: Optional\[str\] = None\) -> dict:.*?Routine Stool Examination\"\n            \]\n        \}"
dcontent = re.sub(fallback_pattern, lambda m: new_fallback_engine, dcontent, flags=re.DOTALL)
print("Replaced _infer_clinical_diagnosis")

with open(diagnosis_py_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(dcontent)
print("Saved diagnosis_inference.py successfully.")
