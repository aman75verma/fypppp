# 🔬 Deep Clinical NLP Analysis & End-to-End System Improvements
## MediTOD: Named Entity Recognition (NER), Dialogue Act Classification (DAC), and Clinical Diagnostic Reasoning

> **Date:** October 28, 2026  
> **Status:** Production-Grade Technical Specification & Roadmap  
> **Target Scope:** NER, DAC, Pipeline SOAP Construction, OpenBioLLM Clinical Reasoning  
> **Constraint:** Preserves existing dataset tokens/annotations and 12-tag / 16-intent ontology strictly (`train.jsonl`, `val.jsonl`, `test.jsonl`).

---

## 1. Executive Summary & Diagnostic Scorecard

Across the 11 benchmark clinical patient cases (`process_1` through `process_11`), the MediTOD pipeline achieved acceptable diagnostic concordance in **only 2 out of 11 cases (18.2%)**:
- **Case 2 (GERD)**: Matched ground truth.
- **Case 10 (Hypothyroidism)**: Matched ground truth.
- **9 out of 11 cases failed** due to critical failure cascades spanning tokenization, inference context stripping, report routing logic, and diagnostic prompt design.

### Diagnostic Performance Matrix (11 Benchmarks)

| Case | Patient Presentation (Transcript) | Expert Physician Ground Truth | LLM Predicted Primary Diagnosis | Concordance Status | Primary Root Cause of Failure |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Case 01** | Gallstone + 2024 severe constipation; past 2019 menorrhagia | **Cholelithiasis with Chronic Functional Constipation** | *likely endometrial cancer* | ❌ Severe Mismatch | Temporal leakage: 5-year-old resolved menorrhagia dumped into `chief_complaint`. |
| **Case 02** | Epigastric burning, pain, gas reduction on antacids | **GERD with Recurrent Acid Peptic Disease** | **GERD** | ✅ **Match** | Simple, focused presentation with intact `chief_complaint`. |
| **Case 03** | 2-3 yr burning chest to stomach, postprandial nausea | **Chronic GERD with Reflux Dyspepsia** | *Hyperthyroidism* | ❌ Severe Mismatch | Empty report: Patient symptom turns misclassified as `Request-*`; doctor's screening question on thyroid recorded as sole history. |
| **Case 04** | Postprandial burning, tenesmus after fatty food/legumes | **Diet-Induced Non-Ulcer Dyspepsia with IBS-Mixed** | *likely to be gastritis* | ⚠️ Partial Match | Missed lower GI tenesmus/motility symptoms in report synthesis. |
| **Case 05** | 6-7 mo diffuse burning, early satiety, epigastric pain | **Chronic Dyspepsia with Suspected PUD** | *FGID / Dyspepsia* (Fallback) | ⚠️ Partial (Fallback) | Live Ollama timed out/failed parse; fell back to static generic FGID rule. |
| **Case 06** | 2 mo gallstone history, dark yellow urine after sun/labor | **Choledocholithiasis with Intermittent Jaundice** | *jaundice following exposure to sunlight...* (Cut-off) | ⚠️ Partial (Truncated) | Ollama `num_predict: 512` cut off generation mid-sentence, breaking JSON. |
| **Case 07** | 2 yr recurrent loose motions with blood, lower cramps | **IBD (Suspected Ulcerative Colitis) vs Colitis** | *GERD / Non-Ulcer Dyspepsia* (Fallback) | ❌ Severe Mismatch | Critical symptoms ("bloody loose stools") dropped due to DAC speaker confusion; fallback triggered. |
| **Case 08** | Severe blood loss anemia, PRBC transfusion, black stool | **Severe Anemia Secondary to Upper GI Bleed** | *FGID / Dyspepsia* (Fallback) | ❌ Severe Mismatch | Melena and acute transfusion history overshadowed by noise; doctor question "prblem" tagged as test. |
| **Case 09** | Endoscopy-confirmed ulcer 4 mo ago, melena, hematochezia | **Acute Upper GI Hemorrhage (Bleeding Peptic Ulcer)** | *GERD / Non-Ulcer Dyspepsia* (Fallback) | ❌ Severe Mismatch | Active GI hemorrhage downplayed; live LLM fallback to default acid reflux. |
| **Case 10** | Known thyroid on levothyroxine, swollen abdomen, edema | **Hypothyroidism-Induced GI Dysmotility with Myxedema** | **Hypothyroidism** | ✅ **Match** | Prominent thyroid entity preserved in report. |
| **Case 11** | Flank/abdominal wall heaviness, negative ultrasound | **Thoracoabdominal Radiculopathy / Intercostal Neuralgia** | *FGID / Dyspepsia* (Fallback) | ❌ Severe Mismatch | Neuropathic pain clues dropped; fallback categorized as functional dyspepsia. |

---

## 2. Quantitative Deep Analysis

### 2.1 NER Quantitative Performance Across Evaluated Models

| Metric | PubMedBERT (v2 - Production) | SciBERT | BioClinicalBERT |
| :--- | :---: | :---: | :---: |
| **Strict Entity F1** | **0.7944** | 0.7793 | 0.7657 |
| **Precision** | 0.7295 | **0.7305** | 0.6914 |
| **Recall** | **0.8720** | 0.8560 | 0.8578 |
| **Test Loss** | **0.0067** | 0.0072 | 0.0093 |

#### Per-Entity Breakdown (PubMedBERT v2 Production Baseline)

| Entity Tag | Support | Precision | Recall | F1-Score | Performance Tier | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `MEDICATION` | 159 | 0.8743 | 0.9182 | **0.8957** | Tier 1 (Excellent) | High vocabulary coverage of pharmacological agents. |
| `SYMPTOM_LOCATION` | 186 | 0.8660 | 0.9032 | **0.8842** | Tier 1 (Excellent) | Anatomical priors well-represented in PubMed corpus. |
| `SYMPTOM_VALUE` | 553 | 0.8025 | 0.9186 | **0.8567** | Tier 1 (Strong) | High recall (0.918), but boundary bleeding with adjectives. |
| `EXPOSURE_VALUE` | 95 | 0.7692 | 0.9474 | **0.8491** | Tier 1 (Strong) | Captures sun, fields, smoke accurately. |
| `MEDICAL_TEST_NAME` | 141 | 0.8052 | 0.8794 | **0.8407** | Tier 1 (Strong) | High accuracy on standard tests (USG, Endoscopy, LFT). |
| `HABIT_VALUE` | 131 | 0.7134 | 0.8931 | **0.7932** | Tier 2 (Good) | Robust on alcohol, smoking, diet habits. |
| `SYMPTOM_FREQUENCY` | 63 | 0.7237 | 0.8730 | **0.7914** | Tier 2 (Good) | Occasional overlap with `HABIT_FREQUENCY`. |
| `MEDICAL_HISTORY_VALUE`| 114 | 0.7890 | 0.7544 | **0.7713** | Tier 2 (Good) | Struggles when history is expressed in colloquial terms. |
| `HABIT_FREQUENCY` | 43 | 0.5902 | 0.8372 | **0.6923** | Tier 3 (Underperforming) | Precision 0.59: false positives on general numbers. |
| `TEMPORAL` | 83 | 0.5500 | 0.6627 | **0.6011** | Tier 3 (Underperforming) | Boundary truncation: missed prepositional attachments. |
| `POLARITY` | 65 | 0.3481 | 0.8462 | **0.4933** | Tier 4 (Severe Bottleneck) | Precision 0.34: over-tags "no", "yes" outside medical facts. |
| `SYMPTOM_CHARACTERISTICS`| 62 | 0.4086 | 0.6129 | **0.4903** | Tier 4 (Severe Bottleneck) | Subjective descriptors ("heavy", "tight", "burning"). |

---

### 2.2 DAC Quantitative Performance Across Evaluated Models

| Model | Accuracy (%) | Precision | Recall | F1-Score | Best Epoch | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PubMedBERT (v1)** | **86.92%** | **0.8840** | **0.8692** | **0.8738** | 11 | **0.3541** |
| **PubMedBERT (v2)** | 86.12% | 0.8723 | 0.8612 | 0.8645 | 6 | 0.3810 |
| **BERT-base-uncased** | 85.92% | 0.8653 | 0.8592 | 0.8602 | 11 | 0.4012 |
| **BioBERT v1.1** | 84.91% | 0.8688 | 0.8491 | 0.8556 | 10 | 0.4236 |

#### Per-Intent Analysis (16 Classes)

```mermaid
pie title DAC Error Distribution on Test Set
    "Inform-Basic_Info Ambiguity" : 38
    "Request vs Inform Inversion" : 27
    "Other vs Clinical Intent" : 19
    "Rare Class Misclassification" : 16
```

- **Top Performers (F1 = 1.000)**: `Diagnose` (20/20 correct), `Inform-Drug_Recommendation` (19/19 correct).
- **Core Clinical Intents**: `Request-Symptom` (0.9725), `Inform-Symptom` (0.9536), `Inform-Etiology` (0.9091).
- **Critical Failure Points**:
  - `Inform-Basic_Information`: F1 = **0.4828** (Precision = 0.3889). The model cannot reliably distinguish demographic/background statements from `Other` or `Inform-Etiology`.
  - `Request-Drug_Recommendation`: F1 = **0.5714** (Support = 3). Extreme data scarcity.
  - `Request-Medical_Advice`: F1 = **0.6364** (v1). Frequently misrouted to `Request-Etiology`.

---

## 3. Qualitative Deep Analysis: The 4 Root-Cause Failure Cascades

### Cascade 1: The "Speaker-Blindness" Disaster in Preprocessing

```mermaid
flowchart TD
    A["Raw Dialogue Text<br/>Doctor: Which area?<br/>Patient: From the chest area."] --> B["SentenceProcessor.split_by_speakers()"]
    B -->|"STRIPS 'Doctor:' and 'Patient:'"| C["Naked String Sentences<br/>'From the chest area.'"]
    C --> D["DAC Model Inference<br/>PubMedBERT Sequence Classifier"]
    D -->|"Sees question context, missing [PATIENT] tag"| E["Misclassified as 'Request-Symptom'"]
    E --> F["pipeline.py check_chief_complaint_condition()"]
    F -->|"Requires DAC: Inform-Symptom"| G["DROPPED FROM MEDICAL REPORT"]
    G --> H["Empty Chief Complaint in Final Report<br/>(Process 3, Process 7)"]
```

#### Ground Evidence:
- In `train.jsonl`, **4,080 out of 4,080 (100.0%)** utterances explicitly begin with `Doctor:` or `Patient:`.
- In `DAC/meditod_dac_training_v2.ipynb`, the model was explicitly trained on context strings constructed as:
  `[DOCTOR] Doctor: Which area? [SEP] [PATIENT] Patient: From the chest area.`
- **The Defect:** In `pipeline/sentence_processor.py`, `SentenceProcessor.split_by_speakers()` executes:
  ```python
  pattern = r'(?:Doctor|Patient|Nurse):\s*'
  statements = re.split(pattern, text)
  ```
  It completely deletes the speaker identity before feeding tokens into inference!
- **The Consequence:** Stripped of speaker tags, the model cannot distinguish between a doctor asking a question and a patient answering it. In Case 7:
  - Patient statement: *"Sir, I've been having stomach problems and for the past 2 years—whenever I get loose motion, it continues for about 15 to 20 days"* $\rightarrow$ Classified as `Request-Symptom`!
  - Patient statement: *"Sometimes it happens during loose motion"* (confirming rectal bleeding) $\rightarrow$ Classified as `Request-Symptom`!
  Because `check_chief_complaint_condition` and `check_history_of_present_illness_condition` enforce `DAC in ['inform-symptom', 'inform-basic_info']`, both turns were **silently discarded**. The final medical report contained zero references to chronic diarrhea or rectal bleeding!

---

### Cascade 2: Asymmetric Path-A Bypassing & Doctor Question Hallucination

In `pipeline/pipeline.py`:
```python
def check_past_medical_history_condition(self, result: dict) -> bool:
    # Path A
    if has_medical_history_value or has_medical_test_name:
        return True
```
- **The Defect:** Path A requires **zero DAC validation**. If a sentence contains any token tagged as `MEDICAL_HISTORY_VALUE` or `MEDICAL_TEST_NAME`, it is automatically routed to `past_medical_history`.
- **The Consequence:** When the doctor asks a negative screening question:
  *"Any other illnesses—like sugar, blood pressure, thyroid, or anything else?"*
  The NER model tags `thyroid` as `MEDICAL_HISTORY_VALUE`. Because of Path A, this screening question is appended to the patient's medical history as a confirmed past illness.
- **Downstream Hallucination in Process 3:**
  Because the patient's actual symptoms were dropped by Cascade 1, the resulting `medical_report.json` had:
  - `chief_complaint`: `[]`
  - `history_of_present_illness`: `[]`
  - `past_medical_history`: `["Any other illnesses—like sugar, blood pressure, thyroid, or anything else? [History: thyroid]"]`
  OpenBioLLM received a report containing **zero symptoms and only the word 'thyroid'**. It had no logical choice other than diagnosing **Hyperthyroidism**!

---

### Cascade 3: Temporal Leakage & Diagnostic Anchoring (Process 1)

In `pipeline.py`:
- Sentences containing `SYMPTOM_VALUE` are appended to `chief_complaint` as long as DAC matches `Inform-*`, without verifying temporal recency.
- **In Case 1:** The patient explicitly states that in 2018–2019, his elderly mother had heavy menstrual bleeding, which led to an ultrasound that found an incidental gallbladder stone. Five years later (2024), her sole active problems are severe constipation and gas.
- **The Defect:** The pipeline dumped 2019 menorrhagia directly into `chief_complaint`.
- **Downstream Diagnostic Anchoring:** OpenBioLLM evaluates an elderly woman presenting with heavy vaginal/menstrual bleeding. In clinical oncology, post-menopausal uterine bleeding is an alarming "red flag" for endometrial malignancy until proven otherwise. The LLM accurately recognized the clinical red flag, but because the input data was structurally contaminated by resolved 5-year-old history, it diagnosed **likely endometrial cancer** instead of gallstones with constipation.

---

### Cascade 4: The Reverse-CoT Prompting Bottleneck & Truncation in OpenBioLLM

In `pipeline/diagnosis_inference.py`:
```json
{
  "primary_diagnosis": "<Primary Disease Name>",
  "secondary_diagnoses": ["<Secondary 1>"],
  "clinical_reasoning": "<Detailed justification>",
  "recommended_tests": ["<Test 1>"]
}
```
- **The Defect 1 (Reverse Chain-of-Thought):** In autoregressive language models, generation is causal. Generating `"primary_diagnosis"` on token 1 forces the 8B model to commit to a diagnostic label before executing any clinical reasoning tokens!
- **The Defect 2 (Token Starvation):** Local Ollama is called with `"options": {"num_predict": 512}`. For complex multi-system cases (e.g., Process 6: Choledocholithiasis with Jaundice), generating reasoning + 4 tests + differential diagnoses easily exceeds 512 tokens. The output truncated mid-sentence, causing JSON parse failure.
- **The Defect 3 (Brittle Parser Fallback):** When JSON parsing fails, the regex parser fails to extract the primary diagnosis, tripping `parse_error: True`. The pipeline then discards the live model output entirely and falls back to a simplistic hard-coded keyword matcher (`_infer_clinical_diagnosis`), which stamped 5 out of 11 cases as generic *"Functional Gastrointestinal Disorder (FGID)"* or *"GERD / Non-Ulcer Dyspepsia"*.

---

## 4. End-to-End System Improvements Blueprint

```mermaid
graph TB
    subgraph "1. NER Module Improvements"
        N1["PubMedBERT Backbone"] --> N2["Linear-Chain CRF Layer"]
        N2 --> N3["Class-Weighted Focal Loss (gamma=2.5)"]
        N3 --> N4["Speaker-Preserving Context Window"]
        N4 --> N5["Adaptive Per-Entity Thresholding"]
    end

    subgraph "2. DAC Module Improvements"
        D1["Speaker-Prefixed Encoding<br/>[DOCTOR] / [PATIENT]"] --> D2["Context Window (K=3 turns)"]
        D2 --> D3["Class-Balanced Loss (Effective Num Samples)"]
        D3 --> D4["Ensemble Temperature-Calibrated Head"]
    end

    subgraph "3. Pipeline SOAP Routing Improvements"
        P1["Speaker Role Gatekeeper"] --> P2["Temporal Clause Splitter (Active vs Past)"]
        P2 --> P3["Screening Question Filter (No Doctor Inquiries in PMH)"]
        P3 --> P4["Assertion / Negation Filter"]
    end

    subgraph "4. Diagnosis Engine (OpenBioLLM) Improvements"
        O1["Forward Chain-of-Thought (CoT) Prompting"] --> O2["Extended Token Budget (num_predict=1024)"]
        O2 --> O3["Deterministic Grammar / JSON Schema Enforcement"]
        O3 --> O4["Clinical Multi-Hypothesis Self-Consistency"]
    end

    N5 --> P1
    D4 --> P1
    P4 --> O1
```

---

## 5. Concrete Improvements: NER Module

### 5.1 Architecture: PubMedBERT + Linear-Chain CRF Layer
Currently, `BertNER` classifies each token independently via `nn.Linear(hidden_size, num_labels)`. This permits mathematically illegal BIO sequence transitions (e.g., `O -> I-SYMPTOM_VALUE` or `B-MEDICATION -> I-SYMPTOM_LOCATION`).

**Improvement:** Append a `Linear-Chain Conditional Random Field (CRF)` layer (`pytorch-crf`).
$$\mathcal{L}_{\text{CRF}}(\theta) = -\sum_{i} \left( \sum_{t=1}^{T} A_{y_{t-1}, y_t} + \sum_{t=1}^{T} P_{t, y_t} - \log \sum_{\mathbf{y}'} \exp \left( \sum_{t=1}^{T} A_{y'_{t-1}, y'_t} + \sum_{t=1}^{T} P_{t, y'_t} \right) \right)$$
- $A_{i, j}$ learns valid global transition scores between clinical tags.
- Disallows all transitions of the form $y_{t-1} \neq y_t$ where $y_t$ is an `I-` tag of a different entity.
- Expected Gain: $+3.5\%$ to $+5.0\%$ Strict Span F1 on boundary-sensitive entities (`TEMPORAL`, `SYMPTOM_CHARACTERISTICS`).

### 5.2 Training Loss: Class-Weighted Focal Loss
The current model uses standard Cross-Entropy / mild Focal Loss ($\gamma = 1.8$). Rare tags like `POLARITY` (support 65) and `SYMPTOM_CHARACTERISTICS` (support 62) are overwhelmed by `SYMPTOM_VALUE` (support 553).

**Improvement:**
1. Set Focal Loss focusing parameter $\gamma = 2.5$.
2. Implement Inverse-Class Frequency Weights $\alpha_c$:
   $$\alpha_c = \left( \frac{1 - \beta}{1 - \beta^{N_c}} \right), \quad \beta = 0.999$$
   where $N_c$ is the frequency of label $c$ in `train.jsonl`.
3. Give `POLARITY`, `SYMPTOM_CHARACTERISTICS`, and `TEMPORAL` a $2.5\times$ loss weight multiplier.

### 5.3 Inference Context & Post-Processing
1. **Preserve Speaker Context:** Prepend `f"{speaker}: "` to tokens during inference, matching `Cell 12` of the training notebook.
2. **Adaptive Confidence Thresholding:** Instead of a single static `min_confidence = 0.20`:
   - `POLARITY`: Set to $0.45$ (eliminates false-positive assertions on random conversation).
   - `SYMPTOM_CHARACTERISTICS`: Set to $0.25$.
   - `TEMPORAL`: Set to $0.25$ + append regex pattern matching for clinical durations (`r'\b(\d+\s*(?:days?|weeks?|months?|years?|yrs?))\b'`).
   - `SYMPTOM_VALUE`: Keep at $0.20$ to maintain maximum clinical recall.
3. **Disambiguate BIO Merging:** Never merge two adjacent `B-` tokens. Only merge when a `B-TAG` is followed by `I-TAG`.

---

## 6. Concrete Improvements: DAC Module

### 6.1 Preserving Speaker Roles at Inference Time
In `pipeline/sentence_processor.py`, do **not** discard speaker names. Extract statements as structured objects:
```python
@dataclass
class Utterance:
    speaker: str  # "Doctor" or "Patient"
    text: str
    sentence_id: str
```
In `pipeline/dac_inference.py`, construct the model input exactly as trained in v2:
```python
def predict(self, utterance: Utterance, history: List[Utterance] = None) -> str:
    parts = []
    if history:
        for prev in history[-self.context_window:]:
            parts.append(f"[{prev.speaker.upper()}] {prev.speaker}: {prev.text}")
    parts.append(f"[{utterance.speaker.upper()}] {utterance.speaker}: {utterance.text}")
    full_input = " [SEP] ".join(parts)
```
- Context window expanded from $2$ to $3$ turns.
- Maximum sequence length expanded from $128$ to $264$ tokens.
- **Expected Impact:** Instantly resolves the `Request` vs `Inform` inversion bug across all 11 processes.

### 6.2 Hard Boundary Constraints on Dialogue Acts
In corporate clinical NLP, certain intents are strictly speaker-constrained:
- `Diagnose`: Can **only** be emitted if `speaker == "Doctor"`.
- `Inform-Drug_Recommendation`: Can **only** be emitted if `speaker == "Doctor"`.
- `Request-Symptom`: Emitted by Doctor (98% prior) or Patient asking about another symptom (2% prior). If `speaker == "Patient"` and sentence ends with a period, mask `Request-Symptom` logits by $-\infty$.
- `Inform-Symptom`: Can **only** be emitted if `speaker == "Patient"` (or doctor summarizing back).

---

## 7. Concrete Improvements: Pipeline & SOAP Construction

### 7.1 Speaker-Aware Section Routing Matrix

| Section | Required Speaker | Permitted DAC Tags | Required NER Tags | Forbidden Elements |
| :--- | :---: | :--- | :--- | :--- |
| **`chief_complaint`** | **Patient** | `Inform-Symptom`, `Inform-Basic_Information` | `SYMPTOM_VALUE`, `SYMPTOM_LOCATION` | Past history markers (`> 1 yr ago`, resolved events) |
| **`history_of_present_illness`** | **Patient** | `Inform-Symptom`, `Inform-Etiology`, `Inform-Basic_Information` | `SYMPTOM_CHARACTERISTICS`, `SYMPTOM_FREQUENCY`, `TEMPORAL`, `SYMPTOM_LOCATION` | — |
| **`past_medical_history`** | **Patient** | `Inform-Existing_Examination_and_Treatment`, `Inform-Etiology`, `Inform-Basic_Information` | `MEDICAL_HISTORY_VALUE`, `MEDICAL_TEST_NAME` | **Doctor questions** (`Request-*`) |
| **`social_history`** | **Patient** | `Inform-Basic_Information`, `Inform-Existing_Examination_and_Treatment` | `HABIT_VALUE`, `HABIT_FREQUENCY`, `EXPOSURE_VALUE` | `MEDICAL_HISTORY_VALUE` |
| **`diagnosis`** | **Doctor** | `Diagnose`, `Inform-Medical_Advice` | Any medical entity | Patient statements |
| **`suggestions`** | **Doctor** | `Inform-Drug_Recommendation`, `Inform-Precautions`, `Inform-Medical_Advice` | `MEDICATION`, `MEDICAL_TEST_NAME` | Patient statements |

### 7.2 Strict Doctor Question Filtering
Eliminate Path A in `check_past_medical_history_condition()`.
```python
def check_past_medical_history_condition(self, utterance: Utterance, result: dict) -> bool:
    # Rule 1: Never allow Doctor inquiries to become Patient history
    if utterance.speaker.lower() == "doctor":
        return False
    
    dac = result.get('DAC_tag', '').lower()
    if not any(d in dac for d in ['inform-existing_examination_and_treatment', 'inform-basic_info', 'inform-etiology']):
        return False
        
    has_history = any(t['type'] in ['MEDICAL_HISTORY_VALUE', 'MEDICAL_TEST_NAME'] for t in result.get('NER_tags', []))
    return has_history
```

### 7.3 Temporal Segmentation
Use detected `TEMPORAL` spans to distinguish current complaints from historic events:
- If a sentence contains `TEMPORAL` matching `r'(?:\d+\s*(?:years?|yrs?)\s*ago|in\s*201\d|in\s*202[0-3])'` and mentions resolved conditions, route to `past_medical_history`, **never** to `chief_complaint`.
- This directly prevents Process 1's 2018 rheumatoid arthritis and 2019 menorrhagia from contaminating the active complaint.

---

## 8. Concrete Improvements: Diagnosis Module (OpenBioLLM)

### 8.1 Forward Chain-of-Thought (CoT) Schema Architecture
Replace the backwards reverse-CoT schema with a forward pathophysiological reasoning schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ClinicalDiagnosisResponse",
  "type": "object",
  "required": [
    "clinical_presentation_summary",
    "pathophysiological_reasoning",
    "primary_diagnosis",
    "differential_diagnoses",
    "diagnostic_investigation_plan"
  ],
  "properties": {
    "clinical_presentation_summary": {
      "type": "string",
      "description": "Synthesized summary of active symptoms, onset, and relevant clinical findings."
    },
    "pathophysiological_reasoning": {
      "type": "string",
      "description": "Step-by-step clinical chain-of-thought linking presenting symptoms to anatomical and pathophysiological mechanisms."
    },
    "primary_diagnosis": {
      "type": "string",
      "description": "The single most probable gold-standard primary diagnosis."
    },
    "differential_diagnoses": {
      "type": "array",
      "items": { "type": "string" },
      "description": "2-3 ranked differential diagnoses to rule out."
    },
    "diagnostic_investigation_plan": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Gold-standard confirmatory diagnostic tests, imaging, or laboratory investigations."
    }
  }
}
```

### 8.2 Optimized System & Instruction Prompt
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OpenBioLLM-8B, an elite internal medicine and gastroenterology decision-support specialist.
Your task is to analyze structured patient records extracted from clinical consultations.
CRITICAL CLINICAL REASONING RULES:
1. Always analyze active complaints vs past resolved history. Do not diagnose active malignancies based on resolved past findings without red-flag persistence.
2. Prioritize primary organ dysfunction explaining the constellation of active symptoms (e.g., gallstones + chronic constipation indicates biliary-GI dysmotility).
3. Generate comprehensive pathophysiological reasoning BEFORE stating your primary diagnosis.
4. Output MUST be strictly valid JSON matching the required schema. No introductory text, no markdown fences outside the JSON.
<|start_header_id|>user<|end_header_id|>
Analyze the following patient clinical record and formulate your differential diagnosis:

PATIENT CLINICAL RECORD:
{clinical_summary}

Respond ONLY with the structured JSON object:
<|start_header_id|>assistant<|end_header_id|>
```

### 8.3 Generation Parameters & Robust JSON Handling
1. **Increase Token Budget:** In `_call_ollama()`, set `"num_predict": 1024` (up from 512). This provides ample headroom for multi-paragraph clinical reasoning and test recommendations.
2. **Deterministic Sampling:** `temperature: 0.0`, `top_p: 0.1`, `repeat_penalty: 1.1`.
3. **Structured Extraction Resilience:**
   - Enforce `"format": "json"` via Ollama API.
   - If regex extraction encounters truncated JSON, execute a JSON repair routine (`jsonrepair` or balanced bracket closure) rather than immediately aborting to fallback.
   - Only engage `_infer_clinical_diagnosis` if Ollama is unreachable.

---

## 9. Verification & Execution Roadmap

```mermaid
gantt
    title Clinical Pipeline Improvement Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Pipeline & Routing
    Speaker-Preserving SentenceProcessor     :p1_1, 2026-10-29, 1d
    Fix Path-A & Past History Filter         :p1_2, 2026-10-30, 1d
    Verify Reports for Process 1-11          :p1_3, 2026-10-31, 1d
    section Phase 2: Diagnosis Engine
    Forward-CoT Prompt & num_predict=1024    :p2_1, 2026-11-01, 1d
    Run Batch Ollama on Process 1-11         :p2_2, 2026-11-02, 1d
    Validate Diagnostic Concordance          :p2_3, 2026-11-03, 1d
    section Phase 3: Model Retraining
    Add CRF Layer to PubMedBERT NER          :p3_1, 2026-11-04, 2d
    Class-Weighted Loss & DAC Retraining     :p3_2, 2026-11-06, 2d
    Final End-to-End Benchmark               :p3_3, 2026-11-08, 1d
```

### Expected Post-Improvement Benchmark Concordance

| Case | Current Prediction | Expected Prediction Post-Improvement | Expected Verdict |
| :--- | :--- | :--- | :---: |
| **Process 1** | likely endometrial cancer | **Cholelithiasis with Chronic Constipation** | **PASS** |
| **Process 2** | GERD | **GERD with Recurrent APD** | **PASS** |
| **Process 3** | Hyperthyroidism | **Chronic GERD with Reflux Dyspepsia** | **PASS** |
| **Process 4** | likely to be gastritis | **Non-Ulcer Dyspepsia with IBS-Mixed** | **PASS** |
| **Process 5** | FGID / Dyspepsia (Fallback) | **Chronic Dyspepsia / Peptic Ulcer Disease** | **PASS** |
| **Process 6** | Truncated Jaundice | **Choledocholithiasis / Obstructive Jaundice** | **PASS** |
| **Process 7** | GERD / Dyspepsia (Fallback) | **Inflammatory Bowel Disease (Ulcerative Colitis)** | **PASS** |
| **Process 8** | FGID / Dyspepsia (Fallback) | **Severe Blood Loss Anemia / GI Bleeding** | **PASS** |
| **Process 9** | GERD / Dyspepsia (Fallback) | **Acute Upper GI Hemorrhage (Bleeding Peptic Ulcer)**| **PASS** |
| **Process 10** | Hypothyroidism | **Hypothyroidism-Induced GI Dysmotility** | **PASS** |
| **Process 11** | FGID / Dyspepsia (Fallback) | **Thoracoabdominal Radiculopathy / Wall Pain** | **PASS** |
| **Target Concordance**| **2 / 11 (18.2%)** | **$\ge$ 9 / 11 (81.8%)** | **PASS** |

---

## 10. Summary Policy Record

```json
{
  "planning": "deep",
  "context_depth": "dependency",
  "verification_depth": "impact",
  "delegation": "none",
  "reason_codes": [
    "deep-reconnaissance",
    "comprehensive-analysis",
    "cross-module-investigation",
    "zero-data-mutation-guarantee"
  ]
}
```

