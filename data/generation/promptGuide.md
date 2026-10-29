# MediTOD NER Batch Generator — Complete Reference

`create_batch_prompts_v3.py` · Anthropic Message Batches API

---

## 1 · Model Selection

| Parameter | Value | Why |
|---|---|---|
| Model | `claude-sonnet-4-6` | Best cost-to-quality ratio for structured generation; follows XML tagging rules precisely |
| `max_tokens` | 1900 | Fits 25-turn dialogues (~1400–1700 actual output tokens) |
| API | Message Batches (async) | 50% cost discount over synchronous calls |
| Prompt caching | `cache_control: ephemeral` on system prompt | 90% discount on repeated system prompt reads |

---

## 2 · Scenario Diversity Lists

Every user prompt is composed from **5 independent axes**. One item is randomly drawn from each per request.

### 2.1 Patient Profiles (40)

| # | Profile | # | Profile |
|---|---|---|---|
| 1 | elderly male (65+) | 21 | auto-rickshaw driver |
| 2 | elderly female (65+) | 22 | street food vendor |
| 3 | middle-aged man (40–55) | 23 | IT professional (night shifts) |
| 4 | middle-aged woman (40–55) | 24 | factory worker |
| 5 | young adult male (20–35) | 25 | police constable |
| 6 | young adult female (20–35) | 26 | domestic help / maid |
| 7 | teenage boy (13–18) | 27 | daily-wage labourer |
| 8 | teenage girl (13–18) | 28 | shopkeeper |
| 9 | child (6–12, accompanied by mother) | 29 | nursing staff |
| 10 | pregnant woman | 30 | school-going child (accompanied by father) |
| 11 | new mother (post-partum, <3 months) | 31 | newly married woman (20–25) |
| 12 | manual labourer | 32 | middle-aged diabetic man on insulin |
| 13 | office worker with sedentary lifestyle | 33 | elderly widow living alone |
| 14 | truck driver | 34 | young athlete / gym-goer |
| 15 | farmer | 35 | long-distance bus driver |
| 16 | construction worker | 36 | tailor / garment worker |
| 17 | teacher | 37 | security guard (12-hour shifts) |
| 18 | college student | 38 | fisherman |
| 19 | retired government employee | 39 | postman / delivery worker |
| 20 | homemaker | 40 | railway employee |

### 2.2 Visit Contexts (19)

| # | Context |
|---|---|
| 1 | first-time visit for new complaint |
| 2 | follow-up after recent hospitalisation |
| 3 | routine annual check-up |
| 4 | emergency walk-in with acute symptom |
| 5 | second opinion after another doctor's advice |
| 6 | pre-operative clearance visit |
| 7 | post-surgery recovery review |
| 8 | lab result discussion visit |
| 9 | medication refill visit |
| 10 | specialist referral visit |
| 11 | walk-in after self-treating for a week/month/several months |
| 12 | brought by family member (patient reluctant) |
| 13 | insurance / medico-legal documentation visit |
| 14 | chronic disease quarterly review |
| 15 | discharge follow-up (within 48 hours) |
| 16 | referred back from higher centre |
| 17 | workplace-mandated health screening |
| 18 | patient returning after missing follow-ups for months |
| 19 | Walk in after the symptoms become severe |

### 2.3 Clinical Focuses (19)

| # | Focus | # | Focus |
|---|---|---|---|
| 1 | respiratory (cough, breathlessness, wheezing) | 11 | liver disorder (jaundice, fatty liver) |
| 2 | gastrointestinal (acidity, gas, stomach pain) | 12 | kidney disease (swelling, creatinine) |
| 3 | cardiovascular (chest pain, palpitations, BP) | 13 | paediatric illness (child fever, diarrhoea) |
| 4 | metabolic (diabetes, thyroid, cholesterol) | 14 | geriatric multi-morbidity |
| 5 | gynaecological (periods, PCOS, discharge) | 15 | allergy / immunological |
| 6 | neurological (headache, dizziness, numbness) | 16 | surgical (hernia, piles, abscess) |
| 7 | skin (rash, itching, fungal infection) | 17 | obstetric (pregnancy complication) |
| 8 | infectious (fever, dengue, typhoid) | 18 | post-COVID / long-viral sequelae |
| 9 | occupational / environmental exposure | 19 | Alcohol abuse |
| 10 | anaemia / nutritional deficiency | | |

### 2.4 Complication Modifiers (28)

| # | Modifier | # | Modifier |
|---|---|---|---|
| 1 | strong family history of condition | 15 | history of substance abuse |
| 2 | known drug allergy | 16 | caring for sick family member simultaneously |
| 3 | managing another chronic illness | 17 | fasting for religious reasons |
| 4 | recently changed medication on own | 18 | poor health literacy, needs simple explanations |
| 5 | delayed seeking care for weeks | 19 | previously misdiagnosed |
| 6 | self-medicating with OTC drugs | 20 | refuses certain tests (fear or cost) |
| 7 | non-compliant (missing doses) | 21 | uncontrolled blood sugar despite treatment |
| 8 | uses Ayurvedic/homeopathic remedies | 22 | recently discharged with new symptoms |
| 9 | anxious, asks many questions | 23 | works night shifts, irregular sleep |
| 10 | downplays symptoms initially | 24 | seasonal worsening of symptoms |
| 11 | no prior medical history | 25 | History of alcohol abuse |
| 12 | recently travelled to rural/congested area | 26 | Severely poor financial condition |
| 13 | cannot afford prescribed medications | 27 | Consults informal health care providers [quacks] |
| 14 | on multiple medications from different doctors | 28 | Walk in after the symptoms become severe |

### 2.5 Style Variants (29)

| # | Style directive |
|---|---|
| 1 | Doctor is brisk and clinical; patient gives short answers |
| 2 | Doctor is empathetic and reassuring; patient is worried and verbose |
| 3 | Patient is defensive about lifestyle habits; doctor gently probes |
| 4 | Patient uses broken Indian-English with simple grammar; doctor responds formally |
| 5 | Doctor is teaching a medical student — narrates findings aloud |
| 6 | Family member interjects with additional history |
| 7 | Doctor is running behind schedule — concise and focused |
| 8 | Patient is first-timer, unsure how to describe symptoms |
| 9 | Patient is a frequent visitor who knows medical terms well |
| 10 | Patient is reluctant to disclose habits; doctor uses open questions |
| 11 | Doctor summarises findings aloud before ordering tests |
| 12 | Patient is argumentative, questions every recommendation |
| 13 | Doctor is thorough — asks about every system before concluding |
| 14 | Patient is stoic, gives minimal responses; doctor extracts info actively |
| 15 | Warm friendly tone — doctor and patient know each other from past visits |
| 16 | Patient is in visible pain, gives fragmented answers |
| 17 | Doctor is mentoring a junior, explains clinical reasoning while consulting |
| 18 | Patient compares with information found on the internet |
| 19 | Patient is emotionally distressed; doctor provides reassurance throughout |
| 20 | Patient switches between topics; doctor guides back systematically |
| 21 | Doctor uses teach-back — asks patient to repeat instructions |
| 22 | Patient has very little knowledge of common medical terms |
| 23 | Patient is illiterate or school dropout, cannot describe the symptoms properly |
| 24 | Patient is extremely shy in front of the doctor |
| 25 | Patient is alcoholic, his/her family member describes the symptoms |
| 26 | Patient is very nervous in the clinical environment. |
| 27 | Doctor knows the patient personally — opens with casual family/life chat before medicine |
| 28 | Patient works in a factory/mine — doctor probes occupational exposures thoroughly |
| 29 | Doctor walks through diagnostic reasoning step-by-step, naming the condition explicitly |

### Combinatorial Space

```
40 × 19 × 19 × 28 × 29 = 11,723,040 unique prompt combinations
```

For n=200, every prompt is guaranteed unique (deduplication enforced in code).

---

## 3 · Prompt Structure

### 3.1 System Prompt (cached)

Sent once, cached for all subsequent requests at 10% token cost.

```
┌─────────────────────────────────────────────────────────────┐
│  Role definition                                            │
│  "You are an expert synthetic medical dialogue generator    │
│   for Indian English clinical NLP datasets..."              │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT FORMAT                                              │
│  - Inline XML tags, no nesting                              │
├─────────────────────────────────────────────────────────────┤
│  NER TAG DEFINITIONS (12 tags)                              │
│  Each tag has: DEFINITION, INCLUDES, EXCLUDES, BOUNDARY,    │
│  ✓ correct examples, ✗ anti-pattern examples                │
│  (see Section 5 for full definitions)                       │
├─────────────────────────────────────────────────────────────┤
│  INTENT CLASSIFICATION — DAC (16 intents)                   │
│  Grouped by REQUEST / INFORM / SPECIAL with full            │
│  definitions and disambiguation boundaries                  │
│  (see Section 5.2 for full definitions)                     │
├─────────────────────────────────────────────────────────────┤
│  QUALITY RULES (11 rules)                                   │
│  1. Medical plausibility                                    │
│  2. Indian English authenticity                             │
│  3. Entity density (both speakers, ≥1 tag/turn, 40–50 tot) │
│  4. Polarity balance (mix yes/no)                           │
│  5. Length: 25 utterances                                   │
│  6. Follow style instruction                                │
│  7. No invented tags + Aggressive SYMPTOM_LOCATION tagging  │
│     + Negative examples for common LLM failure modes        │
│  8. Precise boundaries                                      │
│  9. Natural fillers (umm, ahh — outside tags only)          │
│  10. English only (no Hindi/Hinglish — zero tolerance)      │
│  11. No tag-free zones (no 2+ consecutive untagged turns)   │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT SCHEMA                                              │
│  JSON array with speaker, intent, and text                  │
├─────────────────────────────────────────────────────────────┤
│  ONE-SHOT EXAMPLE (all 12 tags demonstrated)                │
│                                                             │
│  cache_control: { type: "ephemeral" }                       │
└─────────────────────────────────────────────────────────────┘
```

**~2350 tokens**

### 3.2 User Prompt (unique per request)

Template:
```
Scenario: {profile} | {visit_context} | {clinical_focus} | {complication}
Style: {style_variant}
Required terms: TAG1: term1, term2 | TAG2: term3 | ...

Generate a highly realistic Indian clinic dialogue (25 utterances).
Every required term must appear as a tagged span. Additionally, tag ALL
medical entities that arise organically in the dialogue — the required
terms are minimums, not limits.
Ensure the dialogue includes:
  - At least one Diagnose intent turn where the doctor explicitly names the condition.
  - At least one Inform-Drug_Recommendation turn where the doctor prescribes specific medicines.
  - Other intent for any general/social/personal conversation turns (these may have zero NER tags).
Return ONLY the JSON array.
```

Example (rendered):
```
Scenario: truck driver | emergency walk-in with acute symptom |
  cardiovascular concern (chest pain, palpitations, BP) |
  who has been self-medicating with OTC drugs
Style: Patient is in visible pain, gives fragmented answers.
Required terms: SYMPTOM_VALUE: chest pain, palpitations |
  TEMPORAL: since morning | HABIT_FREQUENCY: 2 packs a day |
  MEDICATION: aspirin, sorbitrate | SYMPTOM_FREQUENCY: every few minutes

Generate a highly realistic Indian clinic dialogue (25 utterances).
Every required term must appear as a tagged span. Additionally, tag ALL
medical entities that arise organically in the dialogue — the required
terms are minimums, not limits.
Ensure the dialogue includes:
  - At least one Diagnose intent turn where the doctor explicitly names the condition.
  - At least one Inform-Drug_Recommendation turn where the doctor prescribes specific medicines.
  - Other intent for any general/social/personal conversation turns (these may have zero NER tags).
Return ONLY the JSON array.
```

**~243 tokens**

---

## 4 · Input Tokens (per JSONL line)

Each line in `batch_requests.jsonl` is one API request:

```json
{
  "custom_id": "meditod-phase2b-0001",
  "params": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1900,
    "system": [ { "type": "text", "text": "...", "cache_control": {"type": "ephemeral"} } ],
    "messages": [ { "role": "user", "content": "Scenario: ..." } ]
  }
}
```

| Component | Tokens | Billed as |
|---|---|---|
| System prompt | ~2350 | Cache write (1st request) / Cache read (rest) |
| User prompt | ~243 | Standard input |
| **Total per line** | **~2593** | **~478 effective** (with caching, after 1st) |

---

## 5 · Output Format (exact)

Claude returns a **JSON array** of speaker-text objects with inline XML entity tags:

```json
[
  {"speaker": "Doctor",  "text": "Good morning. What has been bothering you?"},
  {"speaker": "Patient", "text": "Doctor, <TEMPORAL>since two weeks</TEMPORAL> I am having <SYMPTOM_VALUE>breathlessness</SYMPTOM_VALUE>. It is <SYMPTOM_CHARACTERISTICS>worse at night</SYMPTOM_CHARACTERISTICS>."},
  {"speaker": "Doctor",  "text": "Any <SYMPTOM_VALUE>loose motion</SYMPTOM_VALUE> or stomach problem?"},
  {"speaker": "Patient", "text": "<POLARITY>Yes</POLARITY> doctor. My <SYMPTOM_LOCATION>chest</SYMPTOM_LOCATION> feels heavy."},
  {"speaker": "Doctor",  "text": "History of <MEDICAL_HISTORY_VALUE>diabetes</MEDICAL_HISTORY_VALUE> or <MEDICAL_HISTORY_VALUE>hypertension</MEDICAL_HISTORY_VALUE>?"},
  {"speaker": "Patient", "text": "<POLARITY>Yes</POLARITY>, both. Taking <MEDICATION>metformin</MEDICATION> and <MEDICATION>inhaler</MEDICATION>."},
  {"speaker": "Doctor",  "text": "Still <HABIT_VALUE>smoking</HABIT_VALUE>?"},
  {"speaker": "Patient", "text": "<POLARITY>Yes</POLARITY>, <HABIT_FREQUENCY>half a pack a day</HABIT_FREQUENCY>."}
]
```

### 5.1 NER Tag Definitions (12 tags)

Each tag has a strict definition, boundary, and anti-patterns in the system prompt.

| Tag | Definition | INCLUDES | EXCLUDES | Anti-pattern |
|---|---|---|---|---|
| `POLARITY` | Binary confirmation/denial | yes, no, positive, negative, denied, never, confirmed, present, absent | Trend words (increasing, worsening), finding verbs (detected, observed) | ✗ `<POLARITY>increasing</POLARITY>` |
| `SYMPTOM_VALUE` | Name of a current symptom/sign/condition | fever, cough, chest pain, PCOS, jaundice | Body parts (→ LOCATION), past conditions (→ HISTORY), non-medical words (precautions, treatment) | ✗ `<SYMPTOM_VALUE>precautions</SYMPTOM_VALUE>` |
| `SYMPTOM_LOCATION` | Body part or anatomical region | stomach, chest, head, leg, eyes, skin, kidney | Organs as condition names (liver disease → liver=location) | Tag EVERY mention, even if repeated |
| `SYMPTOM_CHARACTERISTICS` | Quality/descriptor/modifier of a symptom | sharp, dull, burning, radiating, constant, throbbing, watery | Work schedules (12-hour shift), time-of-day, non-medical descriptors | ✗ `<SYMPTOM_CHARACTERISTICS>12-hour shift</SYMPTOM_CHARACTERISTICS>` |
| `SYMPTOM_FREQUENCY` | How often a SYMPTOM recurs | daily, every morning, twice a week, on and off, continuously | Habit frequencies (→ HABIT_FREQUENCY) | ✗ Using HABIT_FREQUENCY for symptom recurrence |
| `TEMPORAL` | WHEN something happened (time reference) | since two weeks, last month, in January, six months ago | Recurrence patterns (→ SYMPTOM_FREQUENCY or HABIT_FREQUENCY) | "WHEN" = TEMPORAL; "HOW OFTEN" = FREQUENCY |
| `MEDICATION` | Name/type of a medicine or substance | paracetamol, metformin, inhaler, ORS, ayurvedic churna | Purpose descriptions (for pain, for sleep) | ✗ `<MEDICATION>for pain</MEDICATION>` |
| `MEDICAL_TEST_NAME` | Name of a diagnostic test | CBC, HbA1c, X-ray, CT scan, ECG, blood culture | Test results (raised, normal) | Never tag as MEDICAL_HISTORY_VALUE |
| `MEDICAL_HISTORY_VALUE` | PAST condition/surgery mentioned as history | C-section, old fracture, childhood jaundice | Current symptoms (→ SYMPTOM_VALUE) | Currently experiencing → SYMPTOM_VALUE |
| `HABIT_VALUE` | Name of a lifestyle habit/substance | smoking, alcohol, pan masala, cycling, yoga | Frequency of the habit (→ HABIT_FREQUENCY) | — |
| `HABIT_FREQUENCY` | How much/often a HABIT is performed | a pack a day, 3 beers a day, socially, once in a while | Symptom recurrence (→ SYMPTOM_FREQUENCY) | ✗ `pain <HABIT_FREQUENCY>every morning</HABIT_FREQUENCY>` |
| `EXPOSURE_VALUE` | Environmental/occupational exposure source | dust, chemicals, secondhand smoke, tick bite, mercury | Location of exposure (factory, mine → untagged) | — |

### 5.2 Intent Definitions — DAC (16 intents)

**REQUEST intents** (asking for information):

| Intent | Definition | Example |
|---|---|---|
| `Request-Basic_Information` | Asks for demographics, lifestyle, or general context | "What brings you here?", "How old is the child?" |
| `Request-Symptom` | Asks about current symptoms, nature, severity | "Any fever?", "Is the pain sharp or dull?" |
| `Request-Etiology` | Asks about causes, risk factors, habits, family history | "Any family history of diabetes?" |
| `Request-Existing_Examination_and_Treatment` | Asks about ongoing medications or past tests | "What medicines are you taking?" |
| `Request-Medical_Advice` | Patient asks for guidance on treatment/prognosis | "What should I do about this?" |
| `Request-Precautions` | Asks about things to avoid, side effects, safety | "Can I skip that test?" |
| `Request-Drug_Recommendation` | Patient asks which specific medicine to take | "What medicine should I buy?" |

**INFORM intents** (providing information):

| Intent | Definition | Example |
|---|---|---|
| `Inform-Symptom` | Describes current symptoms, quality, progression | "I have loose motion every morning" |
| `Inform-Basic_Information` | Provides demographics, lifestyle, general context | "I work night shifts at the factory" |
| `Inform-Etiology` | Shares causes, risk factors, habits, family/medical history | "My mother had hypertension" |
| `Inform-Existing_Examination_and_Treatment` | Reports current medications or past test results | "I take metformin and aspirin" |
| `Inform-Medical_Advice` | Doctor explains treatment plan or clinical reasoning | "We need to rule out infection first" |
| `Inform-Precautions` | Warns about risks, side effects, things to avoid | "Avoid painkillers without prescription" |
| `Inform-Drug_Recommendation` | Doctor PRESCRIBES or NAMES specific medicines | "I'm prescribing metronidazole syrup" |

**SPECIAL intents:**

| Intent | Definition | Boundary |
|---|---|---|
| `Diagnose` | Doctor NAMES or IDENTIFIES the condition | Must explicitly name the disease/condition |
| `Other` | Greetings, farewells, personal chat with ZERO medical content | "What brings you here?" = Request-Basic_Information, NOT Other |

---

## 6 · Output Token Estimates

| Metric | Value |
|---|---|
| Turns per dialogue | 25 |
| Avg tokens per turn | ~70–80 (including JSON syntax + tags) |
| **Typical output per request** | **~1400–1700 tokens** |
| `max_tokens` budget | 1900 (headroom for dense dialogues) |
| **Expected tags per turn** | **~2.0** |
| **Total tags per dialogue** | **~45–55** |

### Turn-by-turn tag density (typical 25-turn dialogue)

| Phase | Turns | Tags/turn | Tags total |
|---|---|---|---|
| Opening + chief complaint | 3–4 | 0–3 | ~6 |
| Symptom exploration | 7–8 | 2–3 | ~18 |
| History + medications | 5–6 | 2–3 | ~12 |
| Lifestyle + exposure | 3–4 | 2–3 | ~8 |
| Diagnosis + prescription + plan | 4–5 | 2–3 | ~10 |
| **Total** | **25** | **avg ~2.0** | **~50** |

---

## 7 · Post-Processing Format

After batch completes, run:
```bash
python create_batch_prompts_v3.py --postprocess results.jsonl --out_stem meditod
```

### 7.1 XML → BIO Conversion

```
Input:  <SYMPTOM_VALUE>chest pain</SYMPTOM_VALUE> since yesterday
Output:
  chest      B-SYMPTOM_VALUE
  pain       I-SYMPTOM_VALUE
  since      O
  yesterday  O
```

### 7.2 Output Files

| File | Format | Use |
|---|---|---|
| `meditod_xml.jsonl` | Raw dialogues with inline XML tags | Human review, annotation QC |
| `meditod_bio.jsonl` | BIO-tokenised turns | **NER model training** |
| `meditod_failed.jsonl` | Failed rows with reason | Debugging, re-submission |

### 7.3 BIO Output Structure (per line)

```json
{
  "id": "meditod-phase2b-0001",
  "bio": [
    {
      "speaker": "Patient",
      "tokens": [
        {"token": "Doctor,",        "tag": "O",               "speaker": "Patient"},
        {"token": "since",          "tag": "B-TEMPORAL",      "speaker": "Patient"},
        {"token": "two",            "tag": "I-TEMPORAL",      "speaker": "Patient"},
        {"token": "weeks",          "tag": "I-TEMPORAL",      "speaker": "Patient"},
        {"token": "I",              "tag": "O",               "speaker": "Patient"},
        {"token": "am",             "tag": "O",               "speaker": "Patient"},
        {"token": "having",         "tag": "O",               "speaker": "Patient"},
        {"token": "breathlessness", "tag": "B-SYMPTOM_VALUE", "speaker": "Patient"}
      ]
    }
  ]
}
```

### 7.4 Quality Gate

Each response must pass before entering training set:

| Check | Threshold | On fail |
|---|---|---|
| Valid JSON | Must parse | → `_failed.jsonl` |
| Turn count | 20–30 turns | → `_failed.jsonl` |
| Tagged spans | ≥ 30 spans | → `_failed.jsonl` |
| Intent field | Present + valid (16 classes) | → `_failed.jsonl` |
| Avg tags/turn | ≥ 1.5 | → `_failed.jsonl` |
| Zero-tag turns | Only allowed for `Other` intent (and opening greeting) | → `_failed.jsonl` |
| Consecutive zero-tag | No 3+ consecutive zero-tag turns | → `_failed.jsonl` |

---

## 8 · Token & Cost Analysis (n=200)

### Claude Sonnet 4.6 Pricing

| Component | Rate / 1M tokens | With Batch (50% off) |
|---|---|---|
| Input (standard) | $3.00 | $1.50 |
| Output | $15.00 | $7.50 |
| Cache write (5m TTL) | $3.75 | $1.875 |
| Cache read | $0.30 | $0.15 |

### Token Summary (200 requests)

| Metric | Tokens |
|---|---|
| System prompt (cached) | ~2350 |
| Avg user prompt | ~243 |
| Input without caching | ~518,600 |
| Input with caching (effective) | ~95,600 |
| Cache saving | ~82% |
| Output per request (typical) | ~1,500 |
| **Total output (200 reqs)** | **~300,000** |

### Cost Comparison (200 requests)

| Scenario | Input | Output | **Total** |
|---|---|---|---|
| Standard (no cache, no batch) | $1.56 | $4.50 | **$6.06** |
| Cache only | $0.29 | $4.50 | **$4.79** |
| Batch only (no cache) | $0.78 | $2.25 | **$3.03** |
| **Cache + Batch** | **$0.14** | **$2.25** | **≈ $2.39** |

> **With Cache + Batch you pay ~$2.40 for 200 transcripts** — 61% cheaper than standard pricing.

### Cost Scaling

| Requests | Standard | Cache + Batch | Saving |
|---|---|---|---|
| 100 | ~$3.03 | ~$1.20 | ~$1.83 |
| 200 | ~$6.06 | ~$2.39 | ~$3.67 |
| 500 | ~$15.15 | ~$5.98 | ~$9.17 |
| 1,000 | ~$30.30 | ~$11.95 | ~$18.35 |

---

## Quick Start

```bash
# Generate 200 batch requests
python create_batch_prompts_v3.py --n 200

# Submit to Anthropic
python -c "
import anthropic, json
client = anthropic.Anthropic()
reqs = [json.loads(l) for l in open('batch_requests.jsonl')]
batch = client.messages.batches.create(
    requests=[anthropic.types.message_create_params.Request(
        custom_id=r['custom_id'], params=r['params']
    ) for r in reqs]
)
print('Batch ID:', batch.id)
"

# Monitor: https://console.anthropic.com → Batches

# Post-process results
python create_batch_prompts_v3.py --postprocess results.jsonl --out_stem meditod
```
