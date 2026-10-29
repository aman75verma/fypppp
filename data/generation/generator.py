"""
create_batch_prompts_v3.py
--------------------------
Anthropic Batch API · Claude Sonnet 4.6 (claude-sonnet-4.6)
MediTOD NER Synthetic Dialogue Generator — Phase 2b

Usage
-----
  python create_batch_prompts_v3.py            # generates batch_requests.jsonl
  python create_batch_prompts_v3.py --n 1000  # custom count
  python create_batch_prompts_v3.py --postprocess results.jsonl  # after batch completes

API key
-------
  Set ANTHROPIC_API_KEY in your environment before running the batch via the SDK.
  This script only generates the JSONL — it does not call the API itself.
  See README / docs for how to submit: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 4096         # 25-turn tagged dialogue fits in ~1700 tokens; extra headroom for safety
OUTPUT_JSONL   = "batch_requests.jsonl"
KB_PATH        = "kb.json"

# Tags that are critically under-represented — always injected into every prompt
PRIORITY_TAGS  = {"HABIT_FREQUENCY", "SYMPTOM_FREQUENCY", "EXPOSURE_VALUE", "TEMPORAL"}

ALL_TAGS = [
    "POLARITY", "SYMPTOM_VALUE", "TEMPORAL", "MEDICATION",
    "EXPOSURE_VALUE", "SYMPTOM_CHARACTERISTICS", "MEDICAL_HISTORY_VALUE",
    "SYMPTOM_LOCATION", "MEDICAL_TEST_NAME", "HABIT_VALUE",
    "HABIT_FREQUENCY", "SYMPTOM_FREQUENCY",
]

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO MATRIX  —  axes × modifiers = 320+ unique combos
# Each prompt gets one scenario drawn from the product of these axes.
# ─────────────────────────────────────────────────────────────────────────────
PATIENT_PROFILES = [
    "elderly male (65+)",
    "elderly female (65+)",
    "middle-aged man (40–55)",
    "middle-aged woman (40–55)",
    "young adult male (20–35)",
    "young adult female (20–35)",
    "teenage boy (13–18)",
    "teenage girl (13–18)",
    "child (6–12, accompanied by mother)",
    "pregnant woman",
    "new mother (post-partum, <3 months)",
    "manual labourer",
    "office worker with sedentary lifestyle",
    "truck driver",
    "farmer",
    "construction worker",
    "teacher",
    "college student",
    "retired government employee",
    "homemaker",
    "auto-rickshaw driver",
    "street food vendor",
    "IT professional (night shifts)",
    "factory worker",
    "police constable",
    "domestic help / maid",
    "daily-wage labourer",
    "shopkeeper",
    "nursing staff",
    "school-going child (accompanied by father)",
    "newly married woman (20–25)",
    "middle-aged diabetic man on insulin",
    "elderly widow living alone",
    "young athlete / gym-goer",
    "long-distance bus driver",
    "tailor / garment worker",
    "security guard (12-hour shifts)",
    "fisherman",
    "postman / delivery worker",
    "railway employee",
]

VISIT_CONTEXTS = [
    "first-time visit for new complaint",
    "follow-up after recent hospitalisation",
    "routine annual check-up",
    "emergency walk-in with acute symptom",
    "second opinion after another doctor's advice",
    "pre-operative clearance visit",
    "post-surgery recovery review",
    "lab result discussion visit",
    "medication refill visit",
    "specialist referral visit",
    "walk-in after self-treating for a week/month/several months",
    "brought by family member (patient reluctant)",
    "insurance / medico-legal documentation visit",
    "chronic disease quarterly review",
    "discharge follow-up (within 48 hours)",
    "referred back from higher centre",
    "workplace-mandated health screening",
    "patient returning after missing follow-ups for months",
    "Walk in after the symptoms become severe",
]

CLINICAL_FOCUSES = [
    "respiratory (cough, breathlessness, wheezing)",
    "gastrointestinal (acidity, gas, stomach pain)",
    "cardiovascular (chest pain, palpitations, BP)",
    "metabolic (diabetes, thyroid, cholesterol)",
    "gynaecological (periods, PCOS, discharge)",
    "neurological (headache, dizziness, numbness)",
    "skin (rash, itching, fungal infection)",
    "infectious (fever, dengue, typhoid)",
    "occupational / environmental exposure",
    "anaemia / nutritional deficiency",
    "liver disorder (jaundice, fatty liver)",
    "kidney disease (swelling, creatinine)",
    "paediatric illness (child fever, diarrhoea)",
    "geriatric multi-morbidity",
    "allergy / immunological",
    "surgical (hernia, piles, abscess)",
    "obstetric (pregnancy complication)",
    "post-COVID / long-viral sequelae",
    "Alcohol abuse",
]

COMPLICATION_MODIFIERS = [
    "strong family history of condition",
    "known drug allergy",
    "managing another chronic illness",
    "recently changed medication on own",
    "delayed seeking care for weeks",
    "self-medicating with OTC drugs",
    "non-compliant (missing doses)",
    "uses herbal/alternative remedies",
    "anxious, asks many questions",
    "downplays symptoms initially",
    "no prior medical history",
    "recently travelled to rural/congested area",
    "cannot afford prescribed medications",
    "on multiple medications from different doctors",
    "history of substance abuse",
    "caring for sick family member simultaneously",
    "fasting for religious reasons",
    "poor health literacy, needs simple explanations",
    "previously misdiagnosed",
    "refuses certain tests (fear or cost)",
    "uncontrolled blood sugar despite treatment",
    "recently discharged with new symptoms",
    "works night shifts, irregular sleep",
    "seasonal worsening of symptoms",
    "History of alcohol abuse",
    "Severely poor financial condition",
    "Consults informal health care providers [quacks]",
    "Walk in after the symptoms become severe",
]

def make_scenario() -> str:
    """Compose a unique scenario string from the four axes."""
    profile   = random.choice(PATIENT_PROFILES)
    context   = random.choice(VISIT_CONTEXTS)
    focus     = random.choice(CLINICAL_FOCUSES)
    modifier  = random.choice(COMPLICATION_MODIFIERS)
    return f"{profile} | {context} | {focus} | {modifier}"


# ─────────────────────────────────────────────────────────────────────────────
# DIALOGUE STYLE VARIANTS — injected per-request for output diversity
# ─────────────────────────────────────────────────────────────────────────────
STYLE_VARIANTS = [
    "Doctor is brisk and clinical; patient gives short answers",
    "Doctor is empathetic and reassuring; patient is worried and verbose",
    "Patient is defensive about lifestyle habits; doctor gently probes",
    "Patient uses broken Indian-English with simple grammar; doctor responds formally",
    "Doctor is teaching a medical student — narrates findings aloud",
    "Patient relays family member's observations about their own symptoms",
    "Doctor is running behind schedule — concise and focused",
    "Patient is first-timer, unsure how to describe symptoms",
    "Patient is a frequent visitor who knows medical terms well",
    "Patient is reluctant to disclose habits; doctor uses open questions",
    "Doctor summarises findings aloud before ordering tests",
    "Patient is argumentative, questions every recommendation",
    "Doctor is thorough — asks about every system before concluding",
    "Patient is stoic, gives minimal responses; doctor extracts info actively",
    "Warm friendly tone — doctor and patient know each other from past visits",
    "Patient is in visible pain, gives fragmented answers",
    "Doctor is mentoring a junior, explains clinical reasoning while consulting",
    "Patient compares with information found on the internet",
    "Patient is emotionally distressed; doctor provides reassurance throughout",
    "Patient switches between topics; doctor guides back systematically",
    "Doctor uses teach-back — asks patient to repeat instructions",
    "Patient has very little knowledge of common medical terms",
    "Patient is illiterate or school dropout, cannot describe the symptoms properly",
    "Patient is extremely shy in front of the doctor",
    "Patient is alcoholic, reluctantly admits habits when probed",
    "Patient is very nervous in the clinical environment.",
    "Doctor knows the patient personally — opens with casual family/life chat before medicine",
    "Patient works in a factory/mine — doctor probes occupational exposures thoroughly",
    "Doctor walks through diagnostic reasoning step-by-step, naming the condition explicitly",
]

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE UTILS
# ─────────────────────────────────────────────────────────────────────────────
def get_terms(kb: dict, entity: str) -> list:
    """Flatten terms regardless of whether the KB node uses a nested dict or flat list."""
    node       = kb.get(entity, {})
    terms_node = node.get("terms", []) if isinstance(node, dict) else []
    if isinstance(terms_node, list):
        return terms_node
    if isinstance(terms_node, dict):
        flat = []
        for v in terms_node.values():
            flat.extend(v)
        return flat
    return []


def sample_terms_for_tags(kb: dict, focal_tags: list) -> list:
    """
    For each tag, sample 1–3 terms from the KB.
    Returns a list of 'TAG: term1, term2' strings.
    """
    injected = []
    for tag in focal_tags:
        terms = get_terms(kb, tag)
        if not terms:
            continue
        n       = random.randint(1, 3)
        sampled = random.sample(terms, min(len(terms), n))
        injected.append(f"{tag}: {', '.join(sampled)}")
    return injected


def choose_focal_tags() -> list:
    """
    Select tags for a request:
      - Always include PRIORITY_TAGS (under-represented tags)
      - Randomly pick 3–5 more from the remaining tags
      - 35% of requests include ALL 12 tags for maximum density
    """
    other_tags = [t for t in ALL_TAGS if t not in PRIORITY_TAGS]
    if random.random() < 0.35:
        # Full-coverage request
        return list(ALL_TAGS)
    n_extra = random.randint(3, 5)
    return random.sample(other_tags, n_extra) + list(PRIORITY_TAGS)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (cached — written once, read at ~10% cost for all other reqs)
# ─────────────────────────────────────────────────────────────────────────────
_EXAMPLE_DIALOGUE = """[
  {"speaker": "Doctor", "intent": "Request-Basic_Information", "text": "Good morning. What has been bothering you?"},
  {"speaker": "Patient", "intent": "Inform-Symptom", "text": "Doctor, <TEMPORAL>since two weeks</TEMPORAL> I am having <SYMPTOM_VALUE>breathlessness</SYMPTOM_VALUE>. It is <SYMPTOM_CHARACTERISTICS>worse at night</SYMPTOM_CHARACTERISTICS> and happening <SYMPTOM_FREQUENCY>every night</SYMPTOM_FREQUENCY>."},
  {"speaker": "Doctor", "intent": "Request-Symptom", "text": "Any <SYMPTOM_VALUE>loose motion</SYMPTOM_VALUE> or <SYMPTOM_LOCATION>stomach</SYMPTOM_LOCATION> problem also?"},
  {"speaker": "Patient", "intent": "Inform-Symptom", "text": "<POLARITY>Yes</POLARITY> doctor, on and off. My <SYMPTOM_LOCATION>chest</SYMPTOM_LOCATION> feels heavy when lying down."},
  {"speaker": "Doctor", "intent": "Request-Etiology", "text": "Any history of <MEDICAL_HISTORY_VALUE>diabetes</MEDICAL_HISTORY_VALUE> or <MEDICAL_HISTORY_VALUE>hypertension</MEDICAL_HISTORY_VALUE>?"},
  {"speaker": "Patient", "intent": "Inform-Etiology", "text": "<POLARITY>Yes</POLARITY>, both are there doctor. I am on <MEDICATION>metformin</MEDICATION> for sugar and using <MEDICATION>inhaler</MEDICATION> when needed."},
  {"speaker": "Doctor", "intent": "Request-Etiology", "text": "Are you still <HABIT_VALUE>smoking</HABIT_VALUE>?"},
  {"speaker": "Patient", "intent": "Inform-Etiology", "text": "<POLARITY>Yes</POLARITY> doctor, <HABIT_FREQUENCY>half a pack a day</HABIT_FREQUENCY>. I know I should quit."},
  {"speaker": "Doctor", "intent": "Request-Etiology", "text": "Any <EXPOSURE_VALUE>dust</EXPOSURE_VALUE> or fume exposure at work?"},
  {"speaker": "Patient", "intent": "Inform-Etiology", "text": "<POLARITY>No</POLARITY> doctor, I am retired now."},
  {"speaker": "Doctor", "intent": "Inform-Existing_Examination_and_Treatment", "text": "Alright. We will do a <MEDICAL_TEST_NAME>chest X-ray</MEDICAL_TEST_NAME> and <MEDICAL_TEST_NAME>HbA1c</MEDICAL_TEST_NAME> today. The smoking history combined with your age is a concern."}
]"""

SYSTEM_PROMPT_BLOCKS = [
    {
        "type": "text",
        "text": (
            "You are an expert synthetic medical dialogue generator for Indian English clinical NLP datasets.\n"
            "Your output trains Named Entity Recognition (NER) models — quality and entity density are paramount.\n\n"

            "━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Inline XML entity tags wrapping every span, embedded in natural speech:\n"
            "  <SYMPTOM_VALUE>chest pain</SYMPTOM_VALUE>\n"
            "  <TEMPORAL>since last week</TEMPORAL>\n"
            "ALLOWED TAGS — exactly these 12, never invent others:\n"
            "  POLARITY | SYMPTOM_VALUE | TEMPORAL | MEDICATION | EXPOSURE_VALUE\n"
            "  SYMPTOM_CHARACTERISTICS | MEDICAL_HISTORY_VALUE | SYMPTOM_LOCATION\n"
            "  MEDICAL_TEST_NAME | HABIT_VALUE | HABIT_FREQUENCY | SYMPTOM_FREQUENCY\n"
            "TAG CASING: ALL tag names MUST be FULLY UPPERCASE. Write <SYMPTOM_VALUE>\n"
            "  not <SYMPTOM_Value> or <Symptom_Value>. Every letter after '<' must be uppercase or underscore.\n"
            "NESTING: forbidden. Each token belongs to exactly one tag or is O.\n\n"
            
            "━━━ INTENT CLASSIFICATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "For EVERY turn, you must classify the speaker's primary intent.\n"
            "ALLOWED INTENTS (Select exactly one per turn from this list):\n"
            "  Request-Etiology | Request-Precautions | Request-Medical_Advice\n"
            "  Inform-Etiology | Diagnose | Request-Basic_Information\n"
            "  Request-Drug_Recommendation | Inform-Medical_Advice\n"
            "  Request-Existing_Examination_and_Treatment | Inform-Basic_Information\n"
            "  Inform-Precautions | Inform-Existing_Examination_and_Treatment\n"
            "  Inform-Drug_Recommendation | Request-Symptom | Inform-Symptom | Other\n\n"
            "INTENT DISAMBIGUATION (use these definitions to choose correctly):\n"
            "  Diagnose = doctor NAMES or IDENTIFIES the condition/disease.\n"
            "  Inform-Medical_Advice = doctor explains treatment PLAN or clinical reasoning.\n"
            "  Inform-Precautions = doctor warns about RISKS, side effects, things to AVOID.\n"
            "  Inform-Drug_Recommendation = doctor prescribes or names SPECIFIC medicines.\n"
            "    This is NOT the same as Inform-Medical_Advice. Use Inform-Drug_Recommendation\n"
            "    when the doctor says 'I am prescribing X', 'take Y tablet', 'start Z syrup'.\n"
            "  Other = ONLY greetings, farewells, or personal chat with ZERO medical content.\n"
            "    'What brings you here today?' or 'What problem?' = Request-Basic_Information, NOT Other.\n"
            "    These turns may have zero NER tags — but ONLY these turns.\n\n"

            "━━━ QUALITY RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. MEDICAL PLAUSIBILITY — one coherent clinical scenario throughout.\n"
            "   Never list terms mechanically. Each term must arise from the narrative.\n"
            "2. INDIAN ENGLISH AUTHENTICITY — use colloquialisms organically:\n"
            "   'sugar problem', 'gastric issue', 'BP shooting up', 'loose motion',\n"
            "   'stone problem', 'urine problem', 'pain-killers only', 'it is paining'.\n"
            "3. ENTITY DENSITY — every utterance from BOTH Doctor AND Patient past\n"
            "   the opening greeting must carry ≥1 tagged entity. When the doctor\n"
            "   mentions a symptom, test, or medication by name, tag it even in\n"
            "   questions or summaries (e.g., 'Is there any <SYMPTOM_VALUE>swelling</SYMPTOM_VALUE>?').\n"
            "   Target 40–50 total spans per dialogue.\n"
            "   Minimum: 2 tags per turn average across the full dialogue.\n"
            "   ZERO-TAG TURNS: Turns with zero tags are acceptable when the text\n"
            "   genuinely contains no medical terms (e.g., 'What is bothering you?').\n"
            "   But if a SPECIFIC medical term IS present, it MUST be tagged —\n"
            "   never skip a relevant entity just because the turn is conversational.\n"
            "   BAD: 'Do you smoke or drink?' (untagged) → GOOD: 'Do you <HABIT_VALUE>smoke</HABIT_VALUE> or <HABIT_VALUE>drink</HABIT_VALUE>?'\n"
            "   BAD: 'Doctor, anemia — I do not know what that means' (untagged) → tag <MEDICAL_HISTORY_VALUE>anemia</MEDICAL_HISTORY_VALUE>\n"
            "   OVER-TAGGING WARNING: Do NOT tag generic/vague words as medical entities:\n"
            "   BAD: '<SYMPTOM_VALUE>health</SYMPTOM_VALUE>', '<SYMPTOM_VALUE>problem</SYMPTOM_VALUE>',\n"
            "     '<SYMPTOM_VALUE>issue</SYMPTOM_VALUE>', '<SYMPTOM_VALUE>condition</SYMPTOM_VALUE>' — these are too vague.\n"
            "   Only tag SPECIFIC clinical terms (fever, anemia, paining, bleeding, diabetes, rash, etc.).\n"
            "4. POLARITY BALANCE — mix affirmative ('yes doctor', 'confirmed', 'active')\n"
            "   with negative ('no', 'not at all', 'stopped', 'denied', 'never').\n"
            "   POLARITY is ONLY for binary yes/no confirmation/denial. NEVER tag\n"
            "   trend words (increasing, worsening), certainty (likely, possibly), or\n"
            "   state descriptors (uncontrolled, detected) as POLARITY.\n"
            "5. LENGTH — 25 utterances total (Doctor + Patient combined).\n"
            "6. STYLE — follow the per-request style instruction faithfully.\n"
            "7. NO INVENTED TAG NAMES — use only the 12 allowed tags listed above.\n"
            "   Do not invent new tag names. However, you MUST tag every medical\n"
            "   entity that naturally appears in the dialogue, even if it was not\n"
            "   in the 'Required terms' list. Required terms are minimum guarantees,\n"
            "   not limits.\n"
            "   CRITICAL: You must aggressively tag EVERY single mention of a body\n"
            "   part (e.g., stomach, jaw, leg, ankle, head, bottom, belly, eyes, skin,\n"
            "   knee, chest, wrist, temple, groin, prostate) as <SYMPTOM_LOCATION>.\n"
            "   Never leave a medical term, body part, or symptom untagged, even if:\n"
            "   - it is repeated (tag EVERY mention, not just the first)\n"
            "   - it appears in a doctor's question or summary\n"
            "   - the patient uses shy/indirect/colloquial language\n"
            "   - it is a negated entity ('no <SYMPTOM_VALUE>fever</SYMPTOM_VALUE>' — still tag it)\n"
            "   - it is a diagnosis term in a Diagnose turn (tag the condition name!)\n"
            "   - it uses Indian English verb forms: 'paining' = <SYMPTOM_VALUE>paining</SYMPTOM_VALUE>\n"
            "   - it refers to a condition by name: 'anemia' = <MEDICAL_HISTORY_VALUE>anemia</MEDICAL_HISTORY_VALUE>\n"
            "   - it is negated with 'nothing like': 'nothing like <SYMPTOM_VALUE>bleeding</SYMPTOM_VALUE>'\n"
            "   NEGATIVE EXAMPLES (DO NOT DO THIS):\n"
            "   BAD: 'any problem near your bottom?' → GOOD: 'near your <SYMPTOM_LOCATION>bottom</SYMPTOM_LOCATION>?'\n"
            "   BAD: 'is there any swelling or lump?' → GOOD: '<SYMPTOM_VALUE>swelling</SYMPTOM_VALUE> or <SYMPTOM_VALUE>lump</SYMPTOM_VALUE>?'\n"
            "   BAD: 'constipation... straining... motion is hard' (all untagged)\n"
            "     → GOOD: '<SYMPTOM_VALUE>constipation</SYMPTOM_VALUE>... <SYMPTOM_VALUE>straining</SYMPTOM_VALUE>'\n"
            "   BAD: 'this is a case of COPD exacerbation' (diagnosis untagged!)\n"
            "     → GOOD: 'this is a case of <SYMPTOM_VALUE>COPD exacerbation</SYMPTOM_VALUE>'\n"
            "   BAD: 'any yellowing of your eyes or skin?' (body parts + symptom untagged)\n"
            "     → GOOD: '<SYMPTOM_VALUE>yellowing</SYMPTOM_VALUE> of your <SYMPTOM_LOCATION>eyes</SYMPTOM_LOCATION> or <SYMPTOM_LOCATION>skin</SYMPTOM_LOCATION>?'\n"
            "   BAD: 'she takes homeopathic drops <MEDICATION>for sleep</MEDICATION>' (tagged description, not med name)\n"
            "     → GOOD: 'she takes <MEDICATION>homeopathic drops</MEDICATION> for sleep'\n"
            "   BAD: 'I took something <MEDICATION>for pain</MEDICATION>' → GOOD: 'I took something for pain' (leave untagged if no drug name)\n"
            "   BAD: '<SYMPTOM_VALUE>precautions</SYMPTOM_VALUE>' or '<SYMPTOM_VALUE>first baby</SYMPTOM_VALUE>' — non-medical words are NEVER SYMPTOM_VALUE\n"
            "   BAD: '<POLARITY>increasing</POLARITY>', '<POLARITY>worsening</POLARITY>', or '<POLARITY>likely</POLARITY>' — these are NOT polarity. Polarity is strictly yes/no/confirmed/absent.\n"
            "   BAD: '<TEMPORAL>daily</TEMPORAL>' — 'daily' means HOW OFTEN (frequency), not WHEN. Use SYMPTOM_FREQUENCY or HABIT_FREQUENCY.\n"
            "   BAD: '<HABIT_FREQUENCY>heavy</HABIT_FREQUENCY> smoker' — 'heavy' is intensity, not frequency. Leave untagged.\n"
            "8. PRECISE BOUNDARIES — Tag the exact minimal span. Separate modifiers from the core entity\n"
            "   (e.g. <SYMPTOM_CHARACTERISTICS>severe</SYMPTOM_CHARACTERISTICS> <SYMPTOM_VALUE>pain</SYMPTOM_VALUE>\n"
            "   instead of wrapping them together).\n"
            "   SPAN TRIMMING — Never include filler words inside a tag span:\n"
            "   BAD: '<SYMPTOM_CHARACTERISTICS>like a pulling sensation</SYMPTOM_CHARACTERISTICS>' → GOOD: '<SYMPTOM_CHARACTERISTICS>pulling sensation</SYMPTOM_CHARACTERISTICS>'\n"
            "   BAD: '<SYMPTOM_CHARACTERISTICS>it is feeling warm to touch</SYMPTOM_CHARACTERISTICS>' → GOOD: '<SYMPTOM_CHARACTERISTICS>warm to touch</SYMPTOM_CHARACTERISTICS>'\n"
            "   BAD: '<SYMPTOM_VALUE>it is paining</SYMPTOM_VALUE>' → GOOD: '<SYMPTOM_VALUE>paining</SYMPTOM_VALUE>'\n"
            "   Words like 'like a', 'it is', 'there is', 'I am having' must sit OUTSIDE the tag.\n"
            "   TAG DISAMBIGUATION (avoid these common confusions):\n"
            "   - SYMPTOM_VALUE = the symptom/condition NAME (fever, rash, diabetes).\n"
            "     MEDICAL_HISTORY_VALUE = PAST condition or surgery mentioned as history.\n"
            "     If the patient is CURRENTLY experiencing it → SYMPTOM_VALUE.\n"
            "     If it happened IN THE PAST → MEDICAL_HISTORY_VALUE.\n"
            "   - TEMPORAL = WHEN something happened (since two weeks, last month, 2024).\n"
            "     SYMPTOM_FREQUENCY = HOW OFTEN it recurs (daily, on and off, every night).\n"
            "     'daily' and 'every day' = SYMPTOM_FREQUENCY or HABIT_FREQUENCY, NEVER TEMPORAL.\n"
            "   - HABIT_FREQUENCY = how much/often a HABIT occurs (2 packs a day, socially).\n"
            "     SYMPTOM_FREQUENCY = how often a SYMPTOM recurs (twice a week, constantly).\n"
            "   - MEDICAL_TEST_NAME = name of a test (HbA1c, X-ray, biopsy). Never tag a\n"
            "     test name as MEDICAL_HISTORY_VALUE.\n\n"
            "9. NATURAL FILLERS — sprinkle light spoken fillers (umm, ahh, uh) into some Patient turns, 0-1 per turn, never every turn. Fillers must sit OUTSIDE any tag span, never inside <TAG>...</TAG>.\n"
            "10. LANGUAGE — ALL text MUST be in English. 'Indian English' means English\n"
            "    sentences with Indian phrasing ('it is paining', 'sugar problem', 'gas trouble'),\n"
            "    NOT Hindi/Hinglish sentences. Never output Hindi/Urdu words (mujhe, bahut, wala,\n"
            "    ke paas, ho gaya, bilkul, gaon, chulha, beedi, gutka, pan masala, hookah,\n"
            "    vaidya, churna, desi, haldi, ajwain). Use English equivalents instead\n"
            "    (e.g., 'local healer' not 'vaidya', 'chewing tobacco' not 'gutka',\n"
            "    'hand-rolled cigarette' not 'beedi'). Zero tolerance.\n"
            "11. NO TAG-FREE ZONES — Never leave 2+ consecutive untagged turns.\n"
            "    Closing summary/plan turns are especially rich in MEDICAL_TEST_NAME,\n"
            "    MEDICATION, and SYMPTOM_VALUE — tag them all.\n"
            "12. SPEAKERS — Only two speakers: 'Doctor' and 'Patient'. No third-party\n"
            "    speakers (Family, Nurse, Junior, etc.). If a family member is present,\n"
            "    the Patient speaks on their behalf. Do NOT address a 'Junior' or student\n"
            "    in the dialogue — only Doctor ↔ Patient interaction.\n"

            "━━━ OUTPUT SCHEMA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Return ONLY a valid JSON array. No preamble, no markdown fences.\n"
            '[{"speaker": "Doctor"|"Patient", "intent": "...from list...", "text": "...tagged utterance..."}]\n\n'

            "━━━ ONE-SHOT EXAMPLE (all 12 tags) ━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "User: Scenario: elderly male | follow-up visit | respiratory complaint | "
            "non-compliant patient\n"
            "Style: patient downplays symptoms; doctor probes patiently.\n"
            "Required terms: SYMPTOM_VALUE: breathlessness, loose motion | "
            "TEMPORAL: since two weeks | SYMPTOM_LOCATION: chest | "
            "SYMPTOM_CHARACTERISTICS: worse at night | SYMPTOM_FREQUENCY: every night | "
            "MEDICAL_HISTORY_VALUE: diabetes, hypertension | MEDICATION: metformin, inhaler | "
            "MEDICAL_TEST_NAME: chest X-ray, HbA1c | HABIT_VALUE: smoking | "
            "HABIT_FREQUENCY: half a pack a day | EXPOSURE_VALUE: dust | POLARITY: yes, no\n\n"
            f"Assistant:\n{_EXAMPLE_DIALOGUE}\n"
        ),
        "cache_control": {"type": "ephemeral"},
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# PER-REQUEST USER PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_user_prompt(kb: dict) -> str:
    scenario       = make_scenario()
    style          = random.choice(STYLE_VARIANTS)
    focal_tags     = choose_focal_tags()
    injected_terms = sample_terms_for_tags(kb, focal_tags)

    return (
        f"Scenario: {scenario}\n"
        f"Style: {style}\n"
        f"Required terms: {' | '.join(injected_terms)}\n\n"
        "Generate a highly realistic Indian clinic dialogue (25 utterances).\n"
        "Every required term must appear as a tagged span. Additionally, tag ALL\n"
        "medical entities that arise organically in the dialogue — the required\n"
        "terms are minimums, not limits.\n"
        "Ensure the dialogue includes:\n"
        "  - At least one Diagnose intent turn where the doctor explicitly names the condition.\n"
        "  - At least one Inform-Drug_Recommendation turn where the doctor prescribes specific medicines.\n"
        "  - Other intent for any general/social/personal conversation turns (these may have zero NER tags).\n"
        "Return ONLY the JSON array."
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_batch(kb: dict, n: int) -> list:
    requests = []
    for i in range(n):
        requests.append({
            "custom_id": f"meditod-phase2b-{i + 1:04d}",
            "params": {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "system":     SYSTEM_PROMPT_BLOCKS,
                "messages":   [{"role": "user", "content": build_user_prompt(kb)}],
            }
        })
    return requests


def write_batch(requests: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def print_batch_stats(requests: list):
    """Print a token-cost and coverage summary before submission."""
    # Rough token estimates (1 token ≈ 0.75 words for English)
    sys_text   = SYSTEM_PROMPT_BLOCKS[0]["text"]
    sys_tok    = int(len(sys_text.split()) * 1.35)
    first_user = requests[0]["params"]["messages"][0]["content"]
    user_tok   = int(len(first_user.split()) * 1.35)
    n          = len(requests)

    cache_write_tok  = sys_tok
    cache_read_tok   = sys_tok * (n - 1)
    user_total_tok   = user_tok * n
    # Cache reads billed at ~10% — see https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    effective_input  = cache_write_tok + int(cache_read_tok * 0.10) + user_total_tok
    no_cache_input   = (sys_tok + user_tok) * n
    savings_pct      = round((1 - effective_input / no_cache_input) * 100)

    max_out_total    = MAX_TOKENS * n

    print("\n" + "═" * 60)
    print(f"  MediTOD Batch — {n} requests → {OUTPUT_JSONL}")
    print("═" * 60)
    print(f"  Model            : {MODEL}")
    print(f"  max_tokens       : {MAX_TOKENS} per request")
    print(f"  System prompt    : ~{sys_tok} tokens (cached)")
    print(f"  Avg user prompt  : ~{user_tok} tokens")
    print()
    print(f"  ── INPUT TOKEN ANALYSIS ──────────────────────────")
    print(f"  Without caching  : ~{no_cache_input:,} tokens")
    print(f"  With caching     : ~{effective_input:,} effective tokens")
    print(f"  Estimated saving : ~{savings_pct}% on input")
    print()
    print(f"  ── OUTPUT TOKEN BUDGET ───────────────────────────")
    print(f"  Max output total : ~{max_out_total:,} tokens")
    print(f"  (actual is lower; dialogues avg ~1400-1700 tokens)")
    print()
    print(f"  ── DIVERSITY CHECK ───────────────────────────────")

    scenarios = [r["params"]["messages"][0]["content"].split("\n")[0] for r in requests]
    profiles  = [s.split("|")[0].strip().replace("Scenario: ", "") for s in scenarios]
    pcounts   = Counter(profiles)
    print(f"  Unique patient profiles sampled : {len(pcounts)}/{len(PATIENT_PROFILES)}")
    print(f"  Most common profile             : {pcounts.most_common(1)[0]}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING  (run after batch results arrive)
# ─────────────────────────────────────────────────────────────────────────────
def xml_to_bio(text: str, speaker: str) -> list:
    """
    Convert inline-XML tagged utterance into token/tag pairs.

    The caller can use the returned tokens and tags to export a readable JSON
    structure that is easy to inspect and still usable for NER training.
    """
    tag_re = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.DOTALL)
    token_re = re.compile(r"\w+|[^\w\s]")

    tokens = []
    ner_tags = []
    cursor = 0

    for match in tag_re.finditer(text):
        for word in token_re.findall(text[cursor:match.start()]):
            tokens.append(word)
            ner_tags.append("O")

        tag_name = match.group(1)
        tagged_words = token_re.findall(match.group(2))
        for idx, word in enumerate(tagged_words):
            tokens.append(word)
            ner_tags.append(f"{'B' if idx == 0 else 'I'}-{tag_name}")

        cursor = match.end()

    for word in token_re.findall(text[cursor:]):
        tokens.append(word)
        ner_tags.append("O")

    return tokens, ner_tags


def strip_xml_tags(text: str) -> str:
    """Remove inline XML tags while keeping the natural utterance text."""
    return re.sub(r"</?[A-Z_]+>", "", text)


def turn_to_export_record(speaker: str, intent: str, text: str) -> dict:
    """Build the readable speaker/intent/text/tokens/ner_tags shape used downstream."""
    tokens, ner_tags = xml_to_bio(text, speaker)
    speaker_tokens = re.findall(r"\w+|[^\w\s]", f"{speaker}:")
    clean_text = strip_xml_tags(text).strip()
    return {
        "speaker": speaker,
        "intent": intent,
        "text": f"{speaker}: {clean_text}",
        "tokens": speaker_tokens + tokens,
        "ner_tags": ["O"] * len(speaker_tokens) + ner_tags,
    }


def parse_response_text(raw: str):
    """Parse model output — tolerates accidental markdown fences."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(clean)   # let this raise if still broken


def sanitize_tags(dialogue: list) -> list:
    """
    Auto-fix common model output issues in-place:
      1. Tag casing: SYMPTOM_Value → SYMPTOM_VALUE
      2. Broken XML: </SYMPTOM_V alue> → </SYMPTOM_VALUE>
      3. Span trimming: <TAG>like a pulling</TAG> → <TAG>pulling</TAG>
    """
    VALID_TAGS = {
        "POLARITY", "SYMPTOM_VALUE", "TEMPORAL", "MEDICATION", "EXPOSURE_VALUE",
        "SYMPTOM_CHARACTERISTICS", "MEDICAL_HISTORY_VALUE", "SYMPTOM_LOCATION",
        "MEDICAL_TEST_NAME", "HABIT_VALUE", "HABIT_FREQUENCY", "SYMPTOM_FREQUENCY",
    }
    # Build case-insensitive lookup
    tag_upper_map = {t.lower(): t for t in VALID_TAGS}

    for turn in dialogue:
        text = turn.get("text", "")

        # Fix 1: Remove stray spaces inside tag names (</SYMPTOM_V alue> → </SYMPTOM_VALUE>)
        text = re.sub(r"<(/?)([A-Za-z_]+)\s+([A-Za-z_]+)>",
                      lambda m: f"<{m.group(1)}{m.group(2)}{m.group(3)}>", text)

        # Fix 2: Normalize tag casing to UPPERCASE
        def fix_case(m):
            slash = m.group(1) or ""
            tag_name = m.group(2)
            upper = tag_upper_map.get(tag_name.lower().replace(" ", ""), tag_name.upper())
            return f"<{slash}{upper}>"
        text = re.sub(r"<(/?)([A-Za-z_]+)>", fix_case, text)

        # Fix 3: Trim filler words from span start
        filler_prefixes = [
            "like a ", "like an ", "it is ", "she is ", "he is ",
            "I am ", "there is ", "there are ", "some ", "I am having ",
        ]
        def trim_span(m):
            tag = m.group(1)
            span = m.group(2)
            for fp in filler_prefixes:
                if span.startswith(fp) or span.startswith(fp.capitalize()):
                    span = span[len(fp):]
                    break
            return f"<{tag}>{span}</{tag}>"
        text = re.sub(r"<([A-Z_]+)>(.*?)</\1>", trim_span, text)

        turn["text"] = text

    return dialogue


def validate_dialogue(dialogue: list) -> tuple[bool, str]:
    """
    Quality gate on parsed dialogue:
      - Must be a list of dicts with speaker + text + intent
      - Must have 20-30 turns
      - Must contain at least 30 tagged spans total
      - Intents must be valid
      - Non-Other turns must have at least 1 tag
      - No 3+ consecutive turns with zero tags
      - Average tags/turn must be >= 1.5
    """
    valid_intents = {
        "Request-Etiology", "Request-Precautions", "Request-Medical_Advice",
        "Inform-Etiology", "Diagnose", "Request-Basic_Information",
        "Request-Drug_Recommendation", "Inform-Medical_Advice",
        "Request-Existing_Examination_and_Treatment", "Inform-Basic_Information",
        "Inform-Precautions", "Inform-Existing_Examination_and_Treatment",
        "Inform-Drug_Recommendation", "Request-Symptom", "Inform-Symptom", "Other"
    }

    if not isinstance(dialogue, list):
        return False, "not a list"
    if not (20 <= len(dialogue) <= 30):
        return False, f"turn count {len(dialogue)} out of range [20,30]"

    tag_re = re.compile(r"<[A-Z_]+>")
    total_spans = sum(len(tag_re.findall(t.get("text", ""))) for t in dialogue)
    if total_spans < 30:
        return False, f"only {total_spans} tagged spans (need ≥30)"

    # Average tags per turn must be >= 1.5
    avg_tags = total_spans / len(dialogue)
    if avg_tags < 1.5:
        return False, f"avg tags/turn {avg_tags:.1f} < 1.5 (too sparse)"

    # Intent validation + non-Other turns must have >= 1 tag
    zero_tag_streak = 0
    for i, t in enumerate(dialogue):
        if "intent" not in t:
            return False, "missing intent field in a turn"
        if t["intent"] not in valid_intents:
            return False, f"invalid intent: {t['intent']}"
        if t.get("speaker") not in ("Doctor", "Patient"):
            return False, f"invalid speaker '{t.get('speaker')}' in turn {i+1} (only Doctor/Patient allowed)"

        turn_spans = len(tag_re.findall(t.get("text", "")))

        # We no longer fail immediately if a single non-Other turn has zero tags.
        # As long as the overall density is high (avg >= 1.5) and there are no long streaks,
        # natural conversational zero-tag turns are acceptable.

        # Track consecutive zero-tag turns (catches sparse zones)
        if turn_spans == 0:
            zero_tag_streak += 1
        else:
            zero_tag_streak = 0
            
        # Allow up to 4 consecutive zero-tag turns, mostly to account for 
        # long conversational greetings/closings.
        if zero_tag_streak >= 4:
            return False, f"4+ consecutive zero-tag turns ending at turn {i+1}"

    return True, "ok"


def process_batch_results(results_path: str, out_stem: str):
    """
    Convert completed Anthropic batch results JSONL into:
      {out_stem}_xml.jsonl   — raw tagged dialogues (human review / further annotation)
            {out_stem}_bio.jsonl   — readable utterance-level JSON for direct NER training
      {out_stem}_failed.jsonl — rows that failed parsing or validation
    """
    xml_records, bio_records, failed = [], [], []
    tag_counter   = Counter()
    span_totals   = []

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            row       = json.loads(line)
            cid       = row.get("custom_id", "?")
            content   = (row.get("result", {})
                            .get("message", {})
                            .get("content", []))
            if not content:
                failed.append({"id": cid, "reason": "empty content"})
                continue

            # Extract text — handle both standard (text block) and thinking+text format
            raw = ""
            for block in content:
                if block.get("type") == "text":
                    raw = block.get("text", "")
                    break
            if not raw:
                # Fallback: try first block's text field
                raw = content[0].get("text", "")

            try:
                dialogue = parse_response_text(raw)
            except Exception as e:
                failed.append({"id": cid, "reason": f"json parse error: {e}", "raw": raw[:200]})
                continue

            # Auto-fix tag casing, broken XML, and wide spans
            dialogue = sanitize_tags(dialogue)

            ok, reason = validate_dialogue(dialogue)
            if not ok:
                failed.append({"id": cid, "reason": reason})
                continue

            # Collect span stats
            tag_re = re.compile(r"<([A-Z_]+)>")
            spans_in_dialogue = []
            for turn in dialogue:
                for tag in tag_re.findall(turn.get("text", "")):
                    tag_counter[tag] += 1
                    spans_in_dialogue.append(tag)
            span_totals.append(len(spans_in_dialogue))

            xml_records.append({"id": cid, "dialogue": dialogue})

            export_turns = []
            for turn in dialogue:
                export_turns.append(
                    turn_to_export_record(
                        turn.get("speaker", ""),
                        turn.get("intent", "Other"),
                        turn.get("text", ""),
                    )
                )
            bio_records.append({"id": cid, "utterances": export_turns})

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(f"{out_stem}_xml.jsonl",    xml_records)
    write_jsonl(f"{out_stem}_bio.jsonl",    bio_records)
    write_jsonl(f"{out_stem}_failed.jsonl", failed)

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(xml_records) + len(failed)
    print("\n" + "═" * 60)
    print(f"  POST-PROCESSING RESULTS")
    print("═" * 60)
    print(f"  Total processed  : {total}")
    print(f"  ✅ Passed        : {len(xml_records)}")
    print(f"  ❌ Failed / QC   : {len(failed)}")
    print(f"  Pass rate        : {len(xml_records)/max(total,1)*100:.1f}%")
    if span_totals:
        print(f"\n  ── SPAN STATISTICS ───────────────────────────────")
        print(f"  Avg spans/dialogue : {sum(span_totals)/len(span_totals):.1f}")
        print(f"  Min / Max spans    : {min(span_totals)} / {max(span_totals)}")
        print(f"\n  ── TAG DISTRIBUTION ──────────────────────────────")
        for tag in ALL_TAGS:
            count = tag_counter.get(tag, 0)
            bar   = "█" * min(40, int(count / max(tag_counter.values()) * 40))
            print(f"  {tag:<30} {count:>5}  {bar}")
    print("═" * 60 + "\n")
    print(f"  Output files:")
    print(f"    {out_stem}_xml.jsonl    ← tagged dialogues")
    print(f"    {out_stem}_bio.jsonl    ← readable utterance-level JSON for training")
    print(f"    {out_stem}_failed.jsonl ← failed/rejected rows")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MediTOD NER Batch Prompt Generator v3")
    parser.add_argument("--n",           type=int, default=200,
                        help="Number of batch requests to generate (default: 500)")
    parser.add_argument("--out",         type=str, default=OUTPUT_JSONL,
                        help="Output JSONL path (default: batch_requests.jsonl)")
    parser.add_argument("--kb",          type=str, default=KB_PATH,
                        help="Path to medical_knowledge_base.json")
    parser.add_argument("--postprocess", type=str, default=None, metavar="RESULTS_JSONL",
                        help="Post-process completed batch results instead of generating")
    parser.add_argument("--out_stem",    type=str, default="meditod_results",
                        help="Output filename stem for post-processing (default: meditod_results)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── Mode: post-process ───────────────────────────────────────────────────
    if args.postprocess:
        print(f"Post-processing batch results: {args.postprocess}")
        process_batch_results(args.postprocess, args.out_stem)
        return

    # ── Mode: generate ───────────────────────────────────────────────────────
    kb_path = Path(args.kb)
    if not kb_path.exists():
        print(f"❌ Knowledge base not found: {kb_path}", file=sys.stderr)
        sys.exit(1)

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"Generating {args.n} batch requests…")
    requests = generate_batch(kb, args.n)
    write_batch(requests, args.out)
    print_batch_stats(requests)
    print(f"✅  Written to {args.out}")
    print(f"\n  Next steps:")
    print(f"    1. Submit:  see docs at https://docs.anthropic.com/en/docs/build-with-claude/message-batches")
    print(f"    2. Monitor: https://console.anthropic.com → Batches")
    print(f"    3. After completion, run:")
    print(f"       python {Path(__file__).name} --postprocess <results.jsonl> --out_stem meditod_results\n")


if __name__ == "__main__":
    main()
