import sys
sys.path.insert(0, "pipeline")
from pipeline import PipelineProcessor

proc = PipelineProcessor.__new__(PipelineProcessor)

sentences = [
    "tell me what the problem is??",
    "I'm telling you—I'll share my mother's history a bit, please listen",
    "In 2018, when my mother had knee pain, I took her to a medicine doctor.",
    "After running tests, he said that she has rheumatoid arthritis",
    "okay, 2018?",
    "In 2018, she was on medication for that.",
    "Then after some time—about a year later—her bleeding became heavy, her menstrual bleeding.",
    "That bleeding that happens during menstruation",
    "So we showed it to the doctor then, and he referred us to get a USG done.",
    "After the USG, it was found that there is a stone in the GB",
    "So, since there was a stone, at that time I was in Kalyani, so I showed the report at AIIMS.",
    "So we showed it to the doctor, and he said that since there is no pain or any other symptoms right now, there is no need for an operation at this time",
    "She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days",
    "She is having gas."
]

for s in sentences:
    # mock result dict
    res = {
        "speaker": "Patient" if not s.endswith("?") else "Doctor",
        "sentence": s,
        "NER_tags": [{"entity": "symptom", "type": "B-SYMPTOM_VALUE"}],
        "DAC_tag": "inform-symptom"
    }
    is_cc = proc.check_chief_complaint_condition(res)
    print(f"[{'CC ' if is_cc else 'SKIP'}] {s[:65]}")

