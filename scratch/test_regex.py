import re

def clean_dx(p):
    p = re.sub(r"^(?:suggestive of|indicative of|consistent with|concerning for|a possible|possible|likely|suspected)\s+", "", p.strip(), flags=re.IGNORECASE)
    p = re.sub(r"^(?:the\s+)?(?:most\s+likely\s+)?(?:condition|diagnosis|etiology)\s+(?:is|could be|would be|to be|:)\s*", "", p, flags=re.IGNORECASE).strip()
    p = re.sub(r"^(?:a\s+)?primary\s+diagnosis\s+(?:of|is|:)\s*", "", p, flags=re.IGNORECASE).strip()
    p = re.split(r",\s*which\s+is|\.\s+|\s+due\s+to\s+the|\s+as\s+they", p, flags=re.IGNORECASE)[0].strip()
    p = p.rstrip(".,;:")
    words = p.split()
    if words and words[0].islower():
        p = p.title()
    return p

test_texts = {
    "Case 5": "The patient presents with complaints of burning sensation, pain, and decreased appetite for approximately 6-7 months. The patient also reports that after eating, there is a feeling of heaviness and tightness in the affected area. Given these symptoms, it is important to consider several potential diagnoses. One possibility is gastroesophageal reflux disease (GERD), which can cause burning sensation and discomfort in the chest or throat. Another consideration is peptic ulcer disease (PUD), which may present with similar symptoms of pain and loss of appetite. Additionally, gastritis could also be a potential diagnosis due to its association with abdominal pain and changes in appetite. However, further investigation such as endoscopy will be necessary to confirm the exact cause of these symptoms.",
    "Case 8": "The patient presents with symptoms of malabsorption leading to flatulence upon ingestion of food, anemia requiring blood transfusion, and general weakness. The differential diagnosis includes conditions such as celiac disease, tropical sprue, and Whipple's disease. Further investigation is warranted to confirm the diagnosis.",
    "Case 11": "The patient's chief complaint of feeling heavy on one side near the flank and decreased appetite is suggestive of acute pancreatitis. This condition can cause severe abdominal pain that radiates to the back..."
}

for name, text in test_texts.items():
    match = re.search(
        r"(?:suggestive of|suggests? (?:a possible)?|indicative of|consistent with|concerning for|diagnosis of|diagnosed with|one possibility is|another consideration is|most likely (?:condition|diagnosis|cause|etiology) is)\s+([A-Za-z0-9\s\(\)\-\/]{4,50}?)(?:\.|\,|\n|;|as\b|which\b)",
        text,
        re.IGNORECASE
    )
    if not match:
        match = re.search(
            r"(?:differential diagnosis includes|differentials? include)\s+(?:conditions? such as\s+)?([A-Za-z0-9\s\(\)\-\/]{4,50}?)(?:\.|\,|\n|;|and\b)",
            text,
            re.IGNORECASE
        )
    if match:
        print(f"{name}: MATCH -> '{clean_dx(match.group(1))}'")
    else:
        print(f"{name}: NO MATCH")

