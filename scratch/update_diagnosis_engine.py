import re

with open("pipeline/diagnosis_inference.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update regexes in _parse_llm_response
old_regex_block = '''        p_match = re.search(
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
        )'''

new_regex_block = '''        p_match = re.search(
            r'(?:primary diagnosis|most likely diagnosis|probable diagnosis)[^\\n:.]*?\\b(?:is|could be|would be|:)\\s*([A-Za-z0-9\\s\\(\\)\\-\\/]+?)(?=\\.|\\n|\\s*(?:secondary|clinical reasoning|differential))',
            cleaned,
            re.IGNORECASE
        )
        s_match = re.search(
            r'(?:secondary diagnoses?|differential diagnoses?|differentials?)[^\\n:.]*?\\b(?:could include|would include|include|are|:)\\s*(.+?)(?=\\s+as\\s+they|\\.\\s*(?:however|for|clinical|recommended)|\\.\\n|\\n\\n|\\Z|$)',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )
        r_match = re.search(
            r'(?:for\\s+clinical reasoning|clinical reasoning|clinical justification)(?:\\s*,\\s*|\\s*:\\s*|\\s+is\\s+)(.+?)(?=\\.\\s*recommended tests|\\n\\n|recommended tests|\\Z)',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )
        t_match = re.search(
            r'(?:recommended tests?|diagnostic investigations?|further investigations?)[^\\n:.]*?\\b(?:include|could include|are|suggested are|:)\\s*(.+?)(?=<\\||\\n\\n|\\Z|$)',
            cleaned,
            re.IGNORECASE | re.DOTALL
        )'''

text = text.replace(old_regex_block, new_regex_block)
print("Updated regex block in _parse_llm_response")

# 2. Update keyword definitions in _infer_clinical_diagnosis
old_kw = '''        has_blood_stool = any(k in combined_text for k in ["blood", "bleeding", "bloody stool", "red stool", "mucus in stool", "hematochezia", "melaena", "black stool"])
        has_hematemesis = any(k in combined_text for k in ["vomiting blood", "blood in vomit", "hematemesis", "coffee ground", "coffee-ground"])'''

new_kw = '''        has_blood_in_stool = any(k in combined_text for k in ["bloody stool", "red stool", "blood in stool", "mucus in stool", "hematochezia", "melaena", "melena", "black stool", "tarry stool"])
        has_hematemesis = any(k in combined_text for k in ["vomiting blood", "blood in vomit", "vomited blood", "hematemesis", "coffee ground", "coffee-ground"])'''

text = text.replace(old_kw, new_kw)

old_rule1 = '''        # 1. Hematemesis / Acute Upper GI Bleeding
        if has_hematemesis or (has_vomiting and has_blood_stool):'''

new_rule1 = '''        # 1. Hematemesis / Acute Upper GI Bleeding
        if has_hematemesis or (has_vomiting and has_blood_in_stool):'''

text = text.replace(old_rule1, new_rule1)

old_rule2 = '''        # 2. Bloody Diarrhea / Tenesmus / Mucus -> Inflammatory Bowel Disease / Ulcerative Colitis
        if has_blood_stool and has_diarrhea:'''

new_rule2 = '''        # 2. Bloody Diarrhea / Tenesmus / Mucus -> Inflammatory Bowel Disease / Ulcerative Colitis
        if has_blood_in_stool and has_diarrhea:'''

text = text.replace(old_rule2, new_rule2)

old_rule3 = '''        # 3. Severe Anemia / Occult Bleeding (without overt hematemesis)
        if has_anemia and (has_blood_stool or "black" in combined_text or "pallor" in combined_text):'''

new_rule3 = '''        # 3. Severe Anemia / Occult Bleeding (without overt hematemesis)
        if has_anemia and (has_blood_in_stool or "black" in combined_text or "pallor" in combined_text):'''

text = text.replace(old_rule3, new_rule3)

old_rule8 = '''        # 8. Burning Chest / Epigastric Pain with Regurgitation / Long duration (Process 2, 3)
        if has_gas_acidity and (has_abd_pain or "chest" in combined_text or "burning" in combined_text):'''

new_rule8 = '''        # 8. Burning Chest / Epigastric Pain with Regurgitation / Long duration (Process 2, 3)
        if (has_gas_acidity or "burning" in combined_text) and (has_abd_pain or "chest" in combined_text or has_vomiting or "sleep" in combined_text):'''

text = text.replace(old_rule8, new_rule8)

with open("pipeline/diagnosis_inference.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Saved pipeline/diagnosis_inference.py successfully.")

