import re

text = """The patient's chief complaint includes knee pain, bleeding, and gas. The history of present illness spans from 2018 to the present day with intermittent constipation. Past medical history includes rheumatoid arthritis and a gallbladder stone identified via ultrasonography in 2024. No further tests have been conducted for the stomach.   Based on these findings, the primary diagnosis is Gallstones (Cholelithiasis). The secondary diagnoses could include Rheumatoid Arthritis and Constipation. The clinical reasoning suggests that the patient may be experiencing symptoms related to their gallbladder"""

p_match = re.search(r'(?:primary diagnosis\s*(?::|is)\s*)([^\.\n]+)', text, re.IGNORECASE)
s_match = re.search(r'(?:secondary diagnoses?\s*(?::|could include|include|are)\s*)([^\.\n]+)', text, re.IGNORECASE)
r_match = re.search(r'(?:clinical reasoning\s*(?::|suggests that|indicates that)\s*)(.+)', text, re.IGNORECASE | re.DOTALL)

print("Primary:", p_match.group(1).strip() if p_match else None)
print("Secondary:", s_match.group(1).strip() if s_match else None)
print("Reasoning:", r_match.group(1).strip() if r_match else None)
