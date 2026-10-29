import sys
from pathlib import Path
sys.path.insert(0, "pipeline")
from pipeline import PipelineProcessor

proc = PipelineProcessor.__new__(PipelineProcessor)

# Test 1: Doctor question screening for thyroid (Process 3 failure mode)
doc_screening_question = {
    "sentence_id": "sent_0027",
    "speaker": "Doctor",
    "sentence": "Any other illnesses—like sugar, blood pressure, thyroid, or anything else?",
    "NER_tags": [{"entity": "thyroid", "type": "B-MEDICAL_HISTORY_VALUE"}],
    "DAC_tag": "request-symptom"
}
pmh_result = proc.check_past_medical_history_condition(doc_screening_question)
print(f"Test 1 - Doctor thyroid screening question into PMH: {pmh_result} (Expected: False)")
assert pmh_result is False, "Doctor screening question should NOT be added to PMH!"

# Test 2: Doctor asking chief complaint (Process 3)
doc_cc_question = {
    "sentence_id": "sent_0001",
    "speaker": "Doctor",
    "sentence": "please tell me whats the problem ?",
    "NER_tags": [],
    "DAC_tag": "request-basic_information"
}
cc_result = proc.check_chief_complaint_condition(doc_cc_question)
print(f"Test 2 - Doctor question into CC: {cc_result} (Expected: False)")
assert cc_result is False, "Doctor question should NOT be added to CC!"

# Test 3: Patient presenting burning sensation (Process 3)
patient_symptom = {
    "sentence_id": "sent_0002",
    "speaker": "Patient",
    "sentence": "burning sensation in the stomach",
    "NER_tags": [
        {"entity": "burning sensation", "type": "B-SYMPTOM_VALUE"},
        {"entity": "stomach", "type": "B-SYMPTOM_LOCATION"}
    ],
    "DAC_tag": "inform-symptom"
}
cc_result3 = proc.check_chief_complaint_condition(patient_symptom)
print(f"Test 3 - Patient symptom into CC: {cc_result3} (Expected: True)")
assert cc_result3 is True, "Patient symptom SHOULD be added to CC!"

# Test 4: Patient presenting 2018 rheumatoid arthritis (Process 1 temporal leakage)
patient_past_ra = {
    "sentence_id": "sent_0004",
    "speaker": "Patient",
    "sentence": "In 2018, when my mother had knee pain, I took her to a medicine doctor.",
    "NER_tags": [
        {"entity": "2018", "type": "B-TEMPORAL"},
        {"entity": "knee pain", "type": "B-SYMPTOM_VALUE"}
    ],
    "DAC_tag": "inform-symptom"
}
cc_result4 = proc.check_chief_complaint_condition(patient_past_ra)
print(f"Test 4 - 2018 knee pain into CC: {cc_result4} (Expected: False due to temporal filter)")
assert cc_result4 is False, "Distant 2018 past event should NOT be in chief complaint!"

pmh_result4 = proc.check_past_medical_history_condition(patient_past_ra)
print(f"Test 4 - 2018 knee pain into PMH: {pmh_result4} (Expected: True)")
assert pmh_result4 is True, "Distant 2018 past event SHOULD be in past medical history!"

# Test 5: Patient active severe constipation (Process 1)
patient_constipation = {
    "sentence_id": "sent_0015",
    "speaker": "Patient",
    "sentence": "She has severe constipation—meaning she isn't able to pass stool, and it happens only once every 2–3 days",
    "NER_tags": [
        {"entity": "severe constipation", "type": "B-SYMPTOM_VALUE"},
        {"entity": "stool", "type": "B-SYMPTOM_VALUE"}
    ],
    "DAC_tag": "inform-symptom"
}
cc_result5 = proc.check_chief_complaint_condition(patient_constipation)
print(f"Test 5 - Active severe constipation into CC: {cc_result5} (Expected: True)")
assert cc_result5 is True, "Active severe constipation SHOULD be in chief complaint!"

print("\nALL 5 PIPELINE CONDITION TESTS PASSED!")

