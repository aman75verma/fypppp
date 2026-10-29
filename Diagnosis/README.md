# MediTOD Clinical AI Diagnosis Dossier (11 Benchmark Cases)

**Target Audience:** Clinical Review Board, Internal Medicine Physicians & Gastroenterologists  
**System Version:** PubMedBERT NER v2 + PubMedBERT DAC v3 + OpenBioLLM-8B (Local Inference)  
**Execution Timestamp:** October 2026  
**Diagnostic Concordance:** 100% (All 11 patient cases verified)

---

## Folder Organization

This dossier contains 11 dedicated case subfolders representing real-world gastroenterology and general medicine consultations:

```
doctor_clinical_evaluation_dossier/
├── Clinical_Doctor_Evaluation_Report.pdf  <-- Complete Executive PDF for Physician Review
├── README.md                              <-- This navigation guide
├── Case_01_Cholelithiasis/
│   ├── transcript.txt                     <-- Complete patient-doctor consultation transcript
│   ├── ner_dac_tags.json                  <-- Sentence-level NER spans and Dialogue Act Classification (v3)
│   ├── medical_report.json                <-- De-contaminated Chief Complaint and isolated Past History
│   ├── annotated_report.txt               <-- Clinician-friendly plain text summary chart
│   ├── ground_truth_diagnosis.json        <-- Reference standard clinical diagnosis
│   └── openbiollm_diagnosis.json          <-- Live OpenBioLLM-8B predicted diagnosis & workup
├── Case_02_GERD_Recurrent/
...
└── Case_11_Pancreatitis/
```

---

## Ontology & Architecture Summary

* **NER Model:** PubMedBERT fine-tuned on medical transcripts with frozen lower representation layers (Strict Span Entity F1: 0.7944). Isolates Symptoms, Durations, Locations, Tests, and Prior Conditions.
* **DAC Model:** PubMedBERT fine-tuned with Class-Balanced Focal Loss & Deep 2-Layer MLP Head with explicit `[DOCTOR]` and `[PATIENT]` tokens (Macro F1: 0.8178).
* **Clinical De-Contamination Engine:** Filters screening queries asked by doctors (e.g., *"any sugar or pressure?"*) from polluting patient disease records, and prevents resolved past surgeries/tests from appearing as the active Chief Complaint.
* **Diagnostic Model:** OpenBioLLM-8B executing locally with 1024-token capacity and biomedical reasoning prompt.
