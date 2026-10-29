import sys
from pathlib import Path

BASE = Path(r"c:\Users\ammua\Desktop\final_prompt")
sys.path.insert(0, str(BASE / "pipeline"))

from dac_inference import DACInference
from ner_inference import NERInference

print("Testing DACInference with deployed DAC v3...")
dac = DACInference(str(BASE / "pipeline" / "models" / "dac"))
tag1 = dac.predict("I have been suffering from severe burning sensation in chest", history=[], speaker="patient")
tag2 = dac.predict("Is everything else normal, your appetite and bowel movements?", history=[], speaker="doctor")
print(f"  Sentence 1 DAC Tag: {tag1}")
print(f"  Sentence 2 DAC Tag: {tag2}")

print("\nTesting NERInference with deployed NER v2...")
ner = NERInference(str(BASE / "pipeline" / "models" / "ner"))
entities1 = ner.predict("I have severe burning sensation in my stomach and nausea", history=[])
print(f"  Sentence 1 Entities: {entities1}")

print("\n[OK] BOTH MODELS LOADED AND INFERRED SUCCESSFULLY!")

