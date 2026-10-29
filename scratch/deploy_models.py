import os
import shutil
import zipfile
from pathlib import Path

BASE = Path(r"c:\Users\ammua\Desktop\final_prompt")
PIPELINE_MODELS = BASE / "pipeline" / "models"
MODELS_V1 = BASE / "pipeline" / "models_v1"
MODELS_V2 = BASE / "pipeline" / "models_v2"
MODELS_V3 = BASE / "pipeline" / "models_v3"

os.makedirs(MODELS_V1 / "dac", exist_ok=True)
os.makedirs(MODELS_V1 / "ner", exist_ok=True)
os.makedirs(MODELS_V2 / "dac", exist_ok=True)
os.makedirs(MODELS_V2 / "ner", exist_ok=True)
os.makedirs(MODELS_V3 / "dac", exist_ok=True)
os.makedirs(MODELS_V3 / "ner", exist_ok=True)

# 1. Backup current pipeline models (which were v1) into models_v1
if (PIPELINE_MODELS / "dac" / "best_dac_model.pt").exists():
    for f in (PIPELINE_MODELS / "dac").glob("*"):
        if f.is_file():
            shutil.copy2(f, MODELS_V1 / "dac" / f.name)
if (PIPELINE_MODELS / "ner" / "best_model.pt").exists():
    for f in (PIPELINE_MODELS / "ner").glob("*"):
        if f.is_file():
            shutil.copy2(f, MODELS_V1 / "ner" / f.name)
print("[OK] Backed up v1 models to pipeline/models_v1")

# 2. Populate models_v2
for f in (BASE / "DAC" / "v2_output" / "model").glob("*"):
    if f.is_file():
        shutil.copy2(f, MODELS_V2 / "dac" / f.name)
for f in (BASE / "NER" / "v2_output" / "model").glob("*"):
    if f.is_file():
        shutil.copy2(f, MODELS_V2 / "ner" / f.name)
print("[OK] Populated pipeline/models_v2")

# 3. Extract v3 from zip files into models_v3
z_dac = zipfile.ZipFile(BASE / "DAC" / "v3" / "v3_output.zip")
for member in z_dac.namelist():
    if member.startswith("model/") and not member.endswith("/"):
        fname = os.path.basename(member)
        with open(MODELS_V3 / "dac" / fname, "wb") as out:
            out.write(z_dac.read(member))

z_ner = zipfile.ZipFile(BASE / "NER" / "v3" / "v3_output.zip")
for member in z_ner.namelist():
    if member.startswith("model/") and not member.endswith("/"):
        fname = os.path.basename(member)
        with open(MODELS_V3 / "ner" / fname, "wb") as out:
            out.write(z_ner.read(member))
print("[OK] Extracted v3 models to pipeline/models_v3")

# 4. Deploy to active pipeline/models:
# DAC: Deploy v3 (highest Macro F1: 0.8178, best rare act recognition)
for f in (MODELS_V3 / "dac").glob("*"):
    if f.is_file():
        shutil.copy2(f, PIPELINE_MODELS / "dac" / f.name)

# NER: Deploy v2 (highest Entity F1: 0.7944, best precision: 0.7295)
for f in (MODELS_V2 / "ner").glob("*"):
    if f.is_file():
        shutil.copy2(f, PIPELINE_MODELS / "ner" / f.name)

print("[OK] Deployed DAC v3 and NER v2 to active pipeline/models/")
