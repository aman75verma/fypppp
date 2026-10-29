import json, sys

sys.stdout.reconfigure(encoding='utf-8')

def summarize_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"=== {path} ===")
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        src = "".join(c['source'])
        first_line = src.strip().split("\n")[0] if src.strip() else "(empty)"
        print(f"Cell {i:02d} [{c['cell_type']:8s}]: {first_line[:80]}")

summarize_nb("c:/Users/ammua/Desktop/final_prompt/NER/meditod_ner_training_v2.ipynb")
print()
summarize_nb("c:/Users/ammua/Desktop/final_prompt/DAC/meditod_dac_training_v2.ipynb")
