import json, sys

def dump_nb(path, out_path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"=== {path} ===\n\n")
        for i, c in enumerate(nb['cells']):
            src = "".join(c['source'])
            out.write(f"#################### CELL {i:02d} [{c['cell_type']}] ####################\n")
            out.write(src)
            out.write("\n\n")

dump_nb("c:/Users/ammua/Desktop/final_prompt/NER/meditod_ner_training_v2.ipynb", "c:/Users/ammua/Desktop/final_prompt/scratch/dump_ner_v2.txt")
dump_nb("c:/Users/ammua/Desktop/final_prompt/DAC/meditod_dac_training_v2.ipynb", "c:/Users/ammua/Desktop/final_prompt/scratch/dump_dac_v2.txt")
print("Dumped successfully.")

