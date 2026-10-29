import json, sys
nb = json.load(open(r'c:\Users\ammua\Desktop\final_prompt\DAC\meditod_dac_training_v2.ipynb', 'r', encoding='utf-8'))
print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    ct = cell.get('cell_type', '?')
    outputs = cell.get('outputs', [])
    has_output = any(o.get('text') or o.get('data') for o in outputs) if outputs else False
    print(f'  Cell {i}: type={ct}, outputs={len(outputs)}, has_output={has_output}')

