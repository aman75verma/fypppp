import json
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')

def verify_nb(path):
    print(f"=== Verifying {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Cells: {len(nb['cells'])}")
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"Code cells: {len(code_cells)}")
    
    # Check syntax of code cells (ignoring lines starting with '!')
    for i, c in enumerate(code_cells):
        code_lines = []
        for line in c['source']:
            if line.strip().startswith('!') or line.strip().startswith('%'):
                code_lines.append('# ' + line)
            else:
                code_lines.append(line)
        code = ''.join(code_lines)
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"❌ SyntaxError in code cell {i}: {e}")
            return False
            
    print("✓ All code cells passed AST syntax check!")
    return True

v_ner = verify_nb("c:/Users/ammua/Desktop/final_prompt/NER/v3/meditod_ner_training_v3.ipynb")
v_dac = verify_nb("c:/Users/ammua/Desktop/final_prompt/DAC/v3/meditod_dac_training_v3.ipynb")

if v_ner and v_dac:
    print("\n🎉 ALL V3 NOTEBOOKS VERIFIED SUCCESSFULLY!")
else:
    print("\n❌ Verification failed.")
