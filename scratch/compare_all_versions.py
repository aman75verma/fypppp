import json
import zipfile

print('========================================================================')
print('DAC COMPARISON: v1 vs v2 vs v3')
print('========================================================================')
for v, p in [('v1', 'DAC/v1_output/classification_report.txt'),
            ('v2', 'DAC/v2_output/classification_report.txt'),
            ('v3 (zip)', 'DAC/v3/v3_output.zip')]:
    if 'zip' in v:
        z = zipfile.ZipFile(p)
        content = z.read('classification_report.txt').decode('utf-8')
    else:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
    lines = content.strip().split('\n')
    print(f'*** {v} ***')
    for line in lines:
        if any(k in line for k in ['Test', 'Best', 'accuracy', 'macro avg', 'weighted avg']):
            print('  ', line)
    print()

print('========================================================================')
print('NER COMPARISON: v1 vs v2 vs v3')
print('========================================================================')
for v, p in [('v1 (pipeline)', 'pipeline/models/ner/training_config.json'),
            ('v2', 'NER/v2_output/evaluation_report.txt'),
            ('v3 (zip)', 'NER/v3/v3_output.zip')]:
    if 'zip' in v:
        z = zipfile.ZipFile(p)
        content = z.read('evaluation_report.txt').decode('utf-8')
        lines = content.strip().split('\n')
        print(f'*** {v} ***')
        for line in lines:
            if any(k in line for k in ['Test', 'Best', 'micro avg', 'macro avg', 'weighted avg']):
                print('  ', line)
    elif 'v2' in v:
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.strip().split('\n')
        print(f'*** {v} ***')
        for line in lines:
            if any(k in line for k in ['Test', 'Best', 'micro avg', 'macro avg', 'weighted avg']):
                print('  ', line)
    else:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        print(f'*** {v} ***')
        print(f"   Val F1: {d.get('best_val_f1'):.4f}, Test F1: {d.get('test_f1'):.4f}")
    print()

