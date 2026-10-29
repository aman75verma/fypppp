import json
from collections import Counter

LABEL_LIST = [
    'O',
    'B-SYMPTOM_VALUE',          'I-SYMPTOM_VALUE',
    'B-SYMPTOM_LOCATION',       'I-SYMPTOM_LOCATION',
    'B-SYMPTOM_CHARACTERISTICS', 'I-SYMPTOM_CHARACTERISTICS',
    'B-SYMPTOM_FREQUENCY',      'I-SYMPTOM_FREQUENCY',
    'B-TEMPORAL',               'I-TEMPORAL',
    'B-MEDICATION',             'I-MEDICATION',
    'B-MEDICAL_TEST_NAME',      'I-MEDICAL_TEST_NAME',
    'B-MEDICAL_HISTORY_VALUE',  'I-MEDICAL_HISTORY_VALUE',
    'B-HABIT_VALUE',            'I-HABIT_VALUE',
    'B-HABIT_FREQUENCY',        'I-HABIT_FREQUENCY',
    'B-EXPOSURE_VALUE',         'I-EXPOSURE_VALUE',
    'B-POLARITY',               'I-POLARITY',
]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

with open('data/val.jsonl', 'r', encoding='utf-8') as f:
    dialogues = [json.loads(line) for line in f]

CONTEXT_WINDOW = 2

def build_examples(dialogues, is_train=False):
    examples = []
    for dialogue in dialogues:
        turns = dialogue['utterances']
        for target_idx in range(len(turns)):
            target = turns[target_idx]
            target_tags = target['ner_tags']
            
            context_tokens = []
            context_labels = []
            start = max(0, target_idx - CONTEXT_WINDOW)
            
            for i in range(start, target_idx):
                ctx_turn = turns[i]
                spk = f"{ctx_turn.get('speaker', 'Unknown')}:"
                toks = [spk] + ctx_turn['tokens']
                context_tokens.extend(toks)
                context_labels.extend([-100] * len(toks))
            
            spk = f"{target.get('speaker', 'Unknown')}:"
            target_tokens = [spk] + target['tokens']
            target_label_ids = [-100] + [label2id.get(t, 0) for t in target_tags]
            
            all_tokens = context_tokens + target_tokens
            all_labels = context_labels + target_label_ids
            
            examples.append({
                'tokens': all_tokens,
                'labels': all_labels,
                'dialogue_id': dialogue.get('id', '?'),
                'turn_idx': target_idx,
            })
    return examples

val_examples = build_examples(dialogues, is_train=False)

val_active_labels = []
for ex in val_examples:
    for l in ex['labels']:
        if l != -100:
            val_active_labels.append(l)

c = Counter(val_active_labels)
print('Total active labels in val_examples:', len(val_active_labels))
print('Count of O (id 0):', c[0])
print('Count of all entities:', len(val_active_labels) - c[0])
print('Entity percentage:', (len(val_active_labels) - c[0]) / len(val_active_labels) * 100)

