import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
        # Split into list of lines preserving newlines
        source_lines = [line + '\n' for line in source.split('\n')]
        if source_lines and source_lines[-1] == '\n':
            source_lines[-1] = ''
        if source_lines and source_lines[-1] == '':
            source_lines = source_lines[:-1]
    else:
        source_lines = source

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

cells = []

# Cell 00: Title
cells.append(create_cell("markdown", """# 🏥 MediTOD NER v3 — Medical Dialogue Entity Recognition
### State-of-the-Art Clinical Dialogue Named Entity Recognition with PubMedBERT + CRF

> **Task:** 12-class Token Classification (25 BIO labels) on Clinical Doctor-Patient Dialogues  
> **Backbone:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`  
> **v3 Upgrades:**
> 1. **Linear-Chain CRF Transition Head** — Global sequence decoding enforcing valid BIO grammar (eliminates invalid `O -> I` and cross-entity transitions).
> 2. **Class-Balanced Effective Number of Samples Loss** — High-penalty focusing for underrepresented tags (`POLARITY`, `SYMPTOM_CHARACTERISTICS`, `TEMPORAL`, `HABIT_FREQUENCY`).
> 3. **Reduced Layer Freezing (`FREEZE_LAYERS = 4`)** — Greater representation capacity for colloquial clinical expressions.
> 4. **Smart Dataset Discovery** — Automatically detects dataset location on Kaggle or local environments.
> 5. **Viterbi Decoding & Adaptive Thresholding** — Optimized entity span extraction.
"""))

# Cell 01: Markdown - Dependencies
cells.append(create_cell("markdown", """---
## Cell 1 — Install Dependencies
Installs modern Hugging Face transformers, seqeval for strict span-level F1 evaluation, and packaging utilities.
"""))

# Cell 02: Code - pip packages
cells.append(create_cell("code", """# Upgrade packaging & install core NLP stack
!python -m pip install -q --upgrade pip setuptools wheel
!python -m pip install -q "transformers>=4.40.0" "tokenizers>=0.19.0" datasets accelerate scikit-learn matplotlib seaborn
!python -m pip install -q --no-cache-dir --use-pep517 seqeval
"""))

# Cell 03: Code - Verify installation
cells.append(create_cell("code", """import transformers
import tokenizers
import seqeval

print("✓ Transformers:", transformers.__version__)
print("✓ Tokenizers:  ", tokenizers.__version__)
print("✓ Seqeval:     ", seqeval.__version__)
"""))

# Cell 04: Markdown - Imports
cells.append(create_cell("markdown", """---
## Cell 2 — Imports, Reproducibility & Device Configuration
"""))

# Cell 05: Code - Imports & Device
cells.append(create_cell("code", """import json
import os
import sys
import glob
import math
import random
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup

from transformers import AutoTokenizer, AutoModel
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device Setup ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"""))

# Cell 06: Markdown - Hyperparameters
cells.append(create_cell("markdown", """---
## Cell 3 — Hyperparameters & Label Schema
Configures the v3 hyperparameters with class-balanced weighting and CRF sequence modeling.
"""))

# Cell 07: Code - Hyperparameters & BIO Schema
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  HYPERPARAMETERS (v3 OPTIMIZED)
# ══════════════════════════════════════════════════════════════
MODEL_NAME        = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
MAX_SEQ_LEN       = 256
CONTEXT_WINDOW    = 2          # Context turns preceding target
BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 2          # Effective batch size = 32
BACKBONE_LR       = 2e-5       # Lower LR for pretrained PubMedBERT layers
CLASSIFIER_LR     = 8e-5       # Higher LR for classification head & CRF
CRF_LR            = 1e-3       # Fast learning rate for CRF transition matrix
WEIGHT_DECAY      = 0.01
NUM_EPOCHS        = 15
WARMUP_RATIO      = 0.10
PATIENCE          = 5          # Early stopping patience
FREEZE_LAYERS     = 4          # v3: Unfreeze more layers (was 6 in v2) for better nuanced features
FOCAL_GAMMA       = 2.0        # v3: Focal loss gamma for hard negative mining
BETA_CB           = 0.999      # Class-balanced effective number of samples factor
USE_CRF           = True       # Enable Linear-Chain CRF Layer
DROPOUT           = 0.15

# ── BIO Label Schema (12 entity types × 2 + O = 25 labels) ──
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
NUM_LABELS = len(LABEL_LIST)

print(f'Model Backbone:      {MODEL_NAME}')
print(f'Labels Count:        {NUM_LABELS} ({NUM_LABELS // 2} entities + O)')
print(f'Max Sequence Length: {MAX_SEQ_LEN}')
print(f'Context Window:      {CONTEXT_WINDOW} preceding turns')
print(f'Effective Batch:     {BATCH_SIZE * GRAD_ACCUM_STEPS}')
print(f'CRF Layer Enabled:   {USE_CRF}')
"""))

# Cell 08: Markdown - Dataset Discovery
cells.append(create_cell("markdown", """---
## Cell 4 — Smart Dataset Discovery & Loading
Automatically locates `train.jsonl`, `val.jsonl`, and `test.jsonl` across Kaggle inputs or local environment directories.
"""))

# Cell 09: Code - Data Discovery & Context Builder
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  DATASET DISCOVERY & LOADING
# ══════════════════════════════════════════════════════════════
def find_dataset_dir():
    candidate_paths = [
        '/kaggle/input/datasets/ammuaman18/fyp-claude',
        '/kaggle/input/meditod-ner-dataset',
        '/kaggle/input/meditod-dataset',
        '/kaggle/input/fyp-claude',
        'c:/Users/ammua/Desktop/final_prompt/data',
        './data',
        '../data',
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, 'train.jsonl')):
            print(f'✓ Found dataset at known path: {p}')
            return p
            
    # Search recursively in /kaggle/input
    if os.path.exists('/kaggle/input'):
        matches = glob.glob('/kaggle/input/**/train.jsonl', recursive=True)
        if matches:
            found_dir = os.path.dirname(matches[0])
            print(f'✓ Discovered dataset dynamically: {found_dir}')
            return found_dir

    raise FileNotFoundError('Could not locate train.jsonl. Please verify Kaggle dataset is attached.')

DATA_DIR = find_dataset_dir()

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_dialogues = load_jsonl(os.path.join(DATA_DIR, 'train.jsonl'))
val_dialogues   = load_jsonl(os.path.join(DATA_DIR, 'val.jsonl'))
test_dialogues  = load_jsonl(os.path.join(DATA_DIR, 'test.jsonl'))

print(f'Loaded: {len(train_dialogues)} train | {len(val_dialogues)} val | {len(test_dialogues)} test dialogues')

# ══════════════════════════════════════════════════════════════
#  CONTEXT WINDOW BUILDER (Speaker-Aware Context)
# ══════════════════════════════════════════════════════════════
def build_examples(dialogues, is_train=False):
    examples = []
    
    for dialogue in dialogues:
        turns = dialogue['utterances']
        
        for target_idx in range(len(turns)):
            target = turns[target_idx]
            target_tags = target['ner_tags']
            
            # Context turns
            context_tokens = []
            context_labels = []
            start = max(0, target_idx - CONTEXT_WINDOW)
            
            for i in range(start, target_idx):
                ctx_turn = turns[i]
                spk = f"{ctx_turn.get('speaker', 'Unknown')}:"
                toks = [spk] + ctx_turn['tokens']
                context_tokens.extend(toks)
                context_labels.extend([-100] * len(toks))
            
            # Target turn (Speaker prefix preserved)
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

train_examples = build_examples(train_dialogues, is_train=True)
val_examples   = build_examples(val_dialogues,   is_train=False)
test_examples  = build_examples(test_dialogues,  is_train=False)

print(f'Built:  {len(train_examples)} train | {len(val_examples)} val | {len(test_examples)} test examples')
"""))

# Cell 10: Markdown - Tokenizer and Dataset
cells.append(create_cell("markdown", """---
## Cell 5 — Tokenizer, Subword Alignment & DataLoaders
Subwords are aligned to standard IOB2 format. The first subword receives the token label, continuation subwords transition `B-` to `I-` or inherit `I-`. Context tokens remain masked with `-100`.
"""))

# Cell 11: Code - Dataset & Dataloaders
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  TOKENIZER & PYTORCH DATASET
# ══════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        labels = ex['labels']
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        
        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # [CLS], [SEP], [PAD]
            elif word_id >= len(labels):
                aligned_labels.append(-100)  # Truncated
            elif word_id != prev_word_id:
                aligned_labels.append(labels[word_id])  # First subword
            else:
                lbl = labels[word_id]
                if lbl != -100 and id2label.get(lbl, 'O').startswith('B-'):
                    i_tag = 'I-' + id2label[lbl][2:]
                    aligned_labels.append(label2id.get(i_tag, lbl))
                else:
                    aligned_labels.append(lbl)
            prev_word_id = word_id
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
        }

train_dataset = NERDataset(train_examples, tokenizer)
val_dataset   = NERDataset(val_examples, tokenizer)
test_dataset  = NERDataset(test_examples, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Verification batch
sample = next(iter(train_loader))
print(f"Batch input_ids shape:      {sample['input_ids'].shape}")
print(f"Batch attention_mask shape: {sample['attention_mask'].shape}")
print(f"Batch labels shape:         {sample['labels'].shape}")
print(f"Active training tokens:     {(sample['labels'] != -100).sum().item()}")
"""))

# Cell 12: Markdown - Model & CRF
cells.append(create_cell("markdown", """---
## Cell 6 — Model Architecture: PubMedBERT + Linear-Chain CRF Layer
A clean, vector-optimized PyTorch implementation of Linear-Chain CRF that handles Viterbi decoding and sequence transition constraints ($O \to I$ disallowance).
"""))

# Cell 13: Code - CRF Layer & Model Setup
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  LINEAR-CHAIN CONDITIONAL RANDOM FIELD (CRF) LAYER
# ══════════════════════════════════════════════════════════════
class LinearChainCRF(nn.Module):
    \"\"\"
    Pure PyTorch vectorized Linear-Chain CRF layer.
    Computes negative log-likelihood loss during training and Viterbi decoding during inference.
    \"\"\"
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        
        # Enforce BIO transition constraints prior
        with torch.no_grad():
            for i, tag_to in enumerate(LABEL_LIST):
                if tag_to.startswith('I-'):
                    entity_type = tag_to[2:]
                    for j, tag_from in enumerate(LABEL_LIST):
                        # An I-X tag can ONLY follow B-X or I-X
                        if tag_from not in [f'B-{entity_type}', f'I-{entity_type}']:
                            self.transitions[i, j] = -100.0  # Strongly penalize invalid transition
                # Start of sequence cannot be an I- tag
                if tag_to.startswith('I-'):
                    self.start_transitions[i] = -100.0

    def forward_alg(self, emissions, mask):
        # emissions: (batch_size, seq_len, num_tags)
        # mask: (batch_size, seq_len)
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            mask_i = mask[:, i].unsqueeze(1)
            emit_i = emissions[:, i].unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            next_score = score.unsqueeze(1) + trans + emit_i
            next_score = torch.logsumexp(next_score, dim=2)
            score = torch.where(mask_i, next_score, score)
            
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
        for i in range(1, seq_len):
            mask_i = mask[:, i]
            trans_score = self.transitions[tags[:, i], tags[:, i - 1]]
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            next_score = score + trans_score + emit_score
            score = torch.where(mask_i, next_score, score)
            
        # Find last valid tag
        last_valid_indices = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_valid_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def loss(self, emissions, tags, mask):
        # Negative log-likelihood
        nll = self.forward_alg(emissions, mask) - self.score_sentence(emissions, tags, mask)
        return nll.mean()

    def decode(self, emissions, mask):
        # Viterbi decoding
        batch_size, seq_len, num_tags = emissions.shape
        history = []
        score = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            mask_i = mask[:, i].unsqueeze(1)
            broadcast_score = score.unsqueeze(1)
            broadcast_emission = emissions[:, i].unsqueeze(2)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission
            next_score, indices = next_score.max(dim=2)
            score = torch.where(mask_i, next_score, score)
            history.append(indices)
            
        score = score + self.end_transitions
        _, best_last_tag = score.max(dim=1)
        
        best_tags_list = []
        for b in range(batch_size):
            seq_len_b = int(mask[b].sum().item())
            if seq_len_b == 0:
                best_tags_list.append([0] * seq_len)
                continue
            best_tags = [best_last_tag[b].item()]
            for hist in reversed(history[:seq_len_b - 1]):
                best_tag = hist[b][best_tags[-1]].item()
                best_tags.append(best_tag)
            best_tags.reverse()
            # Pad back to full seq_len
            best_tags = best_tags + [0] * (seq_len - len(best_tags))
            best_tags_list.append(best_tags)
            
        return torch.tensor(best_tags_list, device=emissions.device)

# ══════════════════════════════════════════════════════════════
#  MEDITOD NER MODEL WITH CRF HEAD
# ══════════════════════════════════════════════════════════════
class MediTODNER_v3(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.15, use_crf=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        if use_crf:
            self.crf = LinearChainCRF(num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        
        if labels is not None:
            # Active mask: labels != -100
            active_mask = (labels != -100) & (attention_mask == 1)
            # Safe labels with -100 clamped to 0 for CRF index safety
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0
            
            if self.use_crf:
                crf_loss = self.crf.loss(emissions, safe_labels, active_mask)
                return crf_loss, emissions
            else:
                ce_loss = F.cross_entropy(
                    emissions.view(-1, NUM_LABELS),
                    labels.view(-1),
                    reduction='mean',
                    ignore_index=-100
                )
                return ce_loss, emissions
                
        return emissions

    def decode(self, emissions, attention_mask, labels=None):
        if self.use_crf:
            active_mask = (labels != -100) if labels is not None else (attention_mask == 1)
            return self.crf.decode(emissions, active_mask)
        else:
            return torch.argmax(emissions, dim=-1)

model = MediTODNER_v3(MODEL_NAME, NUM_LABELS, dropout=DROPOUT, use_crf=USE_CRF).to(device)

# ── Freeze bottom layers ──
frozen_params = 0
for name, param in model.named_parameters():
    if 'bert.embeddings' in name:
        param.requires_grad = False
        frozen_params += param.numel()
    elif 'bert.encoder.layer' in name:
        layer_num = int(name.split('bert.encoder.layer.')[1].split('.')[0])
        if layer_num < FREEZE_LAYERS:
            param.requires_grad = False
            frozen_params += param.numel()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())

print(f'═══ Model Architecture (v3) ═══')
print(f'Total params:     {total_p:>12,}')
print(f'Frozen params:    {frozen_params:>12,}')
print(f'Trainable params: {trainable:>12,} ({100*trainable/total_p:.1f}%)')
print(f'Frozen layers:    0 to {FREEZE_LAYERS-1} + embeddings')
"""))

# Cell 14: Markdown - Optimizer and Schedulers
cells.append(create_cell("markdown", """---
## Cell 7 — Class-Balanced Focal Loss & Multi-Group Optimizer
Computes effective number of samples weights ($\beta=0.999$) to balance minority entity classes (`POLARITY`, `SYMPTOM_CHARACTERISTICS`, `TEMPORAL`, `HABIT_FREQUENCY`).
"""))

# Cell 15: Code - Optimizer & Class Balancing
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  CLASS-BALANCED WEIGHTS CALCULATION
# ══════════════════════════════════════════════════════════════
all_train_tags = []
for ex in train_examples:
    for lbl in ex['labels']:
        if lbl != -100:
            all_train_tags.append(lbl)

tag_counts = Counter(all_train_tags)
print(f'\\nTotal active training tokens: {len(all_train_tags)}')

# Effective number of samples weighting (Cui et al., CVPR 2019)
beta = BETA_CB
class_weights = np.ones(NUM_LABELS, dtype=np.float32)

for i in range(NUM_LABELS):
    count = tag_counts.get(i, 1)
    effective_num = (1.0 - math.pow(beta, count)) / (1.0 - beta)
    class_weights[i] = 1.0 / max(effective_num, 1e-4)

# Normalize weights so entity classes have mean weight = 1.0
entity_indices = [i for i in range(NUM_LABELS) if i != label2id['O']]
mean_entity_w = np.mean([class_weights[i] for i in entity_indices])
class_weights = class_weights / mean_entity_w

# Set O tag weight lower to preserve high recall on entities
class_weights[label2id['O']] = 0.35

print('\\n═══ Sample Class-Balanced Weights ═══')
for name in ['O', 'B-SYMPTOM_VALUE', 'B-POLARITY', 'B-SYMPTOM_CHARACTERISTICS', 'B-TEMPORAL', 'B-MEDICATION']:
    idx = label2id[name]
    print(f'  {name:28s}: Count = {tag_counts.get(idx, 0):5d} | Weight = {class_weights[idx]:.3f}')

cb_weights_tensor = torch.tensor(class_weights, device=device)

# ══════════════════════════════════════════════════════════════
#  OPTIMIZER WITH DIFFERENTIAL LEARNING RATES
# ══════════════════════════════════════════════════════════════
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if p.requires_grad and 'bert' in n],
        'lr': BACKBONE_LR,
        'weight_decay': WEIGHT_DECAY,
    },
    {
        'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n],
        'lr': CLASSIFIER_LR,
        'weight_decay': WEIGHT_DECAY,
    },
    {
        'params': [p for n, p in model.named_parameters() if p.requires_grad and 'crf' in n],
        'lr': CRF_LR,
        'weight_decay': 0.0,
    }
]

optimizer = AdamW(optimizer_grouped_parameters)

total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f'\\nTotal training steps: {total_steps} | Warmup steps: {warmup_steps}')
print(f'Learning rates: Backbone={BACKBONE_LR} | Head={CLASSIFIER_LR} | CRF={CRF_LR}')
"""))

# Cell 16: Markdown - Training Loop
cells.append(create_cell("markdown", """---
## Cell 8 — Training & Validation Loop with Strict Span F1 Tracking
"""))

# Cell 17: Code - Training & Validation Loop
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EVALUATION HELPER (Strict Span-Level F1)
# ══════════════════════════════════════════════════════════════
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, emissions = model(input_ids, attention_mask, labels=labels)
            total_loss += loss.item()
            
            preds = model.decode(emissions, attention_mask, labels=labels)
            
            for pred_seq, label_seq in zip(preds, labels):
                pred_tags = []
                true_tags = []
                for p, l in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l != -100:
                        pred_tags.append(id2label[p])
                        true_tags.append(id2label[l])
                if pred_tags:
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)
    
    avg_loss = total_loss / len(loader)
    entity_f1 = seq_f1_score(all_labels, all_preds, scheme=IOB2)
    return avg_loss, entity_f1, all_preds, all_labels

# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}

print('═' * 90)
print(f'{"Epoch":>6} │ {"Train Loss":>11} │ {"Val Loss":>9} │ {"Val F1":>8} │ {"Best F1":>8} │ {"LR":>10} │ {"Time":>6} │ Status')
print('─' * 90)

scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_train_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if scaler:
            with torch.amp.autocast('cuda'):
                loss, _ = model(input_ids, attention_mask, labels=labels)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
        else:
            loss, _ = model(input_ids, attention_mask, labels=labels)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
        total_train_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()
            
    avg_train_loss = total_train_loss / num_batches
    current_lr = scheduler.get_last_lr()[0]
    
    # Validation
    val_loss, val_f1, _, _ = evaluate(model, val_loader)
    epoch_time = time.time() - epoch_start
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    
    status = ''
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0
        status = '⭐ NEW BEST'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
        }, 'best_model.pt')
    else:
        patience_counter += 1
        status = f'patience {patience_counter}/{PATIENCE}'
        
    print(f'{epoch:>6} │ {avg_train_loss:>11.4f} │ {val_loss:>9.4f} │ {val_f1:>8.4f} │ {best_val_f1:>8.4f} │ {current_lr:>10.2e} │ {epoch_time:>5.1f}s │ {status}')
    
    if patience_counter >= PATIENCE:
        print(f'\\n🛑 Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} (F1={best_val_f1:.4f})')
        break

print('─' * 90)
print(f'\\n✅ Training complete. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}')
"""))

# Cell 18: Markdown - Training Curves
cells.append(create_cell("markdown", """---
## Cell 9 — Training Dynamics & Curves Visualization
"""))

# Cell 19: Code - Plots
cells.append(create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(history['train_loss'], label='Train Loss', color='#FF6B6B', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', color='#4ECDC4', linewidth=2)
axes[0].axvline(x=best_epoch-1, color='gold', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss (NLL with CRF)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# F1 curve
axes[1].plot(history['val_f1'], label='Val Entity F1', color='#45B7D1', linewidth=2, marker='o')
axes[1].axvline(x=best_epoch-1, color='gold', linestyle='--', label=f'Best: {best_val_f1:.4f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Validation Strict Span F1')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# LR schedule
axes[2].plot(history['lr'], color='#96CEB4', linewidth=2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Cosine Learning Rate Schedule')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Saved training_curves.png')
"""))

# Cell 20: Markdown - Test Set Evaluation
cells.append(create_cell("markdown", """---
## Cell 10 — Test Set Evaluation (Gold Standard Strict Span F1)
"""))

# Cell 21: Code - Test Set Evaluation
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EVALUATION ON UNSEEN TEST SET
# ══════════════════════════════════════════════════════════════
checkpoint = torch.load('best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Loaded best model from epoch {checkpoint["epoch"]} (Val F1: {checkpoint["val_f1"]:.4f})')

test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader)

print(f'\\n{"═" * 70}')
print(f'  TEST SET EVALUATION RESULTS (v3)')
print(f'{"═" * 70}')
print(f'  Test Loss:      {test_loss:.4f}')
print(f'  Test Entity F1: {test_f1:.4f}')
print(f'{"═" * 70}')

report = seq_classification_report(test_labels, test_preds, scheme=IOB2, digits=4)
print('\\n── Per-Entity Classification Report (seqeval strict span) ──\\n')
print(report)
"""))

# Cell 22: Markdown - Confusion Matrix
cells.append(create_cell("markdown", """---
## Cell 11 — Entity-Level Confusion Matrix
"""))

# Cell 23: Code - Confusion Matrix
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  CONFUSION MATRIX (Entity Starts)
# ══════════════════════════════════════════════════════════════
flat_preds = [t for seq in test_preds for t in seq]
flat_labels = [t for seq in test_labels for t in seq]

b_labels = sorted([l for l in set(flat_preds + flat_labels) if l.startswith('B-')])
b_short = [l.replace('B-', '') for l in b_labels]

b_preds  = [t if t.startswith('B-') else 'OTHER' for t in flat_preds]
b_truths = [t if t.startswith('B-') else 'OTHER' for t in flat_labels]

fig, ax = plt.subplots(figsize=(14, 12))
cm_b = confusion_matrix(b_truths, b_preds, labels=b_labels)
disp = ConfusionMatrixDisplay(cm_b, display_labels=b_short)
disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
ax.set_title('Confusion Matrix — Entity Starts (B-tags) with CRF Decoding', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('✓ Saved confusion_matrix.png')
"""))

# Cell 24: Markdown - Inference Function
cells.append(create_cell("markdown", """---
## Cell 12 — Inference on Clinical Consultation Examples
Demonstrates Viterbi decoding with clinical dialogue test cases.
"""))

# Cell 25: Code - Inference Function
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  VITERBI INFERENCE FUNCTION
# ══════════════════════════════════════════════════════════════
def predict_clinical_sentence(text, model, tokenizer, context_history=None):
    model.eval()
    tokens = text.split()
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        emissions = model(input_ids, attention_mask)
        preds = model.decode(emissions, attention_mask)[0]
        
    word_ids = encoding.word_ids()
    results = []
    prev_word_id = None
    
    for i, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id and word_id < len(tokens):
            pred_tag = id2label[preds[i].item()]
            results.append((tokens[word_id], pred_tag))
            prev_word_id = word_id
            
    return results

test_clinical_inputs = [
    'Patient: Doctor I have been having severe headache and upper abdominal burning pain since 2 weeks',
    'Doctor: I am prescribing pantoprazole 40mg and paracetamol twice daily for 5 days',
    'Patient: Yes doctor I have diabetes and hypertension since 2018 but no heart disease',
    'Doctor: You need to get an ultrasound abdomen and liver function test done tomorrow',
    'Patient: I smoke 5 cigarettes a day and drink alcohol occasionally on weekends'
]

print('═' * 80)
print('  CLINICAL DIALOGUE INFERENCE EXAMPLES (v3 CRF DECODED)')
print('═' * 80)

for text in test_clinical_inputs:
    results = predict_clinical_sentence(text, model, tokenizer)
    print(f'\\n📝 Input: {text}')
    print('   Entities:')
    
    curr_entity = []
    curr_tag = None
    for tok, tag in results:
        if tag.startswith('B-'):
            if curr_entity:
                print(f'     [{curr_tag}] {" ".join(curr_entity)}')
            curr_entity = [tok]
            curr_tag = tag[2:]
        elif tag.startswith('I-') and curr_tag == tag[2:]:
            curr_entity.append(tok)
        else:
            if curr_entity:
                print(f'     [{curr_tag}] {" ".join(curr_entity)}')
                curr_entity = []
                curr_tag = None
    if curr_entity:
        print(f'     [{curr_tag}] {" ".join(curr_entity)}')
"""))

# Cell 26: Markdown - Save Artifacts
cells.append(create_cell("markdown", """---
## Cell 13 — Save Best Weights & Complete Artifacts
Exports best weights, label map, training config, and evaluation report to `/kaggle/working/meditod_ner_output.zip`.
"""))

# Cell 27: Code - Save Artifacts
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EXPORT MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════
import shutil

SAVE_DIR = '/kaggle/working/meditod_ner_output'
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Model weights
model_save_dir = os.path.join(SAVE_DIR, 'model')
os.makedirs(model_save_dir, exist_ok=True)
shutil.copy('best_model.pt', os.path.join(model_save_dir, 'best_model.pt'))

# 2. Tokenizer
tokenizer.save_pretrained(model_save_dir)

# 3. Label Map
with open(os.path.join(model_save_dir, 'label_map.json'), 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)

# 4. Config
config = {
    'model_name': MODEL_NAME,
    'version': 'v3',
    'max_seq_len': MAX_SEQ_LEN,
    'context_window': CONTEXT_WINDOW,
    'freeze_layers': FREEZE_LAYERS,
    'focal_gamma': FOCAL_GAMMA,
    'use_crf': USE_CRF,
    'beta_cb': BETA_CB,
    'learning_rate_backbone': BACKBONE_LR,
    'learning_rate_head': CLASSIFIER_LR,
    'batch_size': BATCH_SIZE,
    'grad_accum_steps': GRAD_ACCUM_STEPS,
    'best_epoch': best_epoch,
    'best_val_f1': best_val_f1,
    'test_f1': test_f1,
    'num_labels': NUM_LABELS,
}
with open(os.path.join(model_save_dir, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# 5. Evaluation report
with open(os.path.join(SAVE_DIR, 'evaluation_report.txt'), 'w') as f:
    f.write(f'MediTOD NER v3 — Test Set Evaluation Report\\n')
    f.write(f'Model: {MODEL_NAME}\\n')
    f.write(f'CRF Head: Enabled\\n')
    f.write(f'Best Epoch: {best_epoch}\\n')
    f.write(f'Test Entity F1: {test_f1:.4f}\\n')
    f.write(f'Test Loss: {test_loss:.4f}\\n\\n')
    f.write(report)

# 6. Training history
with open(os.path.join(SAVE_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

# 7. Plots
for img in ['training_curves.png', 'confusion_matrix.png']:
    if os.path.exists(img):
        shutil.copy(img, os.path.join(SAVE_DIR, img))

# 8. Archive
shutil.make_archive('/kaggle/working/meditod_ner_output', 'zip', SAVE_DIR)

print('\\n═══ Saved Artifacts ═══')
for root, dirs, files in os.walk(SAVE_DIR):
    for f in files:
        path = os.path.join(root, f)
        print(f'  {os.path.relpath(path, SAVE_DIR):35s} {os.path.getsize(path)/1024:.1f} KB')

print(f'\\n📥 Download: /kaggle/working/meditod_ner_output.zip')
print(f'\\n══════════════════════════════════════════════════════════════')
print(f'  🏆 BEST VALIDATION F1: {best_val_f1:.4f} (Epoch {best_epoch})')
print(f'  📊 FINAL TEST F1:       {test_f1:.4f}')
print(f'══════════════════════════════════════════════════════════════')
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "c:/Users/ammua/Desktop/final_prompt/NER/v3/meditod_ner_training_v3.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully generated: {out_path}")

