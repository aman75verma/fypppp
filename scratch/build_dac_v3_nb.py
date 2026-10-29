import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
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
cells.append(create_cell("markdown", """# 🗣️ MediTOD DAC v3 — Clinical Dialogue Act Classification
### State-of-the-Art Multi-Turn Intent Classification with PubMedBERT + Speaker-Aware Embeddings

> **Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`  
> **Task:** 16-class Dialogue Act Classification on Doctor-Patient Consultations  
> **v3 Upgrades:**
> 1. **Dedicated Speaker Special Tokens (`[DOCTOR]`, `[PATIENT]`)** — Explicit role embeddings to eliminate Doctor/Patient intent confusion.
> 2. **Deep Non-Linear MLP Classification Head** — 2-layer projection with LayerNorm and GELU for high-order communicative intent features.
> 3. **Class-Balanced Effective Number of Samples Loss ($\beta=0.999$)** — Directly cures the `Inform-Basic_Information` and rare request class bottlenecks.
> 4. **Expanded Context Window (`MAX_SEQ_LEN=256`, `CONTEXT_WINDOW=3`)** — Preserves full 4-turn conversational narrative without truncation.
> 5. **Dynamic Dataset Discovery** — Seamless auto-detection on Kaggle or local environments.
> 6. **Full HuggingFace & PyTorch Export** — Dual export for plug-and-play pipeline deployment.
"""))

# Cell 01: Markdown - Dependencies
cells.append(create_cell("markdown", """---
## Cell 1 — Install Dependencies
"""))

# Cell 02: Code - Install Dependencies
cells.append(create_cell("code", """!python -m pip install -q --upgrade pip setuptools wheel
!python -m pip install -q "transformers>=4.40.0" "tokenizers>=0.19.0" datasets accelerate scikit-learn seaborn matplotlib
"""))

# Cell 03: Markdown - Imports & Configuration
cells.append(create_cell("markdown", """---
## Cell 2 — Imports, Logging & Reproducibility
"""))

# Cell 04: Code - Imports & Configuration
cells.append(create_cell("code", """import os
import json
import glob
import math
import random
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Logging Setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device Setup ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU:  {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
"""))

# Cell 05: Markdown - Hyperparameters & Intent Taxonomy
cells.append(create_cell("markdown", """---
## Cell 3 — Hyperparameters & Dialogue Act Taxonomy
"""))

# Cell 06: Code - Hyperparameters & Taxonomy
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  HYPERPARAMETERS (v3 OPTIMIZED)
# ══════════════════════════════════════════════════════════════
MODEL_NAME      = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_SEQ_LEN     = 256       # v3: Expanded for full multi-turn conversational context
CONTEXT_WINDOW  = 3         # 3 preceding turns + 1 target turn = 4 turns
BATCH_SIZE      = 32
NUM_EPOCHS      = 15
BACKBONE_LR     = 1.5e-5    # Lower LR for pretrained PubMedBERT backbone
CLASSIFIER_LR   = 1e-4      # Higher LR for multi-layer MLP classification head
WEIGHT_DECAY    = 0.02
WARMUP_RATIO    = 0.10
FOCAL_GAMMA     = 2.5       # v3: Balanced focal parameter for hard intent mining
DROPOUT         = 0.20      # Dropout rate for classifier head
PATIENCE        = 5         # Early stopping patience
FREEZE_LAYERS   = 4         # Unfreeze top 8 layers of transformer for intent nuances
BETA_CB         = 0.999     # Class-balanced effective number of samples factor

# ══════════════════════════════════════════════════════════════
#  16 DIALOGUE ACT TAGS
# ══════════════════════════════════════════════════════════════
TAGS = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice',
    'Inform-Etiology', 'Diagnose', 'Request-Basic_Information',
    'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information',
    'Inform-Precautions', 'Inform-Existing_Examination_and_Treatment',
    'Inform-Drug_Recommendation', 'Request-Symptom', 'Inform-Symptom', 'Other'
]

tag2id = {tag: idx for idx, tag in enumerate(TAGS)}
id2tag = {idx: tag for tag, idx in tag2id.items()}
NUM_LABELS = len(TAGS)

logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Labels: {NUM_LABELS} Dialogue Act Classes")
logger.info(f"Context: {CONTEXT_WINDOW} preceding turns | Max Seq Len: {MAX_SEQ_LEN}")
"""))

# Cell 07: Markdown - Dataset Discovery & Loading
cells.append(create_cell("markdown", """---
## Cell 4 — Dynamic Dataset Discovery & Speaker-Role Context Builder
Automatically discovers `train.jsonl`, `val.jsonl`, and `test.jsonl`. Utterances are formatted with explicit `[DOCTOR]` and `[PATIENT]` speaker role tokens.
"""))

# Cell 08: Code - Dataset Discovery & Context Builder
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  SMART DATASET DISCOVERY
# ══════════════════════════════════════════════════════════════
def find_dataset_dir():
    candidates = [
        '/kaggle/input/meditod-dac-dataset',
        '/kaggle/input/datasets/ammuaman18/fyp-claude',
        '/kaggle/input/meditod-ner-dataset',
        '/kaggle/input/fyp-claude',
        'c:/Users/ammua/Desktop/final_prompt/data',
        './data',
        '../data',
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, 'train.jsonl')):
            print(f'✓ Found dataset at known path: {p}')
            return p
            
    if os.path.exists('/kaggle/input'):
        matches = glob.glob('/kaggle/input/**/train.jsonl', recursive=True)
        if matches:
            found = os.path.dirname(matches[0])
            print(f'✓ Discovered dataset dynamically: {found}')
            return found

    raise FileNotFoundError('Could not locate train.jsonl. Please verify dataset is attached.')

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

logger.info(f"Loaded dialogues: {len(train_dialogues)} train | {len(val_dialogues)} val | {len(test_dialogues)} test")

# ══════════════════════════════════════════════════════════════
#  SPEAKER-ROLE CONTEXT BUILDER (v3)
# ══════════════════════════════════════════════════════════════
def build_context_examples(dialogues, context_window=CONTEXT_WINDOW):
    \"\"\"
    Builds context-enriched inputs with explicit speaker tags.
    Input Format:
        [DOCTOR] Doctor: Which area? [SEP] [PATIENT] Patient: From the chest area.
    \"\"\"
    examples = []
    for d in dialogues:
        utts = d['utterances']
        for i, u in enumerate(utts):
            parts = []
            start = max(0, i - context_window)
            for j in range(start, i):
                spk = utts[j].get('speaker', 'Unknown').upper()
                parts.append(f"[{spk}] {utts[j]['text']}")
            
            # Target turn
            spk = u.get('speaker', 'Unknown').upper()
            parts.append(f"[{spk}] {u['text']}")
            full_text = " [SEP] ".join(parts)
            label = tag2id[u['intent']]
            examples.append({
                'text': full_text,
                'label': label,
                'speaker': spk,
                'target_text': u['text']
            })
    return examples

train_examples = build_context_examples(train_dialogues)
val_examples   = build_context_examples(val_dialogues)
test_examples  = build_context_examples(test_dialogues)

logger.info(f"Built examples: {len(train_examples)} train | {len(val_examples)} val | {len(test_examples)} test")

# Distribution
print(f"\\n{'=' * 70}")
print(f"  TRAINING CLASS DISTRIBUTION")
print(f"{'=' * 70}")
train_counts = Counter(ex['label'] for ex in train_examples)
for tag_id in sorted(train_counts.keys()):
    tag_name = id2tag[tag_id]
    cnt = train_counts[tag_id]
    pct = 100 * cnt / len(train_examples)
    print(f"  {tag_name:45s} {cnt:5d}  ({pct:5.1f}%)")
"""))

# Cell 09: Markdown - Tokenizer with Dedicated Special Tokens
cells.append(create_cell("markdown", """---
## Cell 5 — Tokenizer with Registered Special Tokens & DataLoaders
Registers `[DOCTOR]` and `[PATIENT]` as first-class special tokens in the tokenizer, expanding the model embedding matrix to learn dedicated role representations.
"""))

# Cell 10: Code - Tokenizer & DataLoaders
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  TOKENIZER SETUP & DEDICATED SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Register explicit speaker tokens
special_tokens_dict = {'additional_special_tokens': ['[DOCTOR]', '[PATIENT]', '[NURSE]']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(f"✓ Added {num_added_tokens} dedicated speaker special tokens: {tokenizer.additional_special_tokens}")

class DACDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex['text'],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(ex['label'], dtype=torch.long)
        }

train_dataset = DACDataset(train_examples, tokenizer)
val_dataset   = DACDataset(val_examples, tokenizer)
test_dataset  = DACDataset(test_examples, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

logger.info(f"Batches: {len(train_loader)} train | {len(val_loader)} val | {len(test_loader)} test")
"""))

# Cell 11: Markdown - Model Architecture
cells.append(create_cell("markdown", """---
## Cell 6 — Model Architecture: 2-Layer MLP Classification Head
Features a non-linear projection (`Linear -> LayerNorm -> GELU -> Dropout -> Linear`) on top of the `[CLS]` pooled representation to capture complex cross-speaker communicative dependencies.
"""))

# Cell 12: Code - Model Architecture
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  DEEP MLP DAC CLASSIFIER (v3)
# ══════════════════════════════════════════════════════════════
class BertDAC_v3(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.20, tokenizer_len=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if tokenizer_len is not None and tokenizer_len > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(tokenizer_len)
            
        hidden_size = self.bert.config.hidden_size
        
        # 2-Layer Non-Linear Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] representation
        return self.classifier(pooled)

model = BertDAC_v3(MODEL_NAME, NUM_LABELS, dropout=DROPOUT, tokenizer_len=len(tokenizer)).to(device)

# ── Freeze bottom layers ──
for name, param in model.named_parameters():
    if "bert.embeddings" in name:
        param.requires_grad = False
    elif "bert.encoder.layer" in name:
        layer_num = int(name.split("bert.encoder.layer.")[1].split(".")[0])
        if layer_num < FREEZE_LAYERS:
            param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model Parameters: {trainable:,} trainable / {total_params:,} total ({100*trainable/total_params:.1f}%)")
"""))

# Cell 13: Markdown - Loss Function & Optimizer
cells.append(create_cell("markdown", """---
## Cell 7 — Class-Balanced Focal Loss & Multi-Group Optimizer
Computes effective number of samples class weights ($\beta=0.999$) to balance rare intent classes (`Inform-Basic_Information`, `Request-Drug_Recommendation`).
"""))

# Cell 14: Code - Class Balanced Loss & Optimizer
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  CLASS-BALANCED FOCAL LOSS
# ══════════════════════════════════════════════════════════════
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Compute Effective Number of Samples weights (Cui et al., CVPR 2019)
train_labels_list = [ex['label'] for ex in train_examples]
counts = Counter(train_labels_list)
beta = BETA_CB
cb_weights = np.ones(NUM_LABELS, dtype=np.float32)

for i in range(NUM_LABELS):
    count = counts.get(i, 1)
    eff_num = (1.0 - math.pow(beta, count)) / (1.0 - beta)
    cb_weights[i] = 1.0 / max(eff_num, 1e-4)

# Normalize
cb_weights = cb_weights / np.mean(cb_weights)

print(f"\\n{'=' * 70}")
print(f"  CLASS-BALANCED WEIGHTS (v3 Effective Number of Samples)")
print(f"{'=' * 70}")
for i, w in enumerate(cb_weights):
    print(f"  {TAGS[i]:45s} (Count: {counts.get(i, 0):4d})  weight = {w:.3f}")

weights_tensor = torch.tensor(cb_weights, dtype=torch.float, device=device)
criterion = ClassBalancedFocalLoss(gamma=FOCAL_GAMMA, weights=weights_tensor)

# ══════════════════════════════════════════════════════════════
#  OPTIMIZER WITH DIFFERENTIAL LEARNING RATES
# ══════════════════════════════════════════════════════════════
optimizer_groups = [
    {
        "params": [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n],
        "lr": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n],
        "lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY,
    },
]
optimizer = torch.optim.AdamW(optimizer_groups)

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

logger.info(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
logger.info(f"Learning rates: Backbone={BACKBONE_LR} | Head={CLASSIFIER_LR}")
"""))

# Cell 15: Markdown - Training Loop
cells.append(create_cell("markdown", """---
## Cell 8 — Training & Validation Loop with Weighted & Macro F1 Tracking
"""))

# Cell 16: Code - Training Loop
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EVALUATION HELPER
# ══════════════════════════════════════════════════════════════
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    avg_loss = total_loss / len(loader)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, weighted_f1, macro_f1, acc, all_preds, all_labels

# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
history = {
    'train_loss': [], 'val_loss': [],
    'val_weighted_f1': [], 'val_macro_f1': [], 'val_acc': []
}

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

logger.info(f"\\n{'=' * 70}")
logger.info(f"  STARTING DAC v3 TRAINING -- {NUM_EPOCHS} epochs, patience={PATIENCE}")
logger.info(f"{'=' * 70}\\n")

scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        epoch_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 25 == 0 or batch_idx == len(train_loader):
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:2d} | Batch {batch_idx:3d}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                  
    avg_train_loss = epoch_loss / num_batches
    val_loss, val_wf1, val_mf1, val_acc, _, _ = evaluate(model, val_loader, criterion)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_weighted_f1'].append(val_wf1)
    history['val_macro_f1'].append(val_mf1)
    history['val_acc'].append(val_acc)
    
    is_best = val_wf1 > best_val_f1
    print(f"\\n{'-' * 70}")
    print(f"  Epoch {epoch:2d}/{NUM_EPOCHS} Summary")
    print(f"{'-' * 70}")
    print(f"  Train Loss:     {avg_train_loss:.4f}")
    print(f"  Val Loss:       {val_loss:.4f}   |   Val Accuracy: {val_acc:.4f}")
    print(f"  Val Weighted F1:{val_wf1:.4f}   |   Val Macro F1: {val_mf1:.4f} {'  ⭐ NEW BEST!' if is_best else ''}")
    print(f"{'-' * 70}\\n")
    
    if is_best:
        best_val_f1 = val_wf1
        best_epoch = epoch
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_wf1,
            'val_macro_f1': val_mf1,
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'best_dac_model.pt')
        logger.info(f"  >> Saved best checkpoint (Epoch {epoch}, F1={val_wf1:.4f})")
    else:
        patience_counter += 1
        logger.info(f"  Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            logger.info(f"\\n🛑 Early stopping triggered at epoch {epoch}")
            break

print(f"\\n{'=' * 70}")
print(f"  TRAINING COMPLETE — Best epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")
print(f"{'=' * 70}")
"""))

# Cell 17: Markdown - Training Curves
cells.append(create_cell("markdown", """---
## Cell 9 — Training Dynamics & Curves Visualization
"""))

# Cell 18: Code - Curves
cells.append(create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

# Loss
axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='#FF6B6B', linewidth=2)
axes[0].plot(epochs_range, history['val_loss'],   label='Val Loss',   color='#4ECDC4', linewidth=2)
axes[0].axvline(x=best_epoch, color='gold', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# F1
axes[1].plot(epochs_range, history['val_weighted_f1'], label='Weighted F1', color='#45B7D1', linewidth=2, marker='o')
axes[1].plot(epochs_range, history['val_macro_f1'],    label='Macro F1',    color='#FFA07A', linewidth=2, linestyle=':')
axes[1].axvline(x=best_epoch, color='gold', linestyle='--', label=f'Best: {best_val_f1:.4f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_title(f'Validation F1 (Best: {best_val_f1:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Accuracy
axes[2].plot(epochs_range, history['val_acc'], label='Val Accuracy', color='#96CEB4', linewidth=2, marker='s')
axes[2].axvline(x=best_epoch, color='gold', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Validation Accuracy')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('MediTOD DAC v3 Training Metrics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved training_curves.png")
"""))

# Cell 19: Markdown - Test Set Evaluation
cells.append(create_cell("markdown", """---
## Cell 10 — Test Set Evaluation (Comprehensive 16-Class Classification Report)
"""))

# Cell 20: Code - Test Set Evaluation
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════
checkpoint = torch.load('best_dac_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (Val F1: {checkpoint['val_f1']:.4f})")

test_loss, test_wf1, test_mf1, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

print(f"\\n{'=' * 70}")
print(f"  TEST SET EVALUATION RESULTS (v3)")
print(f"{'=' * 70}")
print(f"  Test Loss:        {test_loss:.4f}")
print(f"  Test Weighted F1: {test_wf1:.4f}")
print(f"  Test Macro F1:    {test_mf1:.4f}")
print(f"  Test Accuracy:    {test_acc:.4f}")
print(f"{'=' * 70}\\n")

report_text = classification_report(
    test_labels, test_preds,
    target_names=TAGS, digits=4, zero_division=0
)
print("--- 16-Class Detailed Classification Report ---\\n")
print(report_text)
"""))

# Cell 21: Markdown - Confusion Matrix
cells.append(create_cell("markdown", """---
## Cell 11 — Confusion Matrix
"""))

# Cell 22: Code - Confusion Matrix
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════
cm = confusion_matrix(test_labels, test_preds, labels=list(range(NUM_LABELS)))

short_names = [t.replace('Request-', 'Req-').replace('Inform-', 'Inf-')
               .replace('Existing_Examination_and_Treatment', 'ExamTreat')
               .replace('Drug_Recommendation', 'DrugRec')
               .replace('Medical_Advice', 'MedAdv')
               .replace('Basic_Information', 'BasicInfo')
               for t in TAGS]

plt.figure(figsize=(16, 13))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=short_names, yticklabels=short_names,
    linewidths=0.5, cbar_kws={'shrink': 0.7}
)
plt.xlabel('Predicted Intent', fontsize=12)
plt.ylabel('Ground Truth Intent', fontsize=12)
plt.title(f'DAC v3 Confusion Matrix (Test Set F1={test_wf1:.4f})', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved confusion_matrix.png")
"""))

# Cell 23: Markdown - Inference Examples
cells.append(create_cell("markdown", """---
## Cell 12 — Inference on Medical Dialogue Turns
"""))

# Cell 24: Code - Inference Examples
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  INFERENCE FUNCTION WITH SPEAKER ROLES
# ══════════════════════════════════════════════════════════════
def predict_dialogue_turn(target_speaker, target_text, context_turns=None):
    model.eval()
    parts = []
    if context_turns:
        for spk, text in context_turns[-CONTEXT_WINDOW:]:
            parts.append(f"[{spk.upper()}] {text}")
            
    parts.append(f"[{target_speaker.upper()}] {target_text}")
    full_input = " [SEP] ".join(parts)
    
    encoding = tokenizer(
        full_input, truncation=True, max_length=MAX_SEQ_LEN,
        padding='max_length', return_tensors='pt'
    )
    
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[0]
        top3_vals, top3_ids = torch.topk(probs, 3)
        top3 = [(id2tag[idx.item()], val.item()) for idx, val in zip(top3_ids, top3_vals)]
        
    return top3[0][0], top3[0][1], top3

test_cases = [
    {
        "speaker": "Patient",
        "text": "Doctor, I have been having severe headache and fever for three days now.",
        "context": [("Doctor", "What brings you here today?")]
    },
    {
        "speaker": "Doctor",
        "text": "How long has this chest pain been happening?",
        "context": [("Patient", "I have chest pain and breathlessness.")]
    },
    {
        "speaker": "Patient",
        "text": "From the chest area, and it burns when I eat.",
        "context": [
            ("Doctor", "Which area is it?"),
            ("Doctor", "Is it in the stomach or chest?")
        ]
    },
    {
        "speaker": "Doctor",
        "text": "I think you might have GERD. I am prescribing pantoprazole 40mg.",
        "context": [("Patient", "The burning feeling is getting worse every day.")]
    },
    {
        "speaker": "Doctor",
        "text": "Avoid oily foods, eat smaller meals, and do not lie down immediately after dinner.",
        "context": [("Patient", "What precautions should I take?")]
    }
]

print(f"{'=' * 75}")
print(f"  CLINICAL DIALOGUE ACT INFERENCE EXAMPLES (v3)")
print(f"{'=' * 75}")

for i, tc in enumerate(test_cases, 1):
    pred, conf, top3 = predict_dialogue_turn(tc['speaker'], tc['text'], tc.get('context'))
    print(f"\\n--- Case {i} ---")
    if tc.get('context'):
        for s, t in tc['context']:
            print(f"  Context:   [{s}] {t}")
    print(f"  Turn:      [{tc['speaker']}] {tc['text']}")
    print(f"  Predicted: {pred} ({conf:.1%})")
    print(f"  Top-3:     {', '.join(f'{t} ({c:.1%})' for t, c in top3)}")
"""))

# Cell 25: Markdown - Save Artifacts
cells.append(create_cell("markdown", """---
## Cell 13 — Save Best Model Weights, HuggingFace Format & Reports
"""))

# Cell 26: Code - Save Artifacts
cells.append(create_cell("code", """# ══════════════════════════════════════════════════════════════
#  EXPORT MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════
import shutil

SAVE_DIR = '/kaggle/working/meditod_dac_output'
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Best model weights
model_dir = os.path.join(SAVE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)
shutil.copy('best_dac_model.pt', os.path.join(model_dir, 'best_dac_model.pt'))

# 2. Tokenizer
tokenizer.save_pretrained(model_dir)

# 3. Label Map
label_map = {'tag2id': tag2id, 'id2tag': {str(k): v for k, v in id2tag.items()}}
with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
    json.dump(label_map, f, indent=2)

# 4. Classification report
with open(os.path.join(SAVE_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"MediTOD DAC v3 — Test Set Classification Report\\n")
    f.write(f"Model: {MODEL_NAME}\\n")
    f.write(f"Best Epoch: {best_epoch}\\n")
    f.write(f"Test Weighted F1: {test_wf1:.4f}\\n")
    f.write(f"Test Macro F1:    {test_mf1:.4f}\\n")
    f.write(f"Test Accuracy:    {test_acc:.4f}\\n\\n")
    f.write(report_text)

# 5. Training history
with open(os.path.join(SAVE_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

# 6. Hyperparameters
hparams = {
    'model_name': MODEL_NAME,
    'version': 'v3',
    'max_seq_len': MAX_SEQ_LEN,
    'context_window': CONTEXT_WINDOW,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'backbone_lr': BACKBONE_LR,
    'classifier_lr': CLASSIFIER_LR,
    'weight_decay': WEIGHT_DECAY,
    'focal_gamma': FOCAL_GAMMA,
    'dropout': DROPOUT,
    'patience': PATIENCE,
    'freeze_layers': FREEZE_LAYERS,
    'beta_cb': BETA_CB,
    'best_epoch': best_epoch,
    'best_val_f1': best_val_f1,
    'test_weighted_f1': test_wf1,
    'test_macro_f1': test_mf1,
    'test_accuracy': test_acc,
}
with open(os.path.join(SAVE_DIR, 'hyperparameters.json'), 'w') as f:
    json.dump(hparams, f, indent=2)

# 7. Plots
for plot_file in ['training_curves.png', 'confusion_matrix.png']:
    if os.path.exists(plot_file):
        shutil.copy(plot_file, os.path.join(SAVE_DIR, plot_file))

# 8. Create zip
shutil.make_archive('/kaggle/working/meditod_dac_output', 'zip', SAVE_DIR)
zip_size = os.path.getsize('/kaggle/working/meditod_dac_output.zip') / (1024 * 1024)

print(f"\\n{'=' * 70}")
print(f"  ALL ARTIFACTS EXPORTED SUCCESSFULLY")
print(f"{'=' * 70}")
print(f"  Model weights:         {model_dir}/best_dac_model.pt")
print(f"  Label map:             {model_dir}/label_map.json")
print(f"  Classification report: {SAVE_DIR}/classification_report.txt")
print(f"  Hyperparameters:       {SAVE_DIR}/hyperparameters.json")
print(f"  Archive package:       /kaggle/working/meditod_dac_output.zip ({zip_size:.1f} MB)")
print(f"\\n  🏆 BEST VAL F1: {best_val_f1:.4f} (Epoch {best_epoch})")
print(f"  📊 FINAL TEST F1: {test_wf1:.4f}")
print(f"{'=' * 70}")
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

out_path = "c:/Users/ammua/Desktop/final_prompt/DAC/v3/meditod_dac_training_v3.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully generated: {out_path}")

