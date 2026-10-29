# Medical NER Model Training Specification — Kaggle Edition

> **Dataset:** `final_200_bio.jsonl` — 190 dialogues, 4,759 turns, 148,110 BIO tokens  
> **Task:** 12-class Token Classification (+ O tag = 25 BIO labels)  
> **Target Platform:** Kaggle Notebook (1x T4 GPU / 2x T4 / 1x P100)

---

## 1. Task Definition
Train a sequence labeling (Token Classification) model to extract **12 distinct medical entities** from colloquial Indian English clinical dialogues. The model reads conversational turns (with 2-turn context window) and predicts BIO labels for every token.

---

## 2. Model Selection — Recommended: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

### Why PubMedBERT over alternatives?

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| **PubMedBERT** | Pre-trained on 21M PubMed abstracts + 3.1B words. Vocabulary includes medical terms natively (`metformin`, `breathlessness`, `hypertension`). Best-in-class on BLURB biomedical benchmark. | Doesn't understand colloquial Indian English ("it is paining", "sugar problem"). | ✅ **RECOMMENDED — Best tradeoff** |
| **BioClinicalBERT** | Trained on MIMIC-III clinical notes — understands patient records. | MIMIC notes are American English, formal style. Very different from our Indian English dialogues. Also outdated (2019 weights). | ⚠️ Good but less aligned |
| **RoBERTa-base** | Strong general English understanding, handles colloquial phrasing well. | No medical pre-training. Will need more epochs to learn "metformin" = medication. | ⚠️ Good baseline |
| **bert-base-uncased** | Fast, well-understood, easy to debug. | Weakest medical vocabulary. | ❌ Only for pipeline validation |
| **GatorTron / BioGPT** | Massive medical pre-training. | Too large for Kaggle T4 (16GB VRAM limit). | ❌ Out of scope |

### Model Loading

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=25,       # 12 tags x 2 (B-/I-) + O
    id2label=id2label,
    label2id=label2id,
)
```

---

## 3. The 25 BIO Labels

```python
LABEL_LIST = [
    "O",
    "B-SYMPTOM_VALUE",        "I-SYMPTOM_VALUE",
    "B-SYMPTOM_LOCATION",     "I-SYMPTOM_LOCATION",
    "B-SYMPTOM_CHARACTERISTICS", "I-SYMPTOM_CHARACTERISTICS",
    "B-SYMPTOM_FREQUENCY",    "I-SYMPTOM_FREQUENCY",
    "B-TEMPORAL",             "I-TEMPORAL",
    "B-MEDICATION",           "I-MEDICATION",
    "B-MEDICAL_TEST_NAME",    "I-MEDICAL_TEST_NAME",
    "B-MEDICAL_HISTORY_VALUE","I-MEDICAL_HISTORY_VALUE",
    "B-HABIT_VALUE",          "I-HABIT_VALUE",
    "B-HABIT_FREQUENCY",      "I-HABIT_FREQUENCY",
    "B-EXPOSURE_VALUE",       "I-EXPOSURE_VALUE",
    "B-POLARITY",             "I-POLARITY",
]
```

---

## 4. Context Window Strategy — Turn History = 2

### Why Context Matters
In medical dialogues, critical entities are **cross-referential**:
- `"Yes"` is POLARITY only because the previous turn asked `"Any fever?"`
- `"last night"` is TEMPORAL anchored to the symptom mentioned 2 turns ago
- `"the same one"` refers to a medication mentioned earlier

### Implementation

For each target utterance U_i, construct the model input as:

```
Input = [CLS] [SPK] U_{i-2} [SEP] [SPK] U_{i-1} [SEP] [SPK] U_i [SEP]
Labels =  -100  -100  ...-100   -100  -100  ...-100   -100  -100  LABELS  -100
```

- **Context turns (U_{i-2}, U_{i-1})**: Labels are set to -100 (ignored by loss)
- **Target turn (U_i)**: Labels are the actual BIO tags
- **Speaker token [SPK]**: Prepend `Doctor:` or `Patient:` as context

### Key Rules
1. Context turns are **read-only** — they provide disambiguation but NO labels
2. If i < 2, use only available history (pad from start of dialogue)
3. Max sequence length = **256 tokens** (context + target combined)
4. Truncate context from the LEFT (oldest turn) if sequence exceeds 256

```python
def build_context_window(dialogue_turns, target_idx, history=2):
    """Build [context_turns, target_turn] with labels only on target."""
    start = max(0, target_idx - history)
    context_tokens = []
    context_labels = []
    
    # Context turns (labels = -100)
    for i in range(start, target_idx):
        turn = dialogue_turns[i]
        speaker_prefix = f"{turn['speaker']}:"
        tokens = [speaker_prefix] + turn['tokens']
        context_tokens.extend(tokens)
        context_labels.extend([-100] * len(tokens))
    
    # Target turn (actual labels)
    target = dialogue_turns[target_idx]
    speaker_prefix = f"{target['speaker']}:"
    target_tokens = [speaker_prefix] + target['tokens']
    target_labels = [-100] + target['ner_tag_ids']  # -100 for speaker prefix
    
    all_tokens = context_tokens + target_tokens
    all_labels = context_labels + target_labels
    
    return all_tokens, all_labels
```

---

## 5. Layer Freezing Strategy

### Recommended: Freeze bottom 6 layers, fine-tune top 6 + classifier head

With 190 dialogues (~4,750 training examples), we have a **medium-sized** dataset. Full fine-tuning risks overfitting to synthetic patterns while losing PubMedBERT's medical vocabulary knowledge.

```python
# Freeze embedding layer + bottom 6 transformer blocks
for name, param in model.named_parameters():
    if "embeddings" in name:
        param.requires_grad = False
    elif "encoder.layer" in name:
        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
        if layer_num < 6:  # Freeze layers 0-5
            param.requires_grad = False

# Verify: only top 6 layers + classifier are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Expected: ~55% trainable
```

### Why this split?
- **Bottom 6 (frozen)**: Already understand medical vocabulary, subword tokenization, syntax
- **Top 6 (trainable)**: Learn our specific entity schema, Indian English patterns, dialogue structure
- **Classifier head (trainable)**: Maps to our 25 BIO labels

---

## 6. Loss Function — Focal Loss + O-Tag Downweighting

### Why not standard CrossEntropy?
Our dataset has 83.5% O-tokens vs 16.5% entity tokens. Standard CE will over-optimize for predicting O and under-detect rare entities.

### Focal Loss Configuration

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.8, alpha=None, label_smoothing=0.03):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha  # Per-class weights
    
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            targets.view(-1),
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none',
            ignore_index=-100,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# O-tag weight configuration
O_TAG_WEIGHT = 0.20
ENTITY_TAG_WEIGHT = 1.0
FOCAL_GAMMA = 1.8
LABEL_SMOOTHING = 0.03

class_weights = torch.ones(25)
class_weights[label2id["O"]] = O_TAG_WEIGHT

criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    alpha=class_weights.to(device),
    label_smoothing=LABEL_SMOOTHING,
)
```

### Hyperparameter Rationale (from Phase 5 of previous training)

| Parameter | Value | Why |
|---|---|---|
| `O_TAG_WEIGHT` | 0.20 | Down from 0.25 — more entity focus |
| `FOCAL_GAMMA` | 1.8 | Up from 1.5 — harder focus on uncertain tokens |
| `LABEL_SMOOTHING` | 0.03 | Down from 0.05 — sharper boundary decisions |

---

## 7. O-Tag Subsampling (ALL_O_KEEP_RATIO)

### Strategy
Utterances containing **zero entity tokens** (100% O-tags) are randomly dropped from training targets with probability 0.75.

```python
ALL_O_KEEP_RATIO = 0.25

def should_train_on_utterance(ner_tags):
    has_entity = any(tag != "O" for tag in ner_tags)
    if has_entity:
        return True
    return random.random() < ALL_O_KEEP_RATIO
```

### Important
- Dropped utterances are **excluded from loss** — NOT from the context window
- They still appear as U_{i-1} or U_{i-2} for the next training target
- Re-sampled every epoch (different subset each time)

---

## 8. CRF Layer — Use It, But Correctly

### Lesson from Phase 4-5: Architecture Parity

In previous training, the CRF layer improved BIO consistency during training but was **silently discarded** at inference by HF pipeline. The fix: ensure inference uses the same CRF Viterbi decoding.

### Recommended: Use CRF with custom inference (not HF pipeline)

```python
from torchcrf import CRF  # pip install pytorch-crf

class BertCRFModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        
        if labels is not None:
            mask = (labels != -100)
            safe_labels = labels.clone()
            safe_labels[~mask] = 0
            loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
            return loss
        else:
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
```

### CRF + Focal Loss adjustments
- Set `FOCAL_GAMMA = 1.5` (slightly lower)
- Set `O_TAG_WEIGHT = 0.25` (slightly higher for CRF transitions)

---

## 9. Complete Hyperparameter Table

| Parameter | Value | Notes |
|---|---|---|
| **Model** | `PubMedBERT-base-uncased-abstract-fulltext` | 110M params |
| **Max Sequence Length** | 256 | Context + target |
| **Context Window** | 2 preceding turns | |
| **Batch Size** | 16 | |
| **Gradient Accumulation** | 2 | Effective batch = 32 |
| **Learning Rate** | 3e-5 | For unfrozen layers |
| **LR Schedule** | Linear decay with warmup | |
| **Warmup** | 10% of total steps | ~475 steps for 10 epochs |
| **Weight Decay** | 0.01 | AdamW |
| **Epochs** | 15 (max) | Early stop patience=5 |
| **Early Stop Metric** | Val Entity-F1 (strict) | |
| **Frozen Layers** | Embeddings + layers 0-5 | |
| **Dropout** | 0.1 | |
| **Loss** | Focal Loss | gamma=1.8, smoothing=0.03 |
| **O-Tag Weight** | 0.20 | |
| **ALL_O_KEEP_RATIO** | 0.25 | |
| **CRF** | Optional | Use with custom inference |
| **Train/Val Split** | 85/15 by dialogue | Stratified |
| **Evaluation** | seqeval strict span F1 | |

### Ablation Runs

| Run | Change | Purpose |
|---|---|---|
| Run 2 | Unfreeze ALL layers, LR=2e-5, epochs=8 | Full fine-tune test |
| Run 3 | RoBERTa-base instead of PubMedBERT | Baseline comparison |
| Run 4 | Context window = 0 | Measure context contribution |
| Run 5 | Add CRF layer | BIO consistency improvement |
| Run 6 | Context window = 3 | More context |

---

## 10. Train/Val Split

Split by **dialogue** (not turn) to avoid data leakage:

```python
from sklearn.model_selection import train_test_split

train_ids, val_ids = train_test_split(
    dialogue_ids, test_size=0.15, random_state=42,
    stratify=["low" if t < 70 else "high" if t > 95 else "med" for t in tags_per_dialogue]
)
# ~161 train, ~29 val dialogues
```

---

## 11. Evaluation

```python
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

report = classification_report(all_labels, all_preds, scheme=IOB2)
f1 = f1_score(all_labels, all_preds, scheme=IOB2)
```

---

## 12. Post-Processing (Inference)

```python
def postprocess_predictions(tokens, predictions):
    cleaned = []
    for tok, pred in zip(tokens, predictions):
        if tok.startswith("##"):
            continue
        if pred.startswith("I-") and (not cleaned or not cleaned[-1][1].endswith(pred[2:])):
            pred = "B-" + pred[2:]
        blacklist = {"the", "a", "an", "is", "are", "was", "were", "report"}
        if tok.lower() in blacklist and pred != "O":
            pred = "O"
        cleaned.append((tok, pred))
    return cleaned
```

---

## 13. Kaggle Notebook Cell Structure

```
Cell 1: pip install transformers datasets seqeval pytorch-crf accelerate
Cell 2: Upload & parse final_200_bio.jsonl, build context windows
Cell 3: Tokenization & BIO label alignment (subword handling)
Cell 4: Model init (PubMedBERT + freeze bottom 6 + optional CRF)
Cell 5: FocalLoss + O-tag weights + AdamW + scheduler
Cell 6: Training loop with ALL_O_KEEP_RATIO + epoch eval
Cell 7: Per-entity P/R/F1 report + confusion matrix
Cell 8: Save best checkpoint + tokenizer + label map
```

---

## 14. Expected Performance

| Metric | Phase 5 (old) | Expected (new) |
|---|---|---|
| **Overall Entity F1** | ~0.65 | **0.72-0.78** |
| **SYMPTOM_VALUE F1** | ~0.70 | **0.78-0.85** |
| **MEDICATION F1** | ~0.75 | **0.82-0.88** |
| **TEMPORAL F1** | ~0.55 | **0.65-0.72** |
| **POLARITY F1** | ~0.50 | **0.60-0.68** |
