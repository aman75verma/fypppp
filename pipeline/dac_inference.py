import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BertDAC(nn.Module):
    """
    Universal DAC Model supporting both:
    - v1/v2 Architecture: Single Linear classification head (vocab: 30,522)
    - v3 Architecture: 2-Layer MLP Non-Linear Head with dedicated speaker tokens (vocab: 30,525)
    """
    def __init__(self, model_name, num_labels, dropout=0.1, is_v3=False, tokenizer_len=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if tokenizer_len is not None and tokenizer_len > self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(tokenizer_len)
            
        hidden_size = self.bert.config.hidden_size
        self.is_v3 = is_v3
        
        if is_v3:
            # v3: 2-Layer Non-Linear Classification Head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels)
            )
        else:
            # v1/v2: Single Linear Head
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  
        if self.is_v3:
            return self.classifier(pooled)
        return self.classifier(self.dropout(pooled))

class DACInference:
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load label map
        label_map_file = os.path.join(model_dir, 'label_map.json')
        if os.path.exists(label_map_file):
            with open(label_map_file, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                if 'id2label' in label_map:
                    self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
                elif 'id2tag' in label_map:
                    self.id2label = {int(k): v for k, v in label_map['id2tag'].items()}
                elif 'tag2id' in label_map:
                    self.id2label = {int(v): k for k, v in label_map['tag2id'].items()}
                else:
                    TAGS = ['Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose', 'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice', 'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions', 'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom', 'Inform-Symptom', 'Other']
                    self.id2label = {i: t for i, t in enumerate(TAGS)}
        else:
            TAGS = ['Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose', 'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice', 'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions', 'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom', 'Inform-Symptom', 'Other']
            self.id2label = {i: t for i, t in enumerate(TAGS)}
            
        num_labels = len(self.id2label)
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        # Load checkpoint to inspect architecture
        ckpt_path = os.path.join(model_dir, "best_dac_model.pt")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Detect if checkpoint is v3 (2-layer MLP head)
        self.is_v3 = any('classifier.0' in k for k in state_dict.keys())
        
        # Setup tokenizer
        if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.is_v3:
                self.tokenizer.add_special_tokens({'additional_special_tokens': ['[DOCTOR]', '[PATIENT]', '[NURSE]']})
        
        # Instantiate model with adapter support
        dropout = 0.2 if self.is_v3 else 0.1
        self.model = BertDAC(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            is_v3=self.is_v3,
            tokenizer_len=len(self.tokenizer)
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.max_seq_len = 256 if self.is_v3 else 128
        self.context_window = 3 if self.is_v3 else 2
        print(f"[OK] DACInference initialized successfully (Mode: {'v3 (Deep MLP + Speaker Tokens)' if self.is_v3 else 'v1/v2 (Linear Head)'}, Device: {self.device})")

    def predict(self, sentence: str, history: list = None, speaker: str = None) -> str:
        parts = []
        if history:
            for turn in history[-self.context_window:]:
                parts.append(turn)
        
        if self.is_v3 and speaker:
            spk_token = f"[{speaker.upper()}]"
            curr_text = f"{spk_token} {sentence}" if not sentence.startswith("[") else sentence
            parts.append(curr_text)
        else:
            parts.append(sentence)
            
        full_input = " [SEP] ".join(parts)
        
        encoding = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
        return self.id2label[preds.item()]
