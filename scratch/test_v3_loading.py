import io
import zipfile
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class BertDAC(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1, is_v3=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.is_v3 = is_v3
        if is_v3:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels)
            )
        else:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  
        if self.is_v3:
            return self.classifier(pooled)
        return self.classifier(self.dropout(pooled))

class MediTODNER(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state))

print("Testing DAC v3 loading from zip...")
z_dac = zipfile.ZipFile(r'c:\Users\ammua\Desktop\final_prompt\DAC\v3\v3_output.zip')
ckpt_dac = torch.load(io.BytesIO(z_dac.read('model/best_dac_model.pt')), map_location='cpu', weights_only=False)
state_dac = ckpt_dac.get('model_state_dict', ckpt_dac)
is_v3_dac = any('classifier.0' in k for k in state_dac.keys())
print(f"Detected is_v3 for DAC: {is_v3_dac}")

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
dac_model = BertDAC(model_name, 16, is_v3=is_v3_dac)
dac_model.load_state_dict(state_dac)
print("DAC v3 successfully loaded state_dict!")

print("\nTesting NER v3 loading from zip...")
z_ner = zipfile.ZipFile(r'c:\Users\ammua\Desktop\final_prompt\NER\v3\v3_output.zip')
ckpt_ner = torch.load(io.BytesIO(z_ner.read('model/best_model.pt')), map_location='cpu', weights_only=False)
state_ner = ckpt_ner.get('model_state_dict', ckpt_ner)
ner_model = MediTODNER(model_name, 25)
ner_model.load_state_dict(state_ner)
print("NER v3 successfully loaded state_dict!")

print("\nVERDICT: BOTH v3 MODELS CAN BE LOADED PERFECTLY.")

