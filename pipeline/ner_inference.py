import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re

class MediTODNER(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state))

class NERInference:
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(f"{model_dir}/training_config.json", 'r') as f:
            config = json.load(f)
            
        with open(f"{model_dir}/label_map.json", 'r') as f:
            self.id2label = {int(k): v for k, v in json.load(f)['id2label'].items()}
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = MediTODNER(config['model_name'], config['num_labels'])
        
        checkpoint = torch.load(f"{model_dir}/best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.max_seq_len = config['max_seq_len']
        self.min_confidence = 0.20  # Lowered from 0.4 to prevent performance drop

    def predict(self, sentence: str, history: list = None) -> list:
        # History is not fully integrated in custom inference as we focus on the sentence.
        # But we can add it to context.
        context_tokens = []
        if history:
            prefix = " ".join(history[-2:]) + " "
            context_tokens = prefix.split()
            
        sent_tokens = sentence.split()
        tokens = context_tokens + sent_tokens
        
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=self.max_seq_len, return_tensors='pt')
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0]
            preds = torch.argmax(probs, dim=-1)
            
        word_ids = encoding.word_ids()
        results = []
        prev_word_id = None
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id != prev_word_id:
                pred_id = preds[i].item()
                confidence = probs[i, pred_id].item()
                tag = self.id2label[pred_id]
                if tag != 'O' and confidence < self.min_confidence:
                    tag = 'O'
                
                # Only keep tags for the target sentence, not context
                if word_id >= len(context_tokens):
                    actual_word = sent_tokens[word_id - len(context_tokens)]
                    if tag != 'O':
                        # Convert to standard format
                        clean_tag = tag.replace('B-', '').replace('I-', '')
                        results.append({'entity': actual_word, 'type': clean_tag})
                prev_word_id = word_id
                
        # Merge adjacent same-type entities
        merged = []
        current = None
        for r in results:
            if not current:
                current = r
            else:
                if current['type'] == r['type']:
                    current['entity'] += f" {r['entity']}"
                else:
                    merged.append(current)
                    current = r
        if current:
            merged.append(current)
            
        # Clean blacklist
        blacklist = {"report", "the", "a", "an", "is", "was", "are"}
        final_results = []
        for m in merged:
            if m['entity'].lower() not in blacklist and len(m['entity']) > 1:
                final_results.append(m)
                
        return final_results
