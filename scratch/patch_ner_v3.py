import json

nb_path = 'NER/v3/meditod_ner_training_v3.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 7 (Hyperparameters)
cell_7 = nb['cells'][7]['source']
src_7 = ''.join(cell_7)
src_7 = src_7.replace("FOCAL_GAMMA       = 2.0        # v3: Focal loss gamma for hard negative mining",
                      "FOCAL_GAMMA       = 1.2        # v3: Adjusted gamma to prevent majority O-tag collapse\nALL_O_KEEP_RATIO  = 0.40       # Subsample pure negative utterances to balance entity gradients")
nb['cells'][7]['source'] = [line + '\n' for line in src_7.split('\n')[:-1]] + [src_7.split('\n')[-1]]

# 2. Update Cell 9 (build_examples)
cell_9 = nb['cells'][9]['source']
src_9 = ''.join(cell_9)
old_target_loop = """            target = turns[target_idx]
            target_tags = target['ner_tags']
            
            # Context turns"""

new_target_loop = """            target = turns[target_idx]
            target_tags = target['ner_tags']
            
            # Subsample pure all-O negative utterances during training
            if is_train:
                has_entity = any(t != 'O' for t in target_tags)
                if not has_entity and random.random() > ALL_O_KEEP_RATIO:
                    continue
            
            # Context turns"""

if old_target_loop in src_9:
    src_9 = src_9.replace(old_target_loop, new_target_loop)
    print("[OK] Successfully patched build_examples in Cell 9")
else:
    print("Warning: old_target_loop not found in Cell 9")

# Make sure random is imported in Cell 9 or Cell 3
if "import random" not in src_9:
    src_9 = "import random\n" + src_9

nb['cells'][9]['source'] = [line + '\n' for line in src_9.split('\n')[:-1]] + [src_9.split('\n')[-1]]

# 3. Update Cell 13 (ClassBalancedFocalLoss)
cell_13 = nb['cells'][13]['source']
src_13 = ''.join(cell_13)

old_focal = """class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits.view(-1, num_classes),
            targets.view(-1),
            weight=self.alpha,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()"""

new_focal = """class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=1.2, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Only compute loss on active target tokens, excluding padding and context (-100)
        valid_mask = targets_flat != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0
            
        active_logits = logits_flat[valid_mask]
        active_targets = targets_flat[valid_mask]
        
        # Unweighted cross entropy to obtain true p_t
        ce = F.cross_entropy(active_logits, active_targets, reduction='none')
        pt = torch.exp(-ce)
        focal_weight = ((1.0 - pt) ** self.gamma)
        
        if self.alpha is not None:
            alpha_t = self.alpha[active_targets]
            loss = focal_weight * alpha_t * ce
        else:
            loss = focal_weight * ce
            
        # Normalize strictly over active tokens
        return loss.mean()"""

if old_focal in src_13:
    src_13 = src_13.replace(old_focal, new_focal)
    print("[OK] Successfully patched ClassBalancedFocalLoss in Cell 13")
else:
    print("Warning: old_focal not found in Cell 13")

nb['cells'][13]['source'] = [line + '\n' for line in src_13.split('\n')[:-1]] + [src_13.split('\n')[-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Saved updated notebook.")
