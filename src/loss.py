# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with correct alpha handling.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    IMPORTANT: alpha should NOT be passed to cross_entropy's weight param,
    as that breaks the pt calculation.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                a = alpha.detach().clone().float()
            else:
                a = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha_buf', a)
        else:
            self.alpha_buf = None

    def forward(self, logits, targets):
        # Compute CE loss WITHOUT weight - this is critical!
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # pt = probability of true class
        pt = torch.exp(-ce_loss)
        
        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class weights) AFTER focal calculation
        if self.alpha_buf is not None:
            alpha = self.alpha_buf
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            # Index alpha by target class
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing and optional class weights."""
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', torch.as_tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot with smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w[targets]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_loss_fn(config, alpha_tensor=None):
    """
    Returns loss module.
    - config: full YAML config dict
    - alpha_tensor: optional torch tensor with per-class weights
    """
    loss_cfg = config.get('loss', {})
    loss_type = loss_cfg.get('type', 'focal')
    label_smoothing = float(loss_cfg.get('label_smoothing', 0.0))
    
    if loss_type == 'focal':
        gamma = float(loss_cfg.get('gamma', 2.0))
        
        # Priority: alpha_tensor param > config alpha > None
        alpha = None
        if alpha_tensor is not None:
            if isinstance(alpha_tensor, torch.Tensor):
                alpha = alpha_tensor.detach().clone().float()
            else:
                alpha = torch.as_tensor(alpha_tensor, dtype=torch.float32)
        else:
            alpha_cfg = loss_cfg.get('alpha', None)
            if alpha_cfg:
                alpha = torch.as_tensor(alpha_cfg, dtype=torch.float32)
        
        return FocalLoss(gamma=gamma, alpha=alpha)
    
    elif loss_type == 'cross_entropy':
        class_weights = loss_cfg.get('class_weights', None)
        weights = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights else None
        
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=weights)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")