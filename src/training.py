# src/training.py
"""Training utilities: epoch loop, evaluation, LR scheduling, model helpers."""

import math
import time
from collections import deque

import torch
from torch.optim import AdamW


# ============================================================
# Average Meter
# ============================================================
class AverageMeter:
    """Tracks running average of a metric."""
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)
    
    @property
    def avg(self):
        return (self.sum / self.cnt) if self.cnt > 0 else 0.0
    
    def reset(self):
        self.sum = 0.0
        self.cnt = 0


# ============================================================
# Learning Rate Scheduling
# ============================================================
def cosine_lr(optimizer, base_lr, step, total_steps, warmup_steps=0, min_lr=1e-6):
    """Apply cosine annealing with optional warmup."""
    if total_steps <= 0:
        return
    
    # Calculate the global learning rate factor (0.0 to 1.0 * base_lr)
    if step < warmup_steps and warmup_steps > 0:
        lr = base_lr * (step / float(max(1, warmup_steps)))
    else:
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = max(0.0, min(1.0, t))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    
    # Apply to optimizer groups
    for g in optimizer.param_groups:
        # If the group has a specific scaling factor (for backbone), use it
        if 'lr_scale' in g:
            g['lr'] = lr * g['lr_scale']
        else:
            g['lr'] = lr


# ============================================================
# Model Parameter Utilities
# ============================================================
def count_parameters(model):
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def unfreeze_backbone(model, verbose=True):
    """Unfreeze all frozen parameters in the model."""
    unfrozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen += 1
    if verbose and unfrozen > 0:
        print(f"🔓 Unfroze {unfrozen} parameters. All layers now trainable.")
    return unfrozen


def create_optimizer(model, base_lr, weight_decay=0.01, discriminative=False, backbone_lr=None):
    """
    Create AdamW optimizer.
    
    If discriminative=True, uses lower LR for backbone.
    If backbone_lr is not provided, defaults to 0.01x (1%) of base_lr.
    """
    if not discriminative:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return AdamW(params, lr=base_lr, weight_decay=weight_decay)
    
    # Determine Backbone LR logic
    if backbone_lr is None:
        backbone_lr = base_lr * 0.01  # Default to 1% if not specified
    
    # Calculate the scale ratio for the scheduler
    # Example: backbone=1e-6, head=1e-4 -> scale = 0.01
    backbone_scale = backbone_lr / base_lr
    
    # Discriminative parameter separation
    classifier_keywords = ['head', 'classifier', 'fc', 'dense', 'norm']
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            is_head = any(k in name.lower() for k in classifier_keywords)
            if is_head:
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    # Define groups with 'lr_scale' for the cosine_lr scheduler to use
    param_groups = [
        {
            'params': backbone_params, 
            'lr': backbone_lr, 
            'lr_scale': backbone_scale, # Vital for scheduler to maintain ratio
            'name': 'backbone'
        },
        {
            'params': head_params, 
            'lr': base_lr, 
            'lr_scale': 1.0, 
            'name': 'head'
        }
    ]
    
    print(f"[optimizer] Discriminative LR: backbone={backbone_lr:.2e} (scale={backbone_scale:.4f}), head={base_lr:.2e}")
    print(f"[optimizer] Params: backbone={len(backbone_params)}, head={len(head_params)}")
    
    return AdamW(param_groups, weight_decay=weight_decay)


# ============================================================
# Training Loop (Single Epoch)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, ckpt_mgr,
                    accum_steps=1, log_interval=10, start_global_step=0,
                    total_steps=None, base_lr=None, warmup_steps=0, min_lr=1e-6):
    """
    Train for one epoch with gradient accumulation and LR scheduling.
    
    Returns: (avg_loss, avg_acc, final_global_step)
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    optimizer.zero_grad()
    
    global_step = int(start_global_step)
    total_steps = total_steps or 1
    # Use provided base_lr, or fallback to optimizer default
    base_lr = base_lr or optimizer.param_groups[-1].get('lr', 1e-4)
    
    batch_times = deque(maxlen=100)
    epoch_start = time.time()
    total_batches = len(loader)

    for step, batch in enumerate(loader):
        # Handle both (images, labels) and (images, labels, meta) formats
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        
        # Check for stop request
        if ckpt_mgr.stop_requested:
            print("⚠️  Stop requested - breaking epoch early.")
            break
        
        if step % 50 == 0:
            ckpt_mgr.check_stop_file()

        t0 = time.time()
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) / float(accum_steps)
        loss.backward()

        # Optimizer step with gradient accumulation
        if (step + 1) % accum_steps == 0:
            # Update LR based on step
            cosine_lr(optimizer, base_lr, global_step, total_steps, warmup_steps, min_lr)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Metrics
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item() * accum_steps, images.size(0))
        acc_meter.update(acc, images.size(0))

        batch_times.append(time.time() - t0)

        # Logging
        if (step + 1) % log_interval == 0 or (step + 1) == total_batches:
            avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
            eta = (total_batches - step - 1) * avg_batch
            eta_str = _format_time(eta)
            
            # Get current LRs for display
            lrs = [g['lr'] for g in optimizer.param_groups]
            lr_str = f"{lrs[-1]:.2e}" # Head LR
            if len(lrs) > 1:
                lr_str += f" (BB: {lrs[0]:.2e})" # Backbone LR if different
                
            print(f"  Batch {step+1}/{total_batches} — loss={loss_meter.avg:.4f} "
                  f"acc={acc_meter.avg:.4f} lr={lr_str} step={global_step} ETA {eta_str}")

    # Handle leftover gradients
    if total_batches % accum_steps != 0:
        cosine_lr(optimizer, base_lr, global_step, total_steps, warmup_steps, min_lr)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

    duration = time.time() - epoch_start
    avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
    print(f"⏱️  Epoch finished — duration: {duration:.1f}s — avg batch: {avg_batch:.3f}s")
    
    return loss_meter.avg, acc_meter.avg, global_step


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a data loader.
    
    Returns: (avg_loss, avg_acc, all_preds, all_labels)
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_preds = []
    all_labels = []
    
    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    return loss_meter.avg, acc_meter.avg, all_preds, all_labels


# ============================================================
# Helpers
# ============================================================
def _format_time(seconds):
    """Format seconds into human readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"