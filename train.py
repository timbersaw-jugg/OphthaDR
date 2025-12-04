#!/usr/bin/env python3
# train.py - Clean training script with dynamic downsampling support
"""
Usage:
  python train.py --config configs/config.yaml --mul 1 --epochs 35 --use-sampler
  python train.py --config configs/config.yaml --mul 4 --epochs 35 --use-class-weights

The --mul flag automatically selects data/downsampled/down_mul{N}.labels.csv
Experiment directory auto-named as: mul{N}_{sampler|classweights|default}
"""

import os
import sys
import csv
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score

# Project imports
from src.dataset import FundusDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.model import build_model
from src.loss import get_loss_fn
from src.checkpoint import CheckpointManager
from src.training import (
    train_one_epoch, evaluate, count_parameters,
    unfreeze_backbone, create_optimizer
)

def save_metrics(epoch, train_loss, train_acc, val_loss, val_acc,val_kappa, out_dir):
    """Append epoch metrics to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    
    header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'timestamp', 'val_kappa']
    row = [epoch, train_loss, train_acc, val_loss, val_acc,val_kappa, 
           datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def setup_data(cfg, force_sampler=False, force_class_weights=False):
    """Setup datasets and data loaders. Returns class_weights if requested."""
    train_tf = get_train_transforms(cfg)
    val_tf = get_val_transforms(cfg)
    
    train_ds = FundusDataset(cfg, split='train', transform=train_tf)
    val_ds = FundusDataset(cfg, split='val', transform=val_tf)
    
    batch_size = int(cfg['training'].get('batch_size', 16))
    workers = int(cfg['training'].get('num_workers', 0))
    num_classes = int(cfg['model'].get('num_classes', 5))
    
    # Get class counts
    labels = train_ds.data['level'].astype(int).values
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    
    # Compute class weights for loss (if requested)
    class_weights = None
    if force_class_weights or cfg['loss'].get('use_class_weights', False):
        inv = 1.0 / (counts + 1e-8)
        class_weights = torch.tensor((inv / inv.sum()) * num_classes, dtype=torch.float32)
        print(f"[data] Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    
    # Weighted sampler for class imbalance
    sampler = None
    use_sampler = cfg['training'].get('use_sampler', False) or force_sampler
    
    if use_sampler:
        weights = 1.0 / (counts + 1e-8)
        sample_weights = torch.tensor(weights[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print("[data] WeightedRandomSampler enabled.")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False
    )
    
    return train_ds, val_ds, train_loader, val_loader, class_weights

def compute_alpha(train_ds, num_classes, scale=1.0):
    """Compute class-balanced alpha weights for focal loss."""
    labels = train_ds.data['level'].astype(int).values
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    inv = 1.0 / (counts + 1e-12)
    alpha = (inv / inv.sum()) * scale
    alpha = alpha / alpha.sum()
    return torch.tensor(alpha, dtype=torch.float32)

def main(config_path, mul=None, epochs_override=None, exp_dir_override=None,
         force_sampler=False, force_class_weights=False, downsampled_dir="data/downsampled"):
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # ===== DYNAMIC DOWNSAMPLING =====
    if mul is not None:
        csv_path = os.path.join(downsampled_dir, f"down_mul{mul}.labels.csv")
        print(f"[train] Looking for downsampled CSV: {csv_path}")

        if os.path.exists(csv_path):
            cfg.setdefault('data', {})
            cfg['data']['label_csv'] = csv_path
            print(f"[train] 📁 Using: {csv_path}")

            # IMPORTANT: Always keep image_dir pointing to original Kaggle images
            # Downsampled CSVs DO NOT contain images.
            if 'image_dir' not in cfg['data'] or not cfg['data']['image_dir']:
                cfg['data']['image_dir'] = "data/kaggle/images/"
            print(f"[train] Using image_dir = {cfg['data']['image_dir']}")

        else:
            raise FileNotFoundError(f"Downsampled CSV for mul={mul} not found at: {csv_path}")


    
    # Auto-generate exp_dir name if not provided
    if exp_dir_override is None and mul is not None:
        balance_type = "sampler" if force_sampler else ("classweights" if force_class_weights else "default")
        exp_dir_override = f"mul{mul}_{balance_type}"
    
    # Seed
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device
    device = torch.device("cpu")
    torch_threads = int(cfg.get('runtime', {}).get('torch_num_threads', 8))
    torch.set_num_threads(torch_threads)
    print(f"[train] Device: {device}, threads: {torch.get_num_threads()}")
    
    # Data
    train_ds, val_ds, train_loader, val_loader, class_weights = setup_data(
        cfg, force_sampler, force_class_weights
    )
    
    # Model
    model = build_model(cfg, freeze_backbone=cfg['model'].get('freeze_backbone', False))
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"[train] Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
    
    # Loss
    num_classes = int(cfg['model'].get('num_classes', 5))
    alpha_tensor = None
    if cfg['training'].get('auto_compute_alpha') and cfg['training'].get('use_alpha'):
        alpha_tensor = compute_alpha(train_ds, num_classes)
        print(f"[train] Auto-computed alpha: {[f'{a:.3f}' for a in alpha_tensor.tolist()]}")
    
    if class_weights is not None:
        cfg['loss']['class_weights'] = class_weights.tolist()
    
    loss_fn = get_loss_fn(cfg, alpha_tensor)
    
    # Training params
    base_lr = float(cfg['training'].get('base_lr', 1e-4))
    weight_decay = float(cfg['training'].get('weight_decay', 0.01))
    total_epochs = epochs_override or int(cfg['training'].get('epochs', 35))
    accum_steps = int(cfg['training'].get('grad_accum_steps', 1))
    warmup_epochs = int(cfg['training'].get('warmup_epochs', 0))
    min_lr = float(cfg['training'].get('min_lr', 1e-6))
    patience_limit = int(cfg['training'].get('early_stop_patience', 15))
    log_interval = int(cfg.get('logging', {}).get('log_interval', 10))
    metrics_interval = int(cfg['training'].get('metrics_interval', 1))
    
    # Unfreeze config
    unfreeze_epoch = int(cfg['model'].get('unfreeze_after', 0))
    backbone_unfrozen = not cfg['model'].get('freeze_backbone', False)
    
    if not backbone_unfrozen and unfreeze_epoch == 0:
        print("[train] ⚠️  Backbone frozen but unfreeze_after=0. It stays frozen!")
    
    # Optimizer (Initial - using just base_lr)
    optimizer = create_optimizer(model, base_lr, weight_decay, discriminative=False)
    
    # Scheduling
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Experiment directory
    out_root = cfg.get('output', {}).get('output_dir', 'experiments')
    if exp_dir_override:
        exp_dir = exp_dir_override if os.path.isabs(exp_dir_override) else os.path.join(out_root, exp_dir_override)
    else:
        exp_dir = os.path.join(out_root, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Checkpoint manager
    ckpt_mgr = CheckpointManager(exp_dir, model, optimizer)
    ckpt_mgr.load()
    
    # Print diagnostics
    print("\n" + "="*60)
    print("TRAINING CONFIG")
    print("="*60)
    print(f"CSV: {cfg['data']['label_csv']}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Batch: {cfg['training'].get('batch_size', 16)} × {accum_steps} = {cfg['training'].get('batch_size', 16)*accum_steps} effective")
    print(f"Epochs: {total_epochs} | Steps/epoch: {steps_per_epoch}")
    print(f"LR Head: {base_lr} | Warmup: {warmup_epochs} epochs")
    print(f"Unfreeze after: epoch {unfreeze_epoch}" if unfreeze_epoch > 0 else "Backbone: unfrozen" if backbone_unfrozen else "Backbone: FROZEN")
    
    try:
        labels = train_ds.data['level'].astype(int).values
        print(f"Classes: {dict(sorted(Counter(labels).items()))}")
    except:
        pass
    print("="*60 + "\n")
    
    # Training loop
    start_epoch = ckpt_mgr.start_epoch
    best_val_acc = ckpt_mgr.best_val_acc
    
    try:
        for epoch in range(start_epoch, total_epochs):
            if ckpt_mgr.stop_requested:
                print(f"\n⚠️  Stop requested. Exiting.")
                break
            
            print(f"\n{'='*40}\nEpoch {epoch+1}/{total_epochs}\n{'='*40}")
            
            # ==========================================
            # UNFREEZE BACKBONE LOGIC (UPDATED)
            # ==========================================
            if not backbone_unfrozen and unfreeze_epoch > 0 and (epoch + 1) >= unfreeze_epoch:
                print(f"\n🔓 Unfreezing backbone at epoch {epoch+1}")
                unfreeze_backbone(model)
                backbone_unfrozen = True
                
                # Retrieve Backbone LR from config (Default to 1/100th of base if missing)
                backbone_lr = float(cfg['training'].get('backbone_lr', base_lr * 0.01))
                print(f"[train] 🧠 Enabling Discriminative LR: Backbone={backbone_lr:.2e}, Head={base_lr:.2e}")
                
                # Re-create optimizer with split learning rates
                optimizer = create_optimizer(
                    model, 
                    base_lr, 
                    weight_decay, 
                    discriminative=True,
                    backbone_lr=backbone_lr
                )
                
                ckpt_mgr.optimizer = optimizer
                trainable, _ = count_parameters(model)
                print(f"[train] Now trainable: {trainable:,}")
            
            # Train
            train_loss, train_acc, ckpt_mgr.global_step = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, ckpt_mgr,
                accum_steps=accum_steps, log_interval=log_interval,
                start_global_step=ckpt_mgr.global_step,
                total_steps=total_steps, base_lr=base_lr,
                warmup_steps=warmup_steps, min_lr=min_lr
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, loss_fn, device)
            f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
            bal_acc = balanced_accuracy_score(val_labels, val_preds)
            print(f"📈 Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_kappa={kappa:.4f}")
            
            # Extended metrics
            if (epoch + 1) % metrics_interval == 0:
                # try:
                #     f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
                #     kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
                #     bal_acc = balanced_accuracy_score(val_labels, val_preds)
                print(f"   → F1={f1:.4f} | Kappa={kappa:.4f} | BalAcc={bal_acc:.4f}")
                # except Exception as e:
                #     print(f"   → Metrics error: {e}")
            
            # Save
            save_metrics(epoch+1, train_loss, train_acc, val_loss, val_acc,kappa, exp_dir)
            
            # Check learning rate for logging
            cur_lr = optimizer.param_groups[-1]['lr'] # Take head LR
            ckpt_mgr.log_epoch(epoch+1, cur_lr, train_loss, train_acc, val_loss, val_acc)
            
            # is_best = val_acc > best_val_acc
            # if is_best:
            #     best_val_acc = val_acc
            #     ckpt_mgr.best_val_acc = best_val_acc
            #     ckpt_mgr.patience = 0
            # else:
            #     ckpt_mgr.patience += 1

            is_best = kappa > ckpt_mgr.best_val_acc
            if is_best:
                ckpt_mgr.best_val_acc = kappa
                ckpt_mgr.patience = 0
            else:
                ckpt_mgr.patience += 1

            
            lr_meta = {'base_lr': base_lr, 'total_steps': total_steps, 
                      'warmup_steps': warmup_steps, 'min_lr': min_lr}
            ckpt_mgr.save(epoch+1, train_loss, train_acc, val_loss, val_acc,
                          lr_meta=lr_meta, is_best=is_best,
                          extra_state={'backbone_unfrozen': backbone_unfrozen})
            
            if ckpt_mgr.patience >= patience_limit:
                print(f"⛔ Early stopping (patience={ckpt_mgr.patience})")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted - saving checkpoint...")
        ckpt_mgr.save(epoch+1, train_loss, train_acc, val_loss, val_acc, is_best=False)
        sys.exit(0)
    
    # print("\n" + "="*60)
    # print(f"🎉 Done! Best val_acc: {ckpt_mgr.best_val_acc:.4f}")
    # print(f"📁 Saved to: {exp_dir}")
    # print("="*60)
    print("\n" + "="*60)
    print(f"🎉 Done! Best val_kappa: {ckpt_mgr.best_val_acc:.4f}")
    print(f"📁 Saved to: {exp_dir}")
    print("="*60)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DR classification model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mul", type=int, default=None, 
                        help="Downsampling multiple (1,2,4,8)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--exp-dir", type=str, default=None)
    parser.add_argument("--downsampled-dir", type=str, default="data/downsampled")
    parser.add_argument("--use-sampler", action="store_true", 
                        help="WeightedRandomSampler for balancing")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Class weights in loss")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Override backbone LR if unfreezing")
    args = parser.parse_args()
    
    main(args.config, mul=args.mul, epochs_override=args.epochs, 
         exp_dir_override=args.exp_dir, force_sampler=args.use_sampler, 
         force_class_weights=args.use_class_weights, downsampled_dir=args.downsampled_dir)