# src/checkpoint.py
"""Checkpoint management with graceful stopping and resume support."""

import os
import sys
import json
import csv
import signal
from pathlib import Path
from datetime import datetime

import torch


class CheckpointManager:
    """Handles checkpoint saving, loading, and graceful stop signals."""
    
    def __init__(self, exp_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.exp_dir = Path(exp_dir)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        
        # State tracking
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_acc = -1.0
        self.patience = 0
        self.stop_requested = False
        
        # Setup
        self._setup_signal_handlers()
        self.log_csv = self.exp_dir / "training_progress.csv"
        self._init_log_file()

    def _setup_signal_handlers(self):
        def handler(sig, frame):
            if not self.stop_requested:
                print("\n" + "="*50)
                print("STOP signal received (Ctrl+C). Will finish current epoch and save.")
                print("Press Ctrl+C again to force exit.")
                print("="*50)
                self.stop_requested = True
            else:
                print("\nSecond interrupt - exiting immediately.")
                sys.exit(1)
        
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def _init_log_file(self):
        if not self.log_csv.exists():
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'global_step', 'lr',
                    'train_loss', 'train_acc', 'val_loss', 'val_acc',
                    'best_val_acc', 'patience'
                ])

    def log_epoch(self, epoch, lr, train_loss, train_acc, val_loss, val_acc):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, epoch, self.global_step, lr,
                train_loss, train_acc, val_loss, val_acc,
                self.best_val_acc, self.patience
            ])

    def save(self, epoch, train_loss, train_acc, val_loss, val_acc, 
             lr_meta=None, is_best=False, extra_state=None):
        """Save checkpoint with all training state."""
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else None
        
        ck = {
            'epoch': int(epoch),
            'global_step': int(self.global_step),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'best_val_acc': float(self.best_val_acc),
            'patience': int(self.patience),
            'current_lr': float(current_lr) if current_lr else None,
            'lr_meta': lr_meta or {},
            'extra_state': extra_state or {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save latest and epoch-specific
        torch.save(ck, self.checkpoint_dir / "latest.pth")
        torch.save(ck, self.checkpoint_dir / f"epoch_{epoch}.pth")
        
        if is_best:
            torch.save(ck, self.checkpoint_dir / "best.pth")
            torch.save(self.model.state_dict(), self.exp_dir / "best_model.pth")
            print(f"✅ New best model (val_acc={val_acc:.4f}) saved.")
        
        # Metadata JSON for quick inspection
        with open(self.checkpoint_dir / "metadata.json", "w") as f:
            json.dump({
                'last_epoch': epoch,
                'global_step': self.global_step,
                'best_val_acc': self.best_val_acc,
                'patience': self.patience,
                'lr': current_lr,
                'time': ck['timestamp']
            }, f, indent=2)
        
        print(f"💾 Checkpoint saved: epoch={epoch}, lr={current_lr:.6e}")

    def load(self, custom_path: str = None):
        """Load checkpoint and restore state. Returns lr_meta dict or False."""
        ck_path = Path(custom_path) if custom_path else (self.checkpoint_dir / "latest.pth")
        
        if not ck_path.exists():
            print("ℹ️  No checkpoint found to resume.")
            return False
        
        try:
            ck = torch.load(str(ck_path), map_location='cpu')
            
            # Load model state
            try:
                self.model.load_state_dict(ck['model_state'])
            except Exception as e:
                print(f"⚠️  Warning loading model state: {e}")
            
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(ck['optimizer_state'])
            except Exception as e:
                print(f"⚠️  Warning loading optimizer state: {e}")
            
            # Restore tracking state
            self.start_epoch = int(ck.get('epoch', 0))
            self.global_step = int(ck.get('global_step', 0))
            self.best_val_acc = float(ck.get('best_val_acc', -1.0))
            self.patience = int(ck.get('patience', 0))
            
            print("\n" + "="*60)
            print(f"📂 Resumed from: {ck_path}")
            print(f"   epoch={self.start_epoch}, step={self.global_step}, best_acc={self.best_val_acc:.4f}")
            print("="*60 + "\n")
            
            return ck.get('lr_meta', {}) or {}
            
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            return False

    def check_stop_file(self):
        """Check if STOP file exists to trigger graceful shutdown."""
        stop_file = self.exp_dir / "STOP"
        if stop_file.exists():
            print("\n⚠️  STOP file detected. Will finish current epoch and stop.")
            try:
                stop_file.unlink()
            except Exception:
                pass
            self.stop_requested = True