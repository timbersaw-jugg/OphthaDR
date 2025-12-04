import timm
import torch
from pathlib import Path
import os


def build_model(config, freeze_backbone=False):
    arch = config['model']['arch']
    num_classes = config['model']['num_classes']
    pretrained_weights = config['model'].get('pretrained_weights', None)

    # Check if pretrained_weights is a valid local file
    is_local_file = False
    if pretrained_weights and isinstance(pretrained_weights, str):
        # Check absolute path or relative to current working directory
        if os.path.exists(pretrained_weights):
            is_local_file = True

    print(f"[model] Building {arch}...")

    if is_local_file:
        print(f"[model] 📂 Loading local weights from: {pretrained_weights}")

        # Create model with pretrained=False (no download) and the target num_classes
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

        # Load checkpoint with robust handling of common formats
        ckpt = torch.load(pretrained_weights, map_location='cpu')
        # some checkpoints nest the state dict under 'model' or 'state_dict'
        if isinstance(ckpt, dict) and ('model' in ckpt or 'state_dict' in ckpt):
            state_dict = ckpt.get('model', ckpt.get('state_dict'))
        else:
            state_dict = ckpt

        # If state_dict contains a top-level key like 'state_dict' (PyTorch Lightning style), unwrap one layer
        if isinstance(state_dict, dict) and all(k.startswith('state_dict.') for k in state_dict.keys()):
            # convert 'state_dict.encoder.weight' -> 'encoder.weight'
            state_dict = {k.replace('state_dict.', ''): v for k, v in state_dict.items()}

        # Defensive: ensure we have a plain mapping of parameter name -> tensor
        if not isinstance(state_dict, dict):
            raise RuntimeError("Loaded checkpoint does not contain a state_dict-like mapping")

        # Remove head/classifier weights that are incompatible with our num_classes
        removed = []
        for k in list(state_dict.keys()):
            # common names to remove: 'head.', 'classifier', 'fc', 'head.fc', 'head.weight' etc.
            # we target keys that refer to the classification head only
            low = k.lower()
            if low.startswith('head.') or '.head.' in low or 'classifier' in low or 'head_fc' in low or 'head.fc' in low:
                removed.append(k)
                del state_dict[k]

        if removed:
            print(f"[model] ⚠️ Removed {len(removed)} head keys from checkpoint: {removed}")

        # Attempt to load remaining weights
        load_msg = model.load_state_dict(state_dict, strict=False)
        # load_state_dict returns a namedtuple / dict-like with missing_keys/unexpected_keys in modern PyTorch
        try:
            # when it's a dict-like (PyTorch 1.8+), show the most important info
            missing = getattr(load_msg, 'missing_keys', None) or load_msg.get('missing_keys', None)
            unexpected = getattr(load_msg, 'unexpected_keys', None) or load_msg.get('unexpected_keys', None)
            if missing:
                print(f"[model] Missing keys after load (expected - e.g. new head): {missing}")
            if unexpected:
                print(f"[model] Unexpected keys in checkpoint (ignored): {unexpected}")
        except Exception:
            # Fallback: just print the raw return
            print(f"[model] load_state_dict returned: {load_msg}")

    else:
        # No local file provided, try downloading from Internet (ImageNet)
        print(f"[model] 🌐 Attempting to download ImageNet weights...")
        try:
            model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
            print("[model] ✅ Download/Cache successful.")
        except Exception as e:
            print(f"[model] ❌ Download failed: {e}")
            print(f"[model] ⚠️ Creating model with RANDOM initialization.")
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)

    # Freeze backbone logic
    if freeze_backbone:
        print("[model] 🔒 Freezing backbone layers...")
        classifier_keywords = ['head', 'classifier', 'fc', 'dense', 'norm']
        for name, param in model.named_parameters():
            if not any(k in name.lower() for k in classifier_keywords):
                param.requires_grad = False

    return model
