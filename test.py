#!/usr/bin/env python3
# test.py - compact test + GradCAM runner (CLI)
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NUM_THREADS", "8")

import argparse
import yaml
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import traceback

from src.dataset import FundusDataset
from src.transforms import get_val_transforms
from src.model import build_model
from src.output_writer import make_record, write_outputs_json

# GradCAM compatibility helper imports (we'll try multiple signatures)
try:
    from src.gradcam import resolve_target_layer, denormalize_tensor, save_overlay, get_roi_bbox
except Exception:
    # if src.gradcam missing helpers, we'll try to import minimal functions from pytorch_grad_cam utils later
    resolve_target_layer = None
    denormalize_tensor = None
    save_overlay = None
    get_roi_bbox = None

# -------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="YAML config path")
    p.add_argument("weights", help="Path to model weights (.pth)")
    p.add_argument("--label-csv", default=None, help="Override labels CSV (path)")
    p.add_argument("--image-dir", default=None, help="Override image directory")
    p.add_argument("--output-dir", default=None, help="Output dir (default: experiments/<exp>/_test_<ts>)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", choices=["cpu","cuda","auto"], default="auto")
    p.add_argument("--target-layer", default=None, help="GradCAM target layer override (e.g. 'stages[-1]')")
    p.add_argument("--no-gradcam", dest="use_gradcam", action="store_false", help="Disable gradcam generation")
    p.add_argument("--strict-load", dest="strict_load", action="store_true", help="Load model with strict=True")
    return p.parse_args()

# Robust checkpoint loader
def robust_load_state(model, ckpt_path, strict=True, map_location=None):
    ck = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ck, dict):
        # common keys
        for k in ("model_state", "state_dict", "model", "state"):
            if k in ck:
                state = ck[k]
                break
        else:
            state = ck
    else:
        state = ck
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        # try flexible keys (strip 'module.' etc)
        new_state = {}
        for k, v in state.items() if isinstance(state, dict) else []:
            nk = k.replace("module.", "")
            new_state[nk] = v
        model.load_state_dict(new_state, strict=strict)

# Compatibility wrapper for GradCAM (tries different constructors / call signatures)
def compute_gradcam_compat(model, target_layer, input_tensor, target_class=None, prefer_cuda=False, cam_instance=None, cam_device=None):
    """
    Tries to compute GradCAM using multiple common APIs for pytorch-grad-cam and your local src.gradcam.
    - If cam_instance is supplied and usable, we'll try to reuse it.
    Returns numpy mask of shape (B, H, W) or None on failure.
    """
    # Try calling user-provided src.gradcam.compute_gradcam first if it exists (safe)
    try:
        import src.gradcam as src_gradcam
        # call with the usual signature if available
        if hasattr(src_gradcam, "compute_gradcam"):
            try:
                # Many versions accept (model, target_layer, input_tensor, target_class, use_cuda)
                return src_gradcam.compute_gradcam(model, target_layer, input_tensor, target_class=target_class, use_cuda=(prefer_cuda and torch.cuda.is_available()))
            except TypeError:
                # fallback without keyword args
                try:
                    return src_gradcam.compute_gradcam(model, target_layer, input_tensor, target_class)
                except Exception:
                    pass
    except Exception:
        pass

    # Otherwise try pytorch-grad-cam usage directly with compatibility handling
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as e:
        # no pytorch-grad-cam installed or import failed
        return None

    device = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")
    if cam_device is None:
        cam_device = device

    # ensure model on cam_device for CAM computations
    model = model.to(cam_device)
    input_tensor = input_tensor.to(cam_device)

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(int(target_class))]

    # If a cam_instance was given, try to reuse it
    if cam_instance is not None:
        try:
            # call cam_instance according to newer API
            try:
                gcam = cam_instance(input_tensor=input_tensor, targets=targets)
                return np.array(gcam)
            except TypeError:
                # older API might be cam_instance(input_tensor, targets)
                gcam = cam_instance(input_tensor, targets)
                return np.array(gcam)
        except Exception:
            # fallback to building a new CAM instance
            cam_instance = None

    # Build a fresh GradCAM instance (try multiple constructor keywords)
    cam = None
    cam_kwargs_options = [
        {"model": model, "target_layers": [target_layer], "device": cam_device},
        {"model": model, "target_layers": [target_layer], "use_cuda": (cam_device.type == "cuda")},
        {"model": model, "target_layers": [target_layer]},
    ]
    for kw in cam_kwargs_options:
        try:
            cam = GradCAM(**kw)
            break
        except TypeError:
            continue
        except Exception:
            continue
    if cam is None:
        # try alternate CAM variant
        try:
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], device=cam_device)
        except Exception:
            cam = None

    if cam is None:
        return None

    # call cam
    try:
        # new API: cam(input_tensor=input_tensor, targets=targets)
        try:
            gcam = cam(input_tensor=input_tensor, targets=targets)
        except TypeError:
            # older API: cam(input_tensor, targets)
            gcam = cam(input_tensor, targets)
        return np.array(gcam)
    except Exception as e:
        # last-ditch: try with no targets
        try:
            try:
                gcam = cam(input_tensor=input_tensor)
            except TypeError:
                gcam = cam(input_tensor)
            return np.array(gcam)
        except Exception:
            return None


def _resolve_image_id_from_meta(meta, bi, fallback):
    if meta is None:
        return fallback
    if isinstance(meta, (list, tuple)):
        m = meta[bi]
        if isinstance(m, dict):
            return m.get('image_id') or m.get('image_id_raw') or fallback
        return str(m)
    if isinstance(meta, dict):
        if 'image_id' in meta:
            return meta['image_id'][bi]
        if 'image_id_raw' in meta:
            return meta['image_id_raw'][bi]
        for k in ('image_id', 'image', 'image_id_raw'):
            if k in meta:
                return meta[k][bi]
    return str(meta)

def _save_confusion_matrix(y_true, y_pred, out_dir):
    labels_sorted = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_sorted],
                         columns=[f"pred_{l}" for l in labels_sorted])
    cm_csv_path = os.path.join(out_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    cm_png_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    xticks = [str(x) for x in labels_sorted]
    yticks = [str(x) for x in labels_sorted]
    plt.xticks(range(len(xticks)), xticks)
    plt.yticks(range(len(yticks)), yticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=150)
    plt.close()
    print(f"[test] Confusion matrix saved to: {cm_csv_path} and {cm_png_path}")
    return cm, cm_csv_path, cm_png_path

# ------------------- main -------------------
def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # override label csv / image dir from CLI if provided
    if args.label_csv:
        cfg.setdefault('data', {})['label_csv'] = args.label_csv
    if args.image_dir:
        cfg.setdefault('data', {})['image_dir'] = args.image_dir

    # output dir default
    if args.output_dir:
        out_dir = args.output_dir
    else:
        base = cfg.get('output', {}).get('output_dir', 'experiments')
        exp_tag = os.path.splitext(os.path.basename(args.weights))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(base, f"{exp_tag}_test_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    cfg.setdefault('output', {})['output_dir'] = out_dir

    # label map from config (optional)
    label_map = cfg.get('data', {}).get('label_map', None)

    # transforms & dataset
    val_tf = get_val_transforms(cfg)
    # FundusDataset expects config keys (we ensured overrides above)
    test_ds = FundusDataset(cfg, split='test', transform=val_tf, return_meta=True)
    n_test = len(test_ds)
    print(f"[test] test dataset length = {n_test}")
    if n_test == 0:
        summary = {
            "test_accuracy": None,
            "test_macro_f1": None,
            "test_kappa_quadratic": None,
            "test_balanced_acc": None,
            "n_samples": 0
        }
        with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("[test] ERROR: test dataset is empty (no split='test' rows). Wrote empty summary and exiting.")
        return

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get('training', {}).get('num_workers', 0) or 0)
    cfg_batch = int(cfg.get('testing', {}).get('batch_size', cfg.get('testing', {}).get('batch', 1)))
    batch_size = args.batch_size if args.batch_size is not None else cfg_batch
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() and cfg.get('device','auto') != 'cpu' else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[test] Using device: {device}")

    # model
    model = build_model(cfg, freeze_backbone=False).to(device)
    robust_load_state(model, args.weights, strict=args.strict_load, map_location=device)
    model.eval()

    use_gradcam = args.use_gradcam and cfg.get('output', {}).get('save_heatmaps', True)
    target_layer_expr = args.target_layer or cfg.get('gradcam', {}).get('target_layer', 'stages[-1]')
    target_layer = None
    cam_instance = None
    cam_device = None

    if use_gradcam:
        # Resolve target layer (use src.gradcam.resolve_target_layer if available, else simple resolver)
        if resolve_target_layer is not None:
            target_layer = resolve_target_layer(model, target_layer_expr)
        else:
            # fallback simple resolver
            if 'stages' in target_layer_expr and hasattr(model, 'stages'):
                target_layer = model.stages[-1]
            elif hasattr(model, target_layer_expr):
                target_layer = getattr(model, target_layer_expr)
            else:
                raise ValueError(f"Could not resolve GradCAM target layer: {target_layer_expr}")

        print(f"[test] GradCAM target layer resolved: {target_layer_expr}")

        # Try to build a reusable cam instance (best-effort) for speed
        prefer_cuda_for_cam = cfg.get('gradcam', {}).get('use_cuda', False) and (device.type == 'cuda')
        try:
            # attempt to build a compat cam via the compatibility wrapper by creating one call
            test_input = torch.zeros((1, 3, int(cfg.get('testing', {}).get('img_size', 224)),
                                         int(cfg.get('testing', {}).get('img_size', 224))), device=device)
            gmask = compute_gradcam_compat(model, target_layer, test_input, target_class=None,
                                           prefer_cuda=prefer_cuda_for_cam, cam_instance=None, cam_device=device)
            # if returned not None, we assume constructor works; we won't keep cam_instance here because creation details vary by version
            cam_instance = None
            cam_device = device
            print(f"[test] GradCAM compatibility check OK (device={cam_device})")
        except Exception as e:
            print(f"[test] GradCAM compatibility check failed: {type(e).__name__}: {e}")
            cam_instance = None
            cam_device = device

    use_cuda_for_gc = cfg.get('gradcam', {}).get('use_cuda', False) and (device.type == 'cuda')

    records = []
    all_preds = []
    all_labels = []

    # ---------- main inference loop ----------
    for idx, item in enumerate(test_loader):
        # dataset returns (images, labels, meta) or (images, labels)
        if len(item) == 3:
            images, labels, meta = item
        else:
            images, labels = item
            meta = None

        # move images to device (but keep grad enabled if use_gradcam)
        images = images.to(device)

        # compute logits
        if use_gradcam:
            logits = model(images)          # requires grad=True
        else:
            with torch.no_grad():
                logits = model(images)

        if logits.dim() > 2:
            logits = logits.view(logits.shape[0], -1)

        # ★ FIX: detach before converting to numpy
        logits_det = logits.detach().cpu()

        probs_np = torch.softmax(logits_det, dim=1).numpy()
        preds_np = logits_det.argmax(dim=1).numpy()


        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        batch_size_local = preds_np.shape[0]
        for bi in range(batch_size_local):
            pred_class = int(preds_np[bi])
            prob_vec = list(map(float, probs_np[bi]))

            true_label = None
            if labels_np.size > 0:
                true_label = int(labels_np[bi])

            fallback_id = f"idx_{idx}_{bi}"
            image_id = _resolve_image_id_from_meta(meta, bi, fallback_id)

            gradcam_heatmap_path = ""
            if use_gradcam and target_layer is not None:
                single_img = images[bi:bi+1]  # already on device

                # compute gradcam with the compat wrapper (retries multiple APIs)
                try:
                    mask = compute_gradcam_compat(model, target_layer, single_img, target_class=pred_class,
                                                  prefer_cuda=use_cuda_for_gc, cam_instance=cam_instance, cam_device=cam_device)
                except Exception as e:
                    print(f"[test] GradCAM exception for {image_id}: {type(e).__name__}: {e}")
                    mask = None

                if mask is not None:
                    # denormalize image for overlay
                    try:
                        if denormalize_tensor is not None:
                            base_img_np = denormalize_tensor(single_img[0].cpu())
                        else:
                            # fallback simple denorm (ImageNet mean/std)
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            t = single_img[0].cpu().numpy()
                            t = np.transpose(t, (1,2,0))
                            t = (t * std) + mean
                            base_img_np = np.clip(t, 0.0, 1.0).astype(np.float32)
                    except Exception:
                        base_img_np = None

                    if base_img_np is not None:
                        heatmap_path = os.path.join(cfg['output']['output_dir'], f"{os.path.splitext(str(image_id))[0]}_gradcam.png")
                        try:
                            # use save_overlay if available
                            if save_overlay is not None:
                                save_overlay(base_img_np, mask, heatmap_path)
                            else:
                                # minimal overlay saving using pytorch-grad-cam utils
                                try:
                                    from pytorch_grad_cam.utils.image import show_cam_on_image
                                    cam_mask = mask[0] if mask.ndim == 3 else mask
                                    cam_img = show_cam_on_image(base_img_np, cam_mask, use_rgb=True)
                                    import cv2
                                    cv2.imwrite(heatmap_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                                except Exception:
                                    # if saving fails, skip
                                    raise
                            gradcam_heatmap_path = heatmap_path
                        except Exception as e:
                            print(f"[test] Failed to save GradCAM overlay for {image_id}: {type(e).__name__}: {e}")
                            gradcam_heatmap_path = ""
                    else:
                        gradcam_heatmap_path = ""
                else:
                    gradcam_heatmap_path = ""

            rec = make_record(image_id=str(image_id), dr_stage=pred_class, confidence=prob_vec,
                              heatmap_path=gradcam_heatmap_path, roi_bbox=None, meta=None, label_map=label_map)
            records.append(rec)

            all_preds.append(pred_class)
            all_labels.append(true_label)

    # write outputs.json (and ndjson, csv summary)
    paths = write_outputs_json(out_dir, records)
    print(f"[test] Wrote outputs to: {paths}")

    # metrics
    all_preds = np.array(all_preds, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    valid_mask = np.array([lbl is not None for lbl in all_labels], dtype=bool)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print("[test] WARNING: No ground-truth labels found among predictions. Skipping metric computation.")
        summary = {
            "test_accuracy": None,
            "test_macro_f1": None,
            "test_kappa_quadratic": None,
            "test_balanced_acc": None,
            "n_samples": 0
        }
    else:
        y_true = all_labels[valid_mask].astype(int)
        y_pred = all_preds[valid_mask].astype(int)

        test_acc = float(accuracy_score(y_true, y_pred))
        test_macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        test_kappa = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        test_bal = float(balanced_accuracy_score(y_true, y_pred))

        summary = {
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_kappa_quadratic": test_kappa,
            "test_balanced_acc": test_bal,
            "n_samples": int(len(y_true))
        }

        cm, cm_csv, cm_png = _save_confusion_matrix(y_true, y_pred, out_dir)
        class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        report_txt = classification_report(y_true, y_pred, zero_division=0)
        report_path = os.path.join(out_dir, "classification_report.json")
        with open(report_path, "w") as fh:
            json.dump(class_report, fh, indent=2)
        print("[test] Classification report:\n", report_txt)
        print(f"[test] Classification report saved to: {report_path}")

    # write summary
    with open(os.path.join(out_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Test summary:", summary)
    print(f"Outputs and test summary saved to {out_dir}")

if __name__ == "__main__":
    main()
