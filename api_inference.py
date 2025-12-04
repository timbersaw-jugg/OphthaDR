# api_inference.py
import yaml
import torch
import numpy as np
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from src.model import build_model
from src.transforms import get_val_transforms
from src.gradcam import (
    resolve_target_layer, compute_gradcam_with_stats, denormalize_tensor,
    save_overlay, save_raw_mask_npy
)
from src.output_writer import make_record, write_outputs_json
import tempfile
from PIL import Image
import json
import base64
import cv2

# -----------------------------
# Load config, model, transforms
# -----------------------------
config_path = "configs/config.yaml"
weights_path = "experiments/mul4_sampler/best_model.pth"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")

model = build_model(cfg)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

transforms = get_val_transforms(cfg)
target_layer = resolve_target_layer(model, cfg["gradcam"].get("target_layer", "stages[-1]"))
use_cuda = cfg["gradcam"].get("use_cuda", False) and torch.cuda.is_available()
output_dir = Path(cfg["output"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
ndjson_path = output_dir / "predictions.ndjson"

# fastapi app
app = FastAPI()

# -----------------------------
# Small helpers
# -----------------------------
def compute_flags(probs, stats):
    flags = []
    top1 = float(max(probs)) if probs is not None else 0.0
    sorted_probs = sorted([float(x) for x in probs]) if probs is not None else [0.0]
    top2 = sorted_probs[-2] if len(sorted_probs) > 1 else 0.0
    if top1 < 0.6:
        flags.append("low_confidence")
    if (top1 - top2) < 0.15:
        flags.append("ambiguous_prediction")
    if stats:
        if stats.get("leakage_fraction", 0.0) > 0.15:
            flags.append("high_leakage")
        ma = stats.get("mask_area_frac", 0.0)
        if ma > 0.20 or ma < 0.005:
            flags.append("diffuse_or_tiny_activation")
    return flags

def overlay_thumbnail_b64(overlay_path, bbox=None, size=(224,224)):
    img = cv2.imread(str(overlay_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if bbox:
        x,y,w,h = bbox
        x2, y2 = min(W, x+w), min(H, y+h)
        if x2 > x and y2 > y:
            crop = img[y:y2, x:x2]
        else:
            crop = img[H//4:3*H//4, W//4:3*W//4]
    else:
        crop = img[H//4:3*H//4, W//4:3*W//4]
    thumb = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii")

def append_ndjson(path: Path, rec: dict):
    with open(path, "a", encoding="utf8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    pil_img = Image.open(tmp_path).convert("RGB")
    img_tensor = transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).item())

    # compute gradcam mask + stats
    mask, stats = compute_gradcam_with_stats(
        model, target_layer, img_tensor, target_class=pred_class,
        use_cuda=use_cuda, cam_instance=None, cam_device=None,
        threshold=cfg["gradcam"].get("threshold", 0.30),
        topk=3,
        retina_mask=None  # optional: if you have per-image retina mask path, load and pass it here
    )

    # save raw mask .npy
    mask_path = output_dir / f"{os.path.splitext(file.filename)[0]}_mask.npy"
    if mask is not None:
        save_raw_mask_npy(mask, str(mask_path))

    # build overlay path and save cleaned overlay (no retina_mask available here)
    base_img_np = denormalize_tensor(img_tensor[0].cpu())
    gradcam_path = output_dir / f"{os.path.splitext(file.filename)[0]}_gradcam.png"
    save_overlay(base_img_np, mask, str(gradcam_path), retina_mask=None, cleaned=False)

    # compute roi bbox (legacy) if requested
    roi_bbox = stats.get("bbox") if stats is not None and cfg["output"].get("save_roi_bbox", False) else None

    # create small overlay thumb b64
    thumb_b64 = overlay_thumbnail_b64(str(gradcam_path), bbox=stats.get("bbox") if stats else None)

    # QA flags
    qa_flags = compute_flags(probs, stats)

    # assemble record for NDJSON / KB
    record = make_record(
        image_id=file.filename,
        dr_stage=pred_class,
        confidence=list(probs),
        heatmap_path=str(gradcam_path),
        roi_bbox=roi_bbox,
        meta={"model": cfg.get("model", {}).get("arch", "unknown"), "preproc": cfg.get("preprocessing", {})},
        label_map=cfg.get("data", {}).get("label_map"),
        retina_mask_path=None,
        mask_path=str(mask_path) if mask is not None else None,
        gradcam_stats=stats,
        overlay_thumb_b64=thumb_b64,
        qa_flags=qa_flags
    )

    # append to NDJSON for downstream LLM / KB ingestion
    append_ndjson(ndjson_path, record)

    # clean up temp
    os.remove(tmp_path)

    # return a concise response (backwards-compatible + new fields)
    return {
        "filename": file.filename,
        "prediction": pred_class,
        "confidence": [float(p) for p in probs],
        "gradcam_path": str(gradcam_path),
        "mask_path": str(mask_path) if mask is not None else None,
        "gradcam_stats": stats,
        "overlay_thumb_b64": thumb_b64,
        "roi_bbox": roi_bbox,
        "qa_flags": qa_flags,
    }

# ------------------------------------------
# CLI MODE FOR LOCAL ONE-SHOT IMAGE INFERENCE
# ------------------------------------------
def infer_from_path(image_path):
    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).item())

    mask, stats = compute_gradcam_with_stats(
        model, target_layer, img_tensor, target_class=pred_class,
        use_cuda=use_cuda
    )

    mask_path = output_dir / (os.path.splitext(os.path.basename(image_path))[0] + "_mask.npy")
    if mask is not None:
        save_raw_mask_npy(mask, str(mask_path))

    base_img_np = denormalize_tensor(img_tensor[0].cpu())
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_gradcam.png"
    gradcam_path = output_dir / out_name
    save_overlay(base_img_np, mask, str(gradcam_path), cleaned=False)

    roi_bbox = stats.get("bbox") if stats is not None and cfg["output"].get("save_roi_bbox", False) else None
    thumb_b64 = overlay_thumbnail_b64(str(gradcam_path), bbox=stats.get("bbox") if stats else None)
    qa_flags = compute_flags(probs, stats)

    record = make_record(
        image_id=os.path.basename(image_path),
        dr_stage=pred_class,
        confidence=list(probs),
        heatmap_path=str(gradcam_path),
        roi_bbox=roi_bbox,
        meta={"model": cfg.get("model", {}).get("arch", "unknown"), "preproc": cfg.get("preprocessing", {})},
        label_map=cfg.get("data", {}).get("label_map"),
        retina_mask_path=None,
        mask_path=str(mask_path) if mask is not None else None,
        gradcam_stats=stats,
        overlay_thumb_b64=thumb_b64,
        qa_flags=qa_flags
    )

    append_ndjson(ndjson_path, record)

    return {
        "filename": os.path.basename(image_path),
        "prediction": pred_class,
        "confidence": [float(p) for p in probs],
        "gradcam_path": str(gradcam_path),
        "mask_path": str(mask_path) if mask is not None else None,
        "gradcam_stats": stats,
        "overlay_thumb_b64": thumb_b64,
        "roi_bbox": roi_bbox,
        "qa_flags": qa_flags
    }

def run_server(host="0.0.0.0", port=8000, reload=False):
    import uvicorn
    uvicorn.run("api_inference:app", host=host, port=int(port), reload=reload)

# -----------------------------
# MAIN (CLI + SERVER)
# -----------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="GradCAM API/CLI Inference")
    parser.add_argument("--mode", choices=["server", "cli"], default="cli",
                        help="server = run FastAPI, cli = run single-image inference")
    parser.add_argument("--image", help="Path to an image for CLI mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.mode == "server":
        print(f"Starting server at http://{args.host}:{args.port}")
        run_server(args.host, args.port, args.reload)
    else:
        if not args.image:
            raise SystemExit("ERROR: --image is required in cli mode.")
        print("Running one-shot inference for:", args.image)
        out = infer_from_path(args.image)
        print(json.dumps(out, indent=2))
        print("Saved GradCAM:", out["gradcam_path"])

#to call python api_inference.py --mode cli --image "path/to/image.jpg"

# or python api_inference.py --mode server --host 0.0.0.0 --port 8000
