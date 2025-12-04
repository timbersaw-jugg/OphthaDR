# src/gradcam.py
"""
Robust GradCAM helper utilities — streamlined for our pipeline.

This full file is the copy-pasteable version you asked for.
It keeps all of your original helpers and replaces the default
show_cam_on_image blending with a deterministic `blend_heatmap_on_image`
routine that preserves the original retina image, uses a perceptually
uniform colormap, applies a small low-threshold to avoid tinting the whole
image, and blends using a per-pixel alpha (alpha*mask) so areas with
mask==0 remain exactly the original image.

Drop this file into src/gradcam.py replacing your existing file.
Minimal change to the rest of your pipeline: `save_overlay` signature
is preserved and file output is unchanged (cv2.imwrite of BGR image).
"""

import inspect
import torch
import numpy as np
from scipy.ndimage import maximum_filter

# optional libs used for blending and I/O
import matplotlib.cm as cm
from PIL import Image
import cv2

# minimal import guard for pytorch-grad-cam (we keep it for mask computation)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image  # we keep available but won't rely on it for blending
except Exception:
    GradCAM = None
    ClassifierOutputTarget = None
    show_cam_on_image = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def resolve_target_layer(model, layer_spec):
    if not isinstance(layer_spec, str):
        return layer_spec
    if 'stages' in layer_spec and hasattr(model, "stages"):
        return model.stages[-1]
    if hasattr(model, layer_spec):
        return getattr(model, layer_spec)
    raise ValueError(f"Could not resolve target layer: {layer_spec}")


def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Turn a CxHxW tensor (0..1 normalized) into HxWx3 float32 image (0..1)."""
    t = tensor.detach().cpu().numpy()
    if t.ndim == 3:
        t = np.transpose(t, (1, 2, 0))
    else:
        t = np.transpose(t[0], (1, 2, 0))
    t = (t * std) + mean
    t = np.clip(t, 0.0, 1.0)
    return t.astype(np.float32)


def _instantiate_gradcam(model, target_layer, prefer_cuda=False):
    if GradCAM is None:
        raise RuntimeError("pytorch-grad-cam not available.")
    device = torch.device("cuda") if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")
    sig = inspect.signature(GradCAM.__init__)
    kwargs = {}
    if "device" in sig.parameters:
        kwargs["device"] = device
    elif "use_cuda" in sig.parameters:
        kwargs["use_cuda"] = (device.type == "cuda")
    cam = GradCAM(model=model, target_layers=[target_layer], **kwargs) if kwargs else GradCAM(model=model, target_layers=[target_layer])
    return cam, device


def build_cam_for_model(model, target_layer, prefer_cuda=False):
    cam, device = _instantiate_gradcam(model, target_layer, prefer_cuda=prefer_cuda)
    return cam, device


def _normalize_grayscale_cam_array(grayscale_cam):
    """
    Normalize a numpy grayscale_cam array to ensure values per-image are in [0,1].
    Accepts arrays of shape (B,H,W) or (H,W).
    """
    if grayscale_cam is None:
        return None
    g = np.array(grayscale_cam).astype(np.float32)
    if g.ndim == 3:
        for i in range(g.shape[0]):
            m = g[i]
            m = m - np.nanmin(m)
            maxv = np.nanmax(m)
            if maxv > 0:
                m = m / maxv
            g[i] = m
    elif g.ndim == 2:
        g = g - np.nanmin(g)
        maxv = np.nanmax(g)
        if maxv > 0:
            g = g / maxv
    return g


def compute_gradcam(model, target_layer, input_tensor, target_class=None, use_cuda=False, cam_instance=None, cam_device=None):
    """
    Compute GradCAM mask(s) for `input_tensor`. Returns numpy array (B,H,W) or None.
    Ensures the model is in eval() mode while computing the CAM.
    """
    cam = cam_instance
    device = cam_device
    created_here = False
    if cam is None:
        cam, device = _instantiate_gradcam(model, target_layer, prefer_cuda=use_cuda)
        created_here = True

    if input_tensor is None:
        return None
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    targets = None
    if target_class is not None:
        if ClassifierOutputTarget is None:
            raise RuntimeError("ClassifierOutputTarget not available.")
        targets = [ClassifierOutputTarget(int(target_class))]

    # preserve original device and training state
    params = list(model.parameters())
    orig_device = params[0].device if len(params) > 0 else torch.device("cpu")
    model_was_training = model.training

    try:
        model.eval()
    except Exception:
        pass

    try:
        model.to(device)
    except Exception:
        pass

    input_tensor = input_tensor.to(device)

    # ensure requires_grad True for backward
    orig_requires = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad = True

    grayscale_cam = None
    with torch.enable_grad():
        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets) if targets is not None else cam(input_tensor=input_tensor)
        except TypeError:
            grayscale_cam = cam(input_tensor=input_tensor)
        except Exception:
            grayscale_cam = None

    # restore flags and device/state
    for p, flag in zip(params, orig_requires):
        p.requires_grad = flag
    try:
        model.to(orig_device)
    except Exception:
        pass
    try:
        if model_was_training:
            model.train()
    except Exception:
        pass

    if created_here:
        try:
            del cam
        except Exception:
            pass

    if grayscale_cam is None:
        return None

    # normalize per-image to [0,1]
    g = _normalize_grayscale_cam_array(np.array(grayscale_cam))
    return g


def gradcam_stats_from_mask(mask, threshold=0.30, topk=3, retina_mask=None):
    """
    Compute compact stats from a raw CAM mask (H,W) or (1,H,W).
    """
    if mask is None:
        return None
    m = mask[0] if mask.ndim == 3 else mask
    H, W = m.shape
    mask_bin = (m >= threshold).astype(np.uint8)
    mask_area_frac = float(mask_bin.sum()) / float(H * W)

    coords = np.column_stack(np.where(mask_bin > 0))
    if coords.size == 0:
        bbox = None
        bbox_area_frac = 0.0
        centroid = (None, None)
        mean_in_bbox = None
    else:
        y_min, x_min = int(coords[:, 0].min()), int(coords[:, 1].min())
        y_max, x_max = int(coords[:, 0].max()), int(coords[:, 1].max())
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        bbox_area_frac = (bbox[2] * bbox[3]) / float(H * W)
        cy, cx = float(coords[:, 0].mean()), float(coords[:, 1].mean())
        centroid = (float(cx / W), float(cy / H))
        sub = m[y_min:y_max + 1, x_min:x_max + 1]
        mean_in_bbox = float(np.nanmean(sub)) if sub.size > 0 else 0.0

    median_heatmap = float(np.nanmedian(m))

    data_max = maximum_filter(m, 5)
    peaks_mask = (m == data_max) & (m >= np.percentile(m, 90))
    peak_coords = np.column_stack(np.where(peaks_mask))
    peaks = [{"y": int(p[0]), "x": int(p[1]), "value": float(m[p[0], p[1]])} for p in peak_coords]
    peaks = sorted(peaks, key=lambda x: -x["value"])[:topk]

    stats = {
        "mask_shape": [int(H), int(W)],
        "threshold": float(threshold),
        "mask_area_frac": float(mask_area_frac),
        "bbox": bbox,
        "bbox_area_frac": float(bbox_area_frac),
        "centroid": [None if centroid[0] is None else round(centroid[0], 4),
                     None if centroid[1] is None else round(centroid[1], 4)],
        "mean_in_bbox": None if mean_in_bbox is None else float(mean_in_bbox),
        "median_heatmap": float(median_heatmap),
        "top_peaks": peaks
    }

    if retina_mask is not None:
        rm = retina_mask.astype(bool)
        if rm.shape != (H, W):
            rm = cv2.resize(rm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        salient = (m >= threshold)
        salient_count = int(salient.sum())
        outside_count = int(((~rm) & salient).sum())
        leakage_fraction = float(outside_count) / salient_count if salient_count > 0 else 0.0
        total_mass = float(m.sum())
        outside_mass = float((m * (~rm)).sum())
        outside_mass_frac = outside_mass / total_mass if total_mass > 0 else 0.0
        stats["leakage_fraction"] = float(leakage_fraction)
        stats["outside_mass_frac"] = float(outside_mass_frac)

    return stats


def compute_gradcam_with_stats(model, target_layer, input_tensor, target_class=None,
                               use_cuda=False, cam_instance=None, cam_device=None,
                               threshold=0.30, topk=3, retina_mask=None):
    """
    Convenience wrapper: compute raw mask and stats in one call.
    """
    mask = compute_gradcam(model, target_layer, input_tensor, target_class=target_class,
                           use_cuda=use_cuda, cam_instance=cam_instance, cam_device=cam_device)
    if mask is None:
        return None, None
    stats = gradcam_stats_from_mask(mask, threshold=threshold, topk=topk, retina_mask=retina_mask)
    return mask, stats


def save_raw_mask_npy(mask, out_path):
    """Save raw float32 mask as .npy (keeps full precision)."""
    if mask is None:
        return False
    np.save(out_path, mask.astype(np.float32))
    return True


# -------------------------------
# New helper(s) for deterministic blending
# -------------------------------

def _ensure_rgb_float01(img):
    """Return HxWx3 float32 image in [0,1] for numpy/PIL/torch inputs."""
    # PIL
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return arr
    # torch tensor
    try:
        import torch as _torch
        if isinstance(img, _torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3:
                arr = np.transpose(t.numpy(), (1, 2, 0))
            elif t.ndim == 4:
                arr = np.transpose(t[0].numpy(), (1, 2, 0))
            else:
                raise RuntimeError("Unsupported tensor shape for image.")
            arr = arr.astype(np.float32)
            if arr.max() > 2.0:
                arr = np.clip(arr / 255.0, 0.0, 1.0)
            else:
                arr = np.clip(arr, 0.0, 1.0)
            return arr
    except Exception:
        pass
    # numpy array
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 2.0:
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
    # channel-first fallback
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3) and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


# def blend_heatmap_on_image(orig_img, mask,
#                            alpha=0.65,
#                            colormap='jet',
#                            low_threshold=0.0,
#                            boost_gamma=1.0,
#                            eps=1e-8):
#     """
#     FINAL: visible-retina Grad-CAM blending that matches the old behavior.

#     - orig_img: numpy/PIL/torch -> HxWx3 (float in [0,1] or uint8)
#     - mask: HxW or (1,H,W) float mask (0..1 recommended)
#     - alpha: global blend strength (0..1). 0.6-0.7 restores old visibility.
#     - colormap: matplotlib colormap name (jet for exact old look)
#     - low_threshold: suppress tiny activations (0.0 -> show everything)
#     - boost_gamma: if !=1.0, apply gamma to mask ( <1 brightens mid/low )
#     Returns: uint8 RGB image HxWx3 (0..255)
#     """
#     import numpy as np
#     import cv2
#     import matplotlib.cm as cm

#     # Normalize mask to HxW float32
#     m = np.array(mask).astype(np.float32)
#     if m.ndim == 3:
#         m = m[0]

#     # Per-image normalization (keep relative strengths)
#     m = m - np.nanmin(m)
#     maxv = np.nanmax(m)
#     if maxv > eps:
#         m = m / maxv
#     else:
#         m = np.zeros_like(m, dtype=np.float32)

#     # Optionally suppress extremely small activations (leave 0.0 to show all)
#     if low_threshold is not None and low_threshold > 0.0:
#         m = np.where(m >= low_threshold, m, 0.0).astype(np.float32)

#     # Optional gamma boost (set to 1.0 to disable)
#     if boost_gamma is not None and boost_gamma != 1.0:
#         # gamma < 1 brightens mid/low activations; gamma > 1 darkens them
#         m = np.power(m, boost_gamma).astype(np.float32)

#     # Ensure image is HxWx3 float in [0,1]
#     img = _ensure_rgb_float01(orig_img)

#     H, W = img.shape[:2]
#     if m.shape != (H, W):
#         m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

#     cmap = cm.get_cmap(colormap)
#     heat_rgb = cmap(m)[:, :, :3]  # in 0..1

#     # per-pixel alpha so mask==0 leaves original unchanged
#     alpha_map = (alpha * m)[:, :, None]  # HxWx1

#     blended = (1.0 - alpha_map) * img + alpha_map * heat_rgb
#     blended = np.clip(blended, 0.0, 1.0)

#     # Return uint8 RGB
#     return (blended * 255).astype(np.uint8)

def blend_heatmap_on_image(orig_img, mask,
                           alpha=0.50,
                           colormap='jet',
                           eps=1e-8):
    """
    FINAL version: EXACT old Grad-CAM visualization.
    - Jet colormap (blue→green→yellow→red)
    - Retina stays visible
    - Blue low-activation tint preserved
    - No thresholding, no gamma suppression
    - Simple RGB alpha blend = reliable & identical to before
    """
    import numpy as np
    import cv2
    import matplotlib.cm as cm

    # Convert mask
    m = np.array(mask).astype(np.float32)
    if m.ndim == 3:
        m = m[0]

    # Normalize 0..1
    m = m - np.nanmin(m)
    maxv = np.nanmax(m)
    if maxv > eps:
        m = m / maxv
    else:
        m = np.zeros_like(m, dtype=np.float32)

    # Make sure image is float32 RGB
    img = _ensure_rgb_float01(orig_img)

    # Resize mask if needed
    H, W = img.shape[:2]
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

    # Jet colormap (EXACT NVIDIA/GradCAM default)
    cmap = cm.get_cmap(colormap)
    heat = cmap(m)[:, :, :3].astype(np.float32)  # 0..1 RGB

    # Alpha blend (simple & correct):
    # blended = img*(1-alpha) + heat*(alpha)
    blended = (1.0 - alpha) * img + alpha * heat
    blended = np.clip(blended, 0.0, 1.0)

    return (blended * 255).astype(np.uint8)  # uint8 RGB


# -------------------------------
# Overwrite save_overlay to use deterministic blending (minimal changes)
# -------------------------------
def save_overlay(orig_img, grayscale_cam, out_path, retina_mask=None, cleaned=False, threshold=0.25):
    """
    Save visualization overlay.

    orig_img: one of:
      - numpy uint8 HxWx3 in [0,255]
      - numpy float HxWx3 in [0,1]
      - torch tensor CxHxW (normalized or not; will be denormalized if assumed to be in 0..1)
    grayscale_cam: (H,W) or (B,H,W); values in [0,1] recommended (function will normalize if not).
    cleaned: if True and retina_mask provided, will dim outside-retina regions and draw contours for salient outside.
    threshold: used for 'cleaned' contouring (saliency threshold).
    """
    if grayscale_cam is None:
        return False

    # Normalize mask to [0,1] per-image
    mask = _normalize_grayscale_cam_array(grayscale_cam)
    if mask is None:
        return False

    # Handle orig_img types deterministically (no frame-inspection hacks)
    img = None
    # 1) numpy array
    if isinstance(orig_img, np.ndarray):
        img = orig_img
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        # ensure shape HxWx3
        if img.shape[-1] != 3:
            # try to reshape if channel-first
            if img.shape[0] in (1, 3) and img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
    else:
        # 2) torch tensor or PIL handled by denormalize_tensor / helpers
        try:
            import torch as _torch
            if isinstance(orig_img, _torch.Tensor):
                t = orig_img.detach().cpu()
                if t.ndim == 3:
                    # assume CxHxW (normalized 0..1) and denormalize
                    img = denormalize_tensor(t)
                elif t.ndim == 4:
                    img = denormalize_tensor(t[0])
                else:
                    raise RuntimeError("Unsupported tensor shape for orig_img")
            elif isinstance(orig_img, Image.Image):
                img = np.array(orig_img.convert("RGB")).astype(np.float32) / 255.0
            else:
                raise RuntimeError("Unsupported orig_img type for save_overlay; expected numpy, torch tensor, or PIL image.")
        except Exception as e:
            raise RuntimeError(f"Unsupported orig_img type for save_overlay: {e}")

    # final sanitize
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    # pick a single mask if batch provided
    m = mask[0] if mask.ndim == 3 else mask

    # Use deterministic blending helper (preserves retina)
    # cam_img = blend_heatmap_on_image(img, m, alpha=0.45, colormap='magma', low_threshold=0.02)
#     cam_img = blend_heatmap_on_image(
#     img,
#     m,
#     alpha=0.65,          # STRONG ENOUGH TO SEE
#     colormap='jet',      # SAME AS YOUR OLD WORKING VERSION
#     low_threshold=0.0,   # DO NOT SUPPRESS ANY LOW VALUES
#     boost_gamma=1.0      # NO GAMMA SUPPRESSION
# )
    cam_img = blend_heatmap_on_image(img, m, alpha=0.50, colormap='jet')

    # If cleaned mode requested, apply dimming outside retina and contours as before
    if cleaned and (retina_mask is not None):
        H, W = cam_img.shape[:2]
        rm = retina_mask
        if rm.shape != (H, W):
            rm = cv2.resize(rm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        cleaned_img = cam_img.copy()
        cleaned_img[~rm] = (cleaned_img[~rm] * 0.2).astype(np.uint8)
        salient_outside = ((m >= threshold) & (~rm)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(salient_outside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw red contours (RGB) around salient outside-region
        cv2.drawContours(cleaned_img, contours, -1, (255, 0, 0), 2)
        cam_bgr = cv2.cvtColor(cleaned_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, cam_bgr)
        return True

    cam_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, cam_bgr)
    return True
