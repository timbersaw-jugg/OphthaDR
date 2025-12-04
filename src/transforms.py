# src/transforms.py (FIXED)
"""Image transforms for training and validation."""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
import torch


class CLAHEGreenChannel:
    """Apply CLAHE to green channel for fundus images."""
    def __call__(self, img):
        img_np = np.array(img)
        g = img_np[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g)
        img_np[:, :, 1] = g_clahe
        return Image.fromarray(img_np)


class BenGrahamCrop:
    """Ben Graham's preprocessing: circular crop and resize."""
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, img):
        np_img = np.array(img)
        h, w = np_img.shape[:2]
        min_dim = min(h, w)
        center = (w // 2, h // 2)
        
        # Create circular mask
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, center, min_dim // 2, 1, -1)
        np_img[mask == 0] = 0
        
        # Crop to square
        x0 = center[0] - min_dim // 2
        y0 = center[1] - min_dim // 2
        crop = np_img[y0:y0 + min_dim, x0:x0 + min_dim]
        
        # Resize
        pil_crop = Image.fromarray(crop).resize((self.output_size, self.output_size))
        return pil_crop


class TransformWrapper:
    """Wraps PIL pre-transforms and Albumentations pipeline."""
    
    def __init__(self, pre_transforms, albumentations_compose, to_tensor=True):
        self.pre_transforms = pre_transforms or []
        self.alb = albumentations_compose
        self.to_tensor = to_tensor

    def __call__(self, img):
        # Apply PIL-based pre-transforms
        for t in self.pre_transforms:
            img = t(img)
        
        # Convert to numpy (uint8, 0-255)
        img_np = np.array(img)
        
        # Apply albumentations
        if self.alb is not None:
            img_np = self.alb(image=img_np)['image']
        
        if self.to_tensor:
            # Handle both normalized (float) and non-normalized (uint8)
            if img_np.dtype == np.uint8:
                tensor_img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            else:
                tensor_img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
            return tensor_img
        
        return img_np


def get_train_transforms(config):
    """Build training transforms from config."""
    aug = config.get('augmentation', {})
    pre = config.get('preprocessing', {})
    pre_transforms = []

    # 1. Setup Pre-transforms (PIL based)
    if aug.get('clahe_green', False):
        pre_transforms.append(CLAHEGreenChannel())
    
    # Check if Ben Graham Crop is enabled
    use_ben_crop = aug.get('ben_graham_crop', False)
    if use_ben_crop:
        # Ben Graham Crop handles resizing internally
        pre_transforms.append(BenGrahamCrop(pre.get('resize', 224)))

    a_trfs = []
    
    # 2. FIXED: Only add A.Resize if Ben Graham Crop is NOT used
    # If we already cropped/resized in step 1, skipping this prevents double interpolation
    if not use_ben_crop:
        a_trfs.append(A.Resize(pre.get('resize', 224), pre.get('resize', 224)))
    
    # Augmentations
    if aug.get('rotate', 0):
        a_trfs.append(A.Rotate(limit=aug['rotate'], p=0.5, border_mode=cv2.BORDER_CONSTANT))
    if aug.get('flip_horizontal', False):
        a_trfs.append(A.HorizontalFlip(p=0.5))
    if aug.get('flip_vertical', False):
        a_trfs.append(A.VerticalFlip(p=0.5))
    if aug.get('color_jitter', False):
        a_trfs.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5))
    if aug.get('cutout', False):
        a_trfs.append(A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            p=0.5
        ))
    
    # Normalize LAST
    if pre.get('normalize', False):
        a_trfs.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    alb_compose = A.Compose(a_trfs) if a_trfs else None
    return TransformWrapper(pre_transforms, alb_compose, to_tensor=True)


def get_val_transforms(config):
    """Build validation transforms from config."""
    aug = config.get('augmentation', {})
    pre = config.get('preprocessing', {})
    pre_transforms = []
    
    if aug.get('clahe_green', False):
        pre_transforms.append(CLAHEGreenChannel())

    use_ben_crop = aug.get('ben_graham_crop', False)
    if use_ben_crop:
        pre_transforms.append(BenGrahamCrop(pre.get('resize', 224)))

    a_trfs = []

    # FIXED: Only resize if Ben Graham didn't already do it
    if not use_ben_crop:
        a_trfs.append(A.Resize(pre.get('resize', 224), pre.get('resize', 224)))
    
    if pre.get('normalize', False):
        a_trfs.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    alb_compose = A.Compose(a_trfs) if a_trfs else None
    return TransformWrapper(pre_transforms, alb_compose, to_tensor=True)