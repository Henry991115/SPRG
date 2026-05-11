"""ROI patch masks and gating helpers for InternVL-style ViT features."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from PIL import Image
import numpy as np
import os


def find_mask_file(image_path: str) -> Optional[str]:
    """Locate a sidecar mask PNG next to the image (several naming conventions)."""
    if not image_path:
        return None
    
    image_dir = os.path.dirname(image_path)
    image_basename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_basename)[0]
    
    if not os.path.exists(image_dir):
        return None
    
    mask_candidate1 = os.path.join(image_dir, f"{image_name_without_ext}_mask.png")
    if os.path.exists(mask_candidate1):
        return mask_candidate1
    
    mask_candidate2 = os.path.join(image_dir, "mask.png")
    if os.path.exists(mask_candidate2):
        return mask_candidate2
    
    return None


def bbox_to_patch_mask(bbox: List[float], image_size: int, patch_size: int, num_patches_per_side: int, 
                       original_image_size: Optional[int] = None) -> torch.Tensor:
    """Map an xyxy box to a boolean patch grid."""
    x1, y1, x2, y2 = bbox
    
    if original_image_size is not None and original_image_size != image_size:
        scale = image_size / original_image_size
        x1 = x1 * scale
        y1 = y1 * scale
        x2 = x2 * scale
        y2 = y2 * scale
    
    x1_norm = max(0, min(1, x1 / image_size))
    y1_norm = max(0, min(1, y1 / image_size))
    x2_norm = max(0, min(1, x2 / image_size))
    y2_norm = max(0, min(1, y2 / image_size))
    
    patch_x1 = int(x1_norm * num_patches_per_side)
    patch_y1 = int(y1_norm * num_patches_per_side)
    patch_x2 = int(x2_norm * num_patches_per_side) + 1
    patch_y2 = int(y2_norm * num_patches_per_side) + 1
    
    mask = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.bool)
    mask[patch_y1:patch_y2, patch_x1:patch_x2] = True
    
    return mask


def load_mask_from_image(mask_path: str, image_size: int, patch_size: int, 
                         num_patches_per_side: int) -> torch.Tensor:
    """Rasterize a grayscale mask image into a patch-level mask."""
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)
    mask_array = np.array(mask_img)
    
    mask_binary = (mask_array > 128).astype(np.uint8)
    
    patch_mask = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.bool)
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            y_start = i * patch_size
            y_end = min((i + 1) * patch_size, image_size)
            x_start = j * patch_size
            x_end = min((j + 1) * patch_size, image_size)
            
            patch_region = mask_binary[y_start:y_end, x_start:x_end]
            if patch_region.sum() > 0:
                patch_mask[i, j] = True
    
    return patch_mask


def create_roi_mask_from_matched_detections(matched_detections: List[Dict], mask_path: Optional[str],
                                           image_size: int, patch_size: int, num_patches_per_side: int,
                                           original_image_size: Optional[int] = None) -> torch.Tensor:
    """Union of mask-file regions and detection bounding boxes."""
    combined_mask = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.bool)
    
    if mask_path and os.path.exists(mask_path):
        mask_from_file = load_mask_from_image(mask_path, image_size, patch_size, num_patches_per_side)
        combined_mask = combined_mask | mask_from_file
    
    for det in matched_detections:
        bbox = det.get("bbox", [])
        if len(bbox) >= 4:
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
            else:
                bbox_xyxy = bbox[:4]
            
            patch_mask = bbox_to_patch_mask(bbox_xyxy, image_size, patch_size, num_patches_per_side, original_image_size)
            combined_mask = combined_mask | patch_mask
    
    return combined_mask


def create_combined_roi_mask(roi_boxes: List[List[float]], image_size: int, patch_size: int,
                            num_patches_per_side: int, mask_path: Optional[str] = None,
                            original_image_size: Optional[int] = None) -> torch.Tensor:
    """Combine ROI boxes with an optional raster mask file."""
    combined_mask = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.bool)
    
    if mask_path and os.path.exists(mask_path):
        mask_from_file = load_mask_from_image(mask_path, image_size, patch_size, num_patches_per_side)
        combined_mask = combined_mask | mask_from_file
    
    for bbox in roi_boxes:
        if len(bbox) >= 4:
            if len(bbox) == 4:
                x, y, w, h = bbox
                bbox_xyxy = [x, y, x + w, y + h]
            else:
                bbox_xyxy = bbox[:4]
            
            patch_mask = bbox_to_patch_mask(bbox_xyxy, image_size, patch_size, num_patches_per_side, original_image_size)
            combined_mask = combined_mask | patch_mask
    
    return combined_mask


def apply_patch_masking(image_features: torch.Tensor, patch_mask: torch.Tensor,
                       gating: bool = True, roi_weight: float = 1.0, background_weight: float = 0.1) -> torch.Tensor:
    """Scale ViT patch tokens by ROI gating or hard masking."""
    if image_features.dim() == 3:
        # [batch, num_patches, hidden_dim]
        batch_size, num_patches, hidden_dim = image_features.shape
        
        device = image_features.device
        dtype = image_features.dtype
        mask_flat = patch_mask.flatten().to(device=device)
        mask_length = mask_flat.shape[0]
        
        if mask_length < num_patches:
            if gating:
                extra_weights = torch.full((num_patches - mask_length,), background_weight, 
                                          dtype=dtype, device=device)
            else:
                extra_weights = torch.ones((num_patches - mask_length,), 
                                          dtype=dtype, device=device)
            
            mask_flat = mask_flat[:min(mask_length, num_patches)]
            
            if gating:
                gate_weights = torch.where(mask_flat, 
                                         torch.tensor(roi_weight, dtype=dtype, device=device),
                                         torch.tensor(background_weight, dtype=dtype, device=device))
            else:
                gate_weights = torch.where(mask_flat,
                                         torch.tensor(1.0, dtype=dtype, device=device),
                                         torch.tensor(1.0, dtype=dtype, device=device))
            
            gate_weights = torch.cat([gate_weights, extra_weights])
        elif mask_length > num_patches:
            mask_flat = mask_flat[:num_patches]
            if gating:
                gate_weights = torch.where(mask_flat,
                                         torch.tensor(roi_weight, dtype=dtype, device=device),
                                         torch.tensor(background_weight, dtype=dtype, device=device))
            else:
                gate_weights = torch.where(mask_flat,
                                         torch.tensor(1.0, dtype=dtype, device=device),
                                         torch.tensor(0.0, dtype=dtype, device=device))
        else:
            if gating:
                gate_weights = torch.where(mask_flat,
                                         torch.tensor(roi_weight, dtype=dtype, device=device),
                                         torch.tensor(background_weight, dtype=dtype, device=device))
            else:
                gate_weights = torch.where(mask_flat,
                                         torch.tensor(1.0, dtype=dtype, device=device),
                                         torch.tensor(0.0, dtype=dtype, device=device))
        
        gate_weights = gate_weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_patches, hidden_dim)
        
        if not hasattr(apply_patch_masking, '_debug_count'):
            apply_patch_masking._debug_count = 0
        apply_patch_masking._debug_count += 1
        if apply_patch_masking._debug_count == 1:
            roi_patches_count = mask_flat.sum().item() if mask_length <= num_patches else 0
            background_patches_count = (mask_length - roi_patches_count) if mask_length <= num_patches else 0
            extra_patches_count = max(0, num_patches - mask_length)
            print(f"[DEBUG apply_patch_masking] Patch count: ROI={roi_patches_count}, "
                  f"Background={background_patches_count}, Extra={extra_patches_count}, Total={num_patches}")
            print(f"[DEBUG apply_patch_masking] Weights: roi_weight={roi_weight}, "
                  f"background_weight={background_weight}")
            print(f"[DEBUG apply_patch_masking] Gate weights stats: min={gate_weights.min().item():.4f}, "
                  f"max={gate_weights.max().item():.4f}, mean={gate_weights.mean().item():.4f}")
        
        image_features = image_features * gate_weights
    
    return image_features


def create_roi_token_mask(patch_mask: torch.Tensor, image_token_start_idx: int, num_patches: int) -> torch.Tensor:
    """Flatten patch_mask when compatible with num_patches."""
    mask_flat = patch_mask.flatten()
    
    if mask_flat.shape[0] != num_patches:
        num_patches_per_side = int(num_patches ** 0.5)
        if mask_flat.shape[0] == num_patches_per_side * num_patches_per_side:
            pass
        else:
            print(f"[Warning] Patch mask size {mask_flat.shape[0]} doesn't match num_patches {num_patches}")
            return torch.zeros(num_patches, dtype=torch.bool)
    
    return mask_flat


def apply_llm_layer_reinjection(hidden_states: torch.Tensor, roi_token_mask: torch.Tensor,
                                roi_weight: float = 1.0, background_weight: float = 0.1,
                                gating: bool = True, image_token_start: int = None,
                                image_token_end: int = None) -> torch.Tensor:
    """Per-token scaling of hidden states for ROI tokens (optional image span)."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    if roi_token_mask.shape[0] != seq_len:
        print(f"[Warning] ROI token mask size {roi_token_mask.shape[0]} doesn't match seq_len {seq_len}")
        return hidden_states
    
    roi_token_mask = roi_token_mask.to(hidden_states.device)
    original_dtype = hidden_states.dtype
    original_device = hidden_states.device
    
    gate_weights = torch.ones(seq_len, dtype=original_dtype, device=original_device)
    
    if image_token_start is not None and image_token_end is not None:
        image_region_mask = torch.zeros(seq_len, dtype=torch.bool, device=original_device)
        if image_token_start < seq_len:
            end_idx = min(image_token_end, seq_len)
            image_region_mask[image_token_start:end_idx] = True
            
            if gating:
                for i in range(image_token_start, end_idx):
                    if i < len(roi_token_mask):
                        if roi_token_mask[i]:
                            gate_weights[i] = roi_weight
                        else:
                            gate_weights[i] = background_weight
            else:
                for i in range(image_token_start, end_idx):
                    if i < len(roi_token_mask) and not roi_token_mask[i]:
                        gate_weights[i] = 0.0
    else:
        if gating:
            gate_weights = torch.where(roi_token_mask, roi_weight, background_weight)
        else:
            gate_weights = torch.where(roi_token_mask, 1.0, 0.0)
    
    gate_weights = gate_weights.unsqueeze(-1).expand(-1, hidden_size)
    hidden_states = hidden_states * gate_weights
    
    return hidden_states

