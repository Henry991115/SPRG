"""InternVL chat templates, forwards, and PIL counterfactual helpers (aligned with official chat)."""

from __future__ import annotations

import importlib.util
import inspect
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_conversation_module(model: torch.nn.Module):
    mfile = inspect.getfile(model.__class__)
    parent = os.path.dirname(mfile)
    path = os.path.join(parent, "conversation.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"conversation.py not found: {path}")
    name = "_internvl_conversation_runtime"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load conversation")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_pil(pil_image: Image.Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """Like eval/model_vqa_med.load_image but accepts a PIL image."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(im) for im in images]
    return torch.stack(pixel_values)


def get_eos_token_id(model: torch.nn.Module, tokenizer) -> int:
    mod = _load_conversation_module(model)
    template = mod.get_conv_template(model.template)
    template.system_message = getattr(model, "system_message", template.system_message)
    tid = tokenizer.convert_tokens_to_ids(template.sep.strip())
    return int(tid)


def encode_user_body_to_input_ids(
    model: torch.nn.Module,
    tokenizer,
    user_body: str,
    num_patches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize user_body without raw <image>; returns input_ids, attention_mask, eos_token_id."""
    q = user_body.replace("<image>", "").strip()
    question = "<image>\n" + q
    query = build_chat_query_from_parts(model, tokenizer, question, num_patches)
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    eos_id = get_eos_token_id(model, tokenizer)
    return input_ids, attention_mask, eos_id


def build_chat_query_from_parts(model: torch.nn.Module, tokenizer, question: str, num_patches: int) -> str:
    _ = tokenizer
    if "<image>" not in question:
        question = "<image>\n" + question
    mod = _load_conversation_module(model)
    template = mod.get_conv_template(model.template)
    template.system_message = getattr(model, "system_message", template.system_message)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    num_tokens = int(model.num_image_token) * num_patches
    image_tokens = "<img>" + "<IMG_CONTEXT>" * num_tokens + "</img>"
    query = query.replace("<image>", image_tokens, 1)
    return query


def internvl_forward_logits(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
    with torch.inference_mode():
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=flags,
            use_cache=False,
        )
    return out.logits[0, -1, :].float()


def internvl_forward_vcd_prefill(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Prefill full prompt with KV cache; return logits at last position and past_key_values."""
    flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
    with torch.inference_mode():
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=flags,
            use_cache=True,
        )
    return out.logits[0, -1, :].float(), out.past_key_values


def internvl_forward_vcd_decode_step(
    model: torch.nn.Module,
    input_ids_new: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
):
    """Decode one new token; past already contains image tokens (skip ViT)."""
    with torch.inference_mode():
        out = model(
            pixel_values=None,
            input_ids=input_ids_new,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return out.logits[0, -1, :].float(), out.past_key_values


def corrupt_pil_bbox(
    pil_image: Image.Image,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    method: str = "gaussian",
    gaussian_std: float = 0.2,
) -> Image.Image:
    """Replace pixels inside each xyxy box (zero / mean gray / Gaussian noise)."""
    arr = np.array(pil_image.convert("RGB"))
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(arr.shape[1], x2), min(arr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if method == "zero":
            arr[y1:y2, x1:x2] = 0
        elif method == "mean":
            arr[y1:y2, x1:x2] = 128
        else:
            noise = np.random.randn(y2 - y1, x2 - x1, 3) * float(gaussian_std) * 255.0
            arr[y1:y2, x1:x2] = np.clip(
                arr[y1:y2, x1:x2].astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
    return Image.fromarray(arr)


def corrupt_pil_mask_in_bbox(
    pil_image: Image.Image,
    mask_path: str,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    method: str = "gaussian",
    gaussian_std: float = 0.2,
) -> Image.Image:
    """Corrupt only mask-positive pixels that fall inside each bbox."""
    m = Image.open(mask_path).convert("L")
    m = m.resize(pil_image.size, Image.NEAREST)
    mask_np = np.array(m)
    arr = np.array(pil_image.convert("RGB"))
    h, w = mask_np.shape
    for box in boxes_xyxy:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        sub_m = mask_np[y1:y2, x1:x2] > 0
        if not np.any(sub_m):
            continue
        ys, xs = np.where(sub_m)
        for yy, xx in zip(ys, xs):
            gy, gx = y1 + yy, x1 + xx
            if method == "zero":
                arr[gy, gx] = 0
            elif method == "mean":
                arr[gy, gx] = 128
            else:
                arr[gy, gx] = np.clip(
                    arr[gy, gx].astype(np.float32)
                    + np.random.randn(3).astype(np.float32) * float(gaussian_std) * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
    return Image.fromarray(arr)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: Optional[float],
) -> int:
    if temperature <= 0 and (top_p is None or top_p >= 1.0):
        return int(torch.argmax(logits).item())
    logits_scaled = logits / max(float(temperature), 1e-6)
    if top_p is not None and top_p < 1.0:
        try:
            from transformers.generation.logits_process import TopPLogitsWarper

            logits_scaled = TopPLogitsWarper(top_p)(None, logits_scaled.unsqueeze(0))[0]
        except Exception:
            pass
    probs = torch.softmax(logits_scaled, dim=-1)
    if temperature <= 0:
        return int(torch.argmax(probs).item())
    return int(torch.multinomial(probs, num_samples=1).item())
