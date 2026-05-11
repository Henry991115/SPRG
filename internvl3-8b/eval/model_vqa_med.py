"""Baseline InternVL VQA inference (no extra context). See HF model card for usage."""
import argparse
import math
import os
import sys

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import torchvision.transforms as T
import os
import sys
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed, logging

logging.set_verbosity_error()

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_path)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Resize and ImageNet-normalize images for InternVL."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Choose the closest (w,h) tile layout from candidate ratios."""
    best_ratio_diff = float('inf')
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
    """Dynamic tiling preprocess (InternVL-style multi-crop)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """Load an image path into a stacked tensor of normalized crops."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load tokenizer and AutoModel (trust_remote_code) for InternVL."""
    if os.path.exists(model_path) and os.path.basename(model_path).startswith('models--'):
        snapshots_dir = os.path.join(model_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir) 
                        if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                model_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"HF hub cache layout detected; using snapshot path: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    print("Loaded AutoTokenizer")
    
    print(f"Loading weights onto device: {device}")
    
    use_flash_attn = False
    try:
        import importlib
        importlib.import_module('flash_attn')
        use_flash_attn = True
        print("flash-attn available; using flash attention")
    except (ImportError, ModuleNotFoundError):
        print("flash-attn not installed; using standard attention")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        device_map="auto"
    )
    
    model = model.eval()
    
    print("Model ready")
    
    return model, tokenizer


def generate_answer(model, tokenizer, pixel_values, question, device="cuda", max_new_tokens=128):
    """Greedy decode via model.chat with an <image> prefix."""
    
    formatted_question = f'<image>\n{question}'
    
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    
    try:
        response = model.chat(
            tokenizer, 
            pixel_values, 
            formatted_question, 
            generation_config
        )
        return response.strip()
    except Exception as e:
        print(f"generate_answer failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="InternVL3-8B baseline VQA inference")
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL3_5-8B",
                       help="Model path or Hugging Face ID")
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to questions JSON file")
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Image folder path")
    parser.add_argument("--answers-file", type=str, required=True,
                       help="Output answers JSONL path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--num-chunks", type=int, default=1,
                       help="Number of shards for parallel inference")
    parser.add_argument("--chunk-idx", type=int, default=0,
                       help="Shard index (0-based)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    
    print("[DEBUG] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    print("[DEBUG] Model loaded")
    
    with open(args.question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    chunk_size = len(questions) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(questions)
    questions_chunk = questions[start_idx:end_idx]
    
    print(f"Processing questions [{start_idx}, {end_idx}), count={len(questions_chunk)}")
    
    results = []
    print(f"[DEBUG] Inference loop: {len(questions_chunk)} questions")
    for idx, question_data in enumerate(tqdm(questions_chunk, desc="infer")):
        if idx == 0:
            print("[DEBUG] First question...")
        
        qid = question_data.get('qid', question_data.get('question_id', 0))
        question = question_data.get('question', question_data.get('text', ''))
        image_file = question_data.get('img_name', question_data.get('image', question_data.get('image_id', '')))
        ground_truth = question_data.get('answer', question_data.get('ground_truth', ''))
        question_type = question_data.get('question_type', 'binary')
        
        if idx == 0:
            print(f"[DEBUG] QID: {qid}, Image: {image_file}, Question type: {question_type}")
        
        is_binary = question_type != "multi-choice"
        if question_type == "multi-choice":
            if len(question_data.get('choices', [])) >= 4:
                question += f" Please select from the following choices: {question_data['choices']}"
            else:
                if idx == 0:
                    print("[DEBUG] Skip multi-choice (not enough options)")
                continue
        else:
            question += " Please answer Yes or No."
        
        if "/" in image_file:
            image_path = os.path.join(args.image_folder, image_file)
        else:
            image_path = os.path.join(args.image_folder, image_file)
        
        if idx == 0:
            print(f"[DEBUG] Image path: {image_path}")
        
        try:
            if idx == 0:
                print("[DEBUG] Preprocessing image...")
            pixel_values = load_image(image_path, input_size=448, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16)
            if args.device == "cuda":
                pixel_values = pixel_values.cuda()
            if idx == 0:
                print(f"[DEBUG] pixel_values shape: {pixel_values.shape}")
            
            if idx == 0:
                print("[DEBUG] Generating answer...")
            answer = generate_answer(model, tokenizer, pixel_values, question, args.device)
            if idx == 0:
                print(f"[DEBUG] Answer preview: {answer[:50]}...")
            
            result = {
                "question_id": qid,
                "prompt": question,
                "model_answer": answer,
                "ground_truth": ground_truth,
                "image_id": image_file,
                "model_id": args.model_path,
                "metadata": {}
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error on question {qid}: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "question_id": qid,
                "prompt": question,
                "model_answer": f"Error: {str(e)}",
                "ground_truth": ground_truth,
                "image_id": image_file,
                "model_id": args.model_path,
                "metadata": {"error": str(e)}
            }
            results.append(result)
    
    with open(args.answers_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Done. Saved to: {args.answers_file}")


if __name__ == "__main__":
    main()

