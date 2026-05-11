"""Medical VQA with optional vision-tower ROI hooks (patch gating/masking)."""
import argparse
import torch
import os
import sys
import json
import re
from tqdm import tqdm
import shortuuid
from typing import List, Optional, Dict
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed, logging

logging.set_verbosity_error()

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_path)

eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, eval_dir)

from roi_utils import (
    find_mask_file,
    create_roi_mask_from_matched_detections,
    create_combined_roi_mask,
    apply_patch_masking
)

from model_vqa_med import load_model_and_tokenizer, load_image, generate_answer

MEDICAL_TERM_MAPPING = {
    "pneumothorax": ["lung", "pleura", "pneumothorax", "thoracic", "chest", "air"],
    "lung": ["lung", "pneumothorax", "pleura", "thoracic", "chest", "pulmonary", "respiratory"],
    "pleura": ["pleura", "lung", "pneumothorax", "thoracic"],
    "liver": ["liver", "hepatic", "hepat"],
    "heart": ["heart", "cardiac", "cardiovascular", "myocardial"],
    "cardiac": ["heart", "cardiac", "cardiovascular", "myocardial"],
    "cancer": ["cancer", "tumor", "tumour", "mass", "lesion", "malignancy"],
    "tumor": ["cancer", "tumor", "tumour", "mass", "lesion", "malignancy"],
    "fracture": ["fracture", "bone", "broken", "break"],
    "infection": ["infection", "inflammatory", "inflammation", "pneumonia"],
    "pneumonia": ["lung", "pneumonia", "infection", "inflammatory"],
    "effusion": ["effusion", "fluid", "pleural", "pericardial"],
    "kidney": ["kidney", "renal"],
    "brain": ["brain", "cerebral", "intracranial"],
}


def extract_keywords(text: str) -> set:
    """Alphanumeric tokens from text (length > 2)."""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return set(token for token in tokens if len(token) > 2)


def extract_medical_entities(question: str) -> set:
    """Keywords plus MEDICAL_TERM_MAPPING expansions."""
    question_lower = question.lower()
    keywords = extract_keywords(question)
    related_terms = set(keywords)
    
    for keyword in keywords:
        if keyword in MEDICAL_TERM_MAPPING:
            related_terms.update(MEDICAL_TERM_MAPPING[keyword])
    
    for term, related in MEDICAL_TERM_MAPPING.items():
        if term in question_lower:
            related_terms.update(related)
    
    return related_terms


def label_matches_question(label: str, question_entities: set, question_text: str) -> bool:
    """Whether a detection label matches question entities."""
    if not question_entities:
        return True
    
    label_lower = label.lower()
    if label_lower in question_text.lower():
        return True
    
    label_keywords = extract_keywords(label_lower)
    if label_keywords.intersection(question_entities):
        return True
    
    for entity in question_entities:
        if entity in label_lower:
            return True
    
    return False


def build_image_path(image_folder: str, image_file: str) -> str:
    """Resolve image path for flat or nested img_name layouts."""
    if "/" in image_file:
        return os.path.join(image_folder, image_file)
    else:
        img_dir = os.path.splitext(image_file)[0]
        possible_names = [image_file, f"{img_dir}.jpg", "source.jpg"]
        for name in possible_names:
            candidate = os.path.join(image_folder, img_dir, name)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(image_folder, img_dir, image_file)


def load_matched_detections(image_folder: str, img_name: str, question_text: str = "") -> List[Dict]:
    """Parse detection.json; filter boxes by question entities when question_text is set."""
    try:
        if "/" in img_name:
            parts = img_name.split("/")
            img_dir = "/".join(parts[:-1])
        else:
            img_dir = os.path.splitext(img_name)[0]
        
        detection_path = os.path.join(image_folder, img_dir, "detection.json")
        
        if not os.path.exists(detection_path):
            return []
        
        with open(detection_path, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)
        
        question = question_text or ""
        matched_detections = []
        
        if not question:
            for item in detection_data:
                if isinstance(item, dict):
                    for label, bbox in item.items():
                        if len(bbox) >= 4:
                            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                            matched_detections.append({
                                "label": label,
                                "bbox": [x1, y1, x2, y2],
                                "bbox_orig": [x, y, w, h]
                            })
        else:
            question_entities = extract_medical_entities(question)
            
            if isinstance(detection_data, list):
                for item in detection_data:
                    if isinstance(item, dict):
                        for label, bbox in item.items():
                            if len(bbox) >= 4:
                                if label_matches_question(label, question_entities, question):
                                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                                    x1, y1 = int(x), int(y)
                                    x2, y2 = int(x + w), int(y + h)
                                    matched_detections.append({
                                        "label": label,
                                        "bbox": [x1, y1, x2, y2],
                                        "bbox_orig": [x, y, w, h]
                                    })
        
        if detection_data:
            total_detections = sum(len(item) for item in detection_data if isinstance(item, dict))
            print(f"[DEBUG load_matched_detections] Image: {img_name}, Total detections in file: {total_detections}, "
                  f"Matched detections: {len(matched_detections)}")
            if matched_detections:
                print(f"[DEBUG] Matched labels: {[det['label'] for det in matched_detections]}")
                print(f"[DEBUG] Matched bboxes: {[det['bbox'] for det in matched_detections]}")
        
        return matched_detections
    except Exception as e:
        print(f"[Warning] Failed to load matched detections for {img_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def setup_roi_hooks(model, matched_detections, mask_path, image_size=448, patch_size=14,
                   roi_method="gating", roi_weight=1.0, background_weight=0.1, original_image_size=None):
    """Register forward hooks on the vision encoder to apply ROI patch weights."""
    hooks = []
    
    if not matched_detections and (not mask_path or not os.path.exists(mask_path)):
        print("[WARNING] No matched detections and no mask file, ROI hooks will not be applied")
        return hooks
    
    num_patches_per_side = image_size // patch_size
    
    if mask_path and os.path.exists(mask_path):
        print(f"[DEBUG] Creating ROI mask from mask file: {mask_path}")
        roi_mask = create_roi_mask_from_matched_detections(
            matched_detections, mask_path, image_size, patch_size, num_patches_per_side, original_image_size
        )
    elif matched_detections:
        print(f"[DEBUG] Creating ROI mask from {len(matched_detections)} detections")
        bboxes = [det["bbox"] for det in matched_detections]
        roi_mask = create_combined_roi_mask(
            bboxes, image_size, patch_size, num_patches_per_side, original_image_size=original_image_size
        )
    else:
        print("[WARNING] No detections or mask file available, cannot create ROI mask")
        return hooks
    
    def vision_hook(module, input, output):
        """Legacy hook applying ROI mask (unused if vision_hook_with_config is registered)."""
        if hasattr(output, 'last_hidden_state'):
            image_features = output.last_hidden_state
        elif isinstance(output, tuple):
            image_features = output[0]
        elif isinstance(output, torch.Tensor):
            image_features = output
        else:
            image_features = output
        
        gating = (roi_method == "gating")
        image_features = apply_patch_masking(image_features, roi_mask, gating=gating,
                                             roi_weight=roi_weight, background_weight=background_weight)
        
        if hasattr(output, 'last_hidden_state'):
            output.last_hidden_state = image_features
            return output
        else:
            return image_features
    
    if not hasattr(model, '_roi_config'):
        model._roi_config = {}
    model._roi_config.update({
        "roi_mask": roi_mask,
        "roi_method": roi_method,
        "roi_weight": roi_weight,
        "background_weight": background_weight,
        "enabled": True
    })
    
    print(f"[DEBUG] ROI mask stored in model config: shape={roi_mask.shape if roi_mask is not None else None}")
    
    hook_call_count = [0]
    
    def vision_hook_with_config(module, input, output):
        """Forward hook: apply ROI gating/masking using model._roi_config."""
        hook_call_count[0] += 1
        
        if not model._roi_config.get("enabled", False):
            if hook_call_count[0] == 1:
                print(f"[DEBUG] ROI hook called but disabled (call #{hook_call_count[0]})")
            return output
        
        roi_mask_local = model._roi_config.get("roi_mask")
        if roi_mask_local is None:
            if hook_call_count[0] == 1:
                print(f"[DEBUG] ROI hook called but no mask (call #{hook_call_count[0]})")
            return output
        
        if hasattr(output, 'last_hidden_state'):
            image_features = output.last_hidden_state
        elif isinstance(output, tuple):
            image_features = output[0]
        elif isinstance(output, torch.Tensor):
            image_features = output
        else:
            image_features = output
        
        if hook_call_count[0] == 1:
            print(f"[DEBUG] ROI hook called! (call #{hook_call_count[0]})")
            print(f"[DEBUG] Image features shape: {image_features.shape if isinstance(image_features, torch.Tensor) else 'N/A'}")
            print(f"[DEBUG] ROI mask shape: {roi_mask_local.shape if isinstance(roi_mask_local, torch.Tensor) else 'N/A'}")
            if isinstance(roi_mask_local, torch.Tensor):
                roi_patches = roi_mask_local.sum().item()
                total_patches = roi_mask_local.numel()
                print(f"[DEBUG] ROI patches: {roi_patches}/{total_patches} ({roi_patches/total_patches*100:.1f}%)")
        
        gating = (model._roi_config.get("roi_method", "gating") == "gating")
        roi_weight = model._roi_config.get("roi_weight", 1.0)
        background_weight = model._roi_config.get("background_weight", 1.0)
        
        try:
            image_features_original = image_features.clone() if isinstance(image_features, torch.Tensor) else image_features
            image_features = apply_patch_masking(image_features, roi_mask_local, gating=gating,
                                                 roi_weight=roi_weight, background_weight=background_weight)
            
            if hook_call_count[0] == 1 and isinstance(image_features, torch.Tensor) and isinstance(image_features_original, torch.Tensor):
                original_mean = image_features_original.mean().item()
                modified_mean = image_features.mean().item()
                
                if isinstance(roi_mask_local, torch.Tensor):
                    mask_flat = roi_mask_local.flatten().to(device=image_features.device)
                    num_patches = image_features.shape[1]
                    mask_length = mask_flat.shape[0]
                    
                    num_patches_to_analyze = min(mask_length, num_patches)
                    
                    if num_patches_to_analyze > 0:
                        mask_for_stats = mask_flat[:num_patches_to_analyze]
                        
                        batch_size, _, hidden_dim = image_features.shape
                        mask_expanded = mask_for_stats.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_patches_to_analyze, hidden_dim)
                        
                        features_slice_original = image_features_original[:, :num_patches_to_analyze, :]
                        features_slice_modified = image_features[:, :num_patches_to_analyze, :]
                        
                        roi_features_original = features_slice_original[mask_expanded].view(-1) if mask_expanded.sum() > 0 else torch.tensor([], device=image_features.device)
                        roi_features_modified = features_slice_modified[mask_expanded].view(-1) if mask_expanded.sum() > 0 else torch.tensor([], device=image_features.device)
                        
                        bg_mask_expanded = ~mask_expanded
                        bg_features_original = features_slice_original[bg_mask_expanded].view(-1) if bg_mask_expanded.sum() > 0 else torch.tensor([], device=image_features.device)
                        bg_features_modified = features_slice_modified[bg_mask_expanded].view(-1) if bg_mask_expanded.sum() > 0 else torch.tensor([], device=image_features.device)
                        
                        print(f"[DEBUG] Features mean before ROI: {original_mean:.6f}, after ROI: {modified_mean:.6f}")
                        if len(roi_features_original) > 0:
                            roi_patches_total = len(roi_features_original) // hidden_dim
                            roi_patches_per_sample = roi_patches_total // batch_size
                            print(f"[DEBUG] ROI region: mean before={roi_features_original.mean().item():.6f}, "
                                  f"after={roi_features_modified.mean().item():.6f}, "
                                  f"patches={roi_patches_per_sample}/{num_patches_to_analyze} per sample "
                                  f"(total: {roi_patches_total} across {batch_size} samples)")
                        if len(bg_features_original) > 0:
                            bg_patches_total = len(bg_features_original) // hidden_dim
                            bg_patches_per_sample = bg_patches_total // batch_size
                            print(f"[DEBUG] Background region: mean before={bg_features_original.mean().item():.6f}, "
                                  f"after={bg_features_modified.mean().item():.6f}, "
                                  f"patches={bg_patches_per_sample}/{num_patches_to_analyze} per sample "
                                  f"(total: {bg_patches_total} across {batch_size} samples)")
                        
                        if mask_length < num_patches:
                            extra_patches = num_patches - mask_length
                            print(f"[DEBUG] Extra patches (not in mask): {extra_patches} patches per sample")
                    else:
                        print(f"[DEBUG] Features mean before ROI: {original_mean:.6f}, after ROI: {modified_mean:.6f}")
                        print(f"[DEBUG] Warning: Cannot analyze ROI/background regions (mask_length={mask_length}, num_patches={num_patches})")
                
                print(f"[DEBUG] ROI method: {model._roi_config.get('roi_method', 'gating')}, "
                      f"roi_weight: {roi_weight}, background_weight: {background_weight}")
            
            if hasattr(output, 'last_hidden_state'):
                output.last_hidden_state = image_features
                return output
            elif isinstance(output, tuple):
                return (image_features,) + output[1:]
            else:
                return image_features
        except Exception as e:
            print(f"[Warning] Failed to apply ROI masking in hook: {e}")
            import traceback
            traceback.print_exc()
            return output
    
    vision_module = None
    vision_path = None
    
    if hasattr(model, 'vision_model'):
        vision_module = model.vision_model
        vision_path = "model.vision_model"
    elif hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            vision_module = vision_tower
            vision_path = "model.get_vision_tower()"
    elif hasattr(model, 'visual'):
        vision_module = model.visual
        vision_path = "model.visual"
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision_module = model.model.visual
        vision_path = "model.model.visual"
    elif hasattr(model, 'vision_tower'):
        vision_module = model.vision_tower
        vision_path = "model.vision_tower"
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'visual'):
        vision_module = model.encoder.visual
        vision_path = "model.encoder.visual"
    elif hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        vision_module = model.model.vision_model
        vision_path = "model.model.vision_model"
    
    if vision_module is None:
        print("[WARNING] Could not find vision encoder module. Printing model structure...")
        print(f"[DEBUG] Model type: {type(model)}")
        print(f"[DEBUG] Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
        if hasattr(model, 'model'):
            print(f"[DEBUG] model.model type: {type(model.model)}")
            print(f"[DEBUG] model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')][:20]}")
        print("[ERROR] ROI constraint will NOT be applied because vision encoder was not found!")
        return hooks
    
    try:
        hook = vision_module.register_forward_hook(vision_hook_with_config)
        hooks.append(hook)
        print(f"[DEBUG] Successfully registered ROI hook on {vision_path} ({type(vision_module).__name__})")
    except Exception as e:
        print(f"[ERROR] Failed to register ROI hook on {vision_path}: {e}")
        import traceback
        traceback.print_exc()
    
    return hooks


def generate_answer_with_roi(model, tokenizer, pixel_values, question, device="cuda",
                            matched_detections=None, mask_path=None, roi_method="gating",
                            roi_weight=1.0, background_weight=0.1, original_image_size=None,
                            image_size=448, patch_size=14):
    """model.chat with temporary ROI hooks when detections are provided."""
    hooks = []
    if matched_detections:
        hooks = setup_roi_hooks(model, matched_detections, mask_path,
                               image_size=image_size, patch_size=patch_size,
                               roi_method=roi_method, roi_weight=roi_weight,
                               background_weight=background_weight, original_image_size=original_image_size)
        
        if not hooks:
            print("[WARNING] ROI hooks were not registered. ROI constraint may not be applied.")
        else:
            print(f"[DEBUG] ROI hooks registered successfully: {len(hooks)} hooks")
    
    try:
        formatted_question = f'<image>\n{question}'
        
        generation_config = dict(
            max_new_tokens=128,
            do_sample=False
        )
        
        response = model.chat(
            tokenizer, 
            pixel_values, 
            formatted_question, 
            generation_config
        )
        
        return response.strip()
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except:
                pass
        
        if hasattr(model, '_roi_config'):
            model._roi_config["enabled"] = False


def main():
    parser = argparse.ArgumentParser(description="InternVL3-8B VQA with ROI constraints")
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
    parser.add_argument("--use-roi-constraint", type=lambda x: (str(x).lower() == 'true'), default=True,
                       nargs='?', const=True,
                       help="Enable visual ROI (patch masking/gating); pass False to disable")
    parser.add_argument("--roi-method", type=str, default="gating",
                       choices=["gating", "masking"],
                       help="ROI method: gating or masking")
    parser.add_argument("--roi-weight", type=float, default=1.0,
                       help="ROI region weight")
    parser.add_argument("--background-weight", type=float, default=0.1,
                       help="Background region weight")
    parser.add_argument("--image-size", type=int, default=448,
                       help="Image size")
    parser.add_argument("--patch-size", type=int, default=14,
                       help="Patch size")
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
    
    print(f"Processing {len(questions)} questions")
    
    results = []
    for idx, question_data in enumerate(tqdm(questions, desc="infer")):
        qid = question_data.get('qid', question_data.get('question_id', 0))
        question = question_data.get('question', question_data.get('text', ''))
        image_file = question_data.get('img_name', question_data.get('image', question_data.get('image_id', '')))
        ground_truth = question_data.get('answer', question_data.get('ground_truth', ''))
        question_type = question_data.get('question_type', 'binary')
        
        if question_type == "multi-choice":
            if len(question_data.get('choices', [])) >= 4:
                question += f" Please select from the following choices: {question_data['choices']}"
            else:
                continue
        else:
            question += " Please answer Yes or No."
        
        matched_detections = []
        mask_path = None
        roi_boxes = None
        
        if args.use_roi_constraint:
            matched_detections = load_matched_detections(args.image_folder, image_file, question)
            
            print(f"[DEBUG] Image: {image_file}, Question: {question[:50]}...")
            print(f"[DEBUG] Matched detections count: {len(matched_detections)}")
            
            if matched_detections:
                image_path = build_image_path(args.image_folder, image_file)
                print(f"[DEBUG] Looking for mask file, image_path: {image_path}, exists: {os.path.exists(image_path)}")
                mask_path = find_mask_file(image_path)
                
                if mask_path:
                    print(f"[DEBUG] Using mask file: {mask_path}")
                else:
                    roi_boxes = [det["bbox"] for det in matched_detections]
                    print(f"[DEBUG] Mask file not found, using bboxes: {roi_boxes}")
                    image_dir = os.path.dirname(image_path)
                    if os.path.exists(image_dir):
                        mask_files = [f for f in os.listdir(image_dir) if 'mask' in f.lower()]
                        print(f"[DEBUG] Available mask files in {image_dir}: {mask_files}")
            else:
                mask_path = None
                roi_boxes = None
                print(f"[DEBUG] Skipping ROI constraint (no matched detections)")
        
        image_path = build_image_path(args.image_folder, image_file)
        
        try:
            if not os.path.exists(image_path):
                print(f'[Error] Image not found: {image_path}')
                result = {
                    "question_id": qid,
                    "prompt": question,
                    "model_answer": "",
                    "ground_truth": ground_truth,
                    "image_id": image_file,
                    "model_id": args.model_path,
                    "has_roi": False,
                    "metadata": {"error": "image_not_found"}
                }
                results.append(result)
                continue
            
            pixel_values = load_image(image_path, input_size=args.image_size, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16)
            if args.device == "cuda":
                pixel_values = pixel_values.cuda()
            
            try:
                original_image = Image.open(image_path)
                original_image_size = max(original_image.size) if original_image else None
            except:
                original_image_size = None
            
            has_roi = False
            roi_method_used = None
            if args.use_roi_constraint and matched_detections:
                print(f"[DEBUG] Applying ROI constraint: method={args.roi_method}, "
                      f"roi_weight={args.roi_weight}, background_weight={args.background_weight}, "
                      f"num_detections={len(matched_detections)}")
                has_roi = True
                roi_method_used = args.roi_method
            
            answer = generate_answer_with_roi(
                model, tokenizer, pixel_values, question, args.device,
                matched_detections=matched_detections if has_roi else None, 
                mask_path=mask_path if has_roi else None,
                roi_method=args.roi_method, roi_weight=args.roi_weight,
                background_weight=args.background_weight, original_image_size=original_image_size,
                image_size=args.image_size, patch_size=args.patch_size
            )
            
            metadata = {}
            if has_roi:
                metadata["roi_weight"] = args.roi_weight
                metadata["background_weight"] = args.background_weight
                metadata["num_detections"] = len(matched_detections) if matched_detections else 0
                if mask_path:
                    metadata["mask_path"] = mask_path
            
            result = {
                "question_id": qid,
                "prompt": question,
                "model_answer": answer,
                "ground_truth": ground_truth,
                "image_id": image_file,
                "model_id": args.model_path,
                "has_roi": has_roi,
                "roi_method": roi_method_used,
                "metadata": metadata
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
                "has_roi": False,
                "metadata": {"error": str(e)}
            }
            results.append(result)
    
    with open(args.answers_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Done. Saved to: {args.answers_file}")


if __name__ == "__main__":
    main()

