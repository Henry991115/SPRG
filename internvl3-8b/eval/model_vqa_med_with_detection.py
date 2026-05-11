"""VQA with detection JSON injected into the text prompt."""
import argparse
import torch
import os
import sys
import json
import re
from tqdm import tqdm
import shortuuid
from typing import List
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed, logging

logging.set_verbosity_error()

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_path)

eval_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, eval_dir)

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
    "tumour": ["cancer", "tumor", "tumour", "mass", "lesion", "malignancy"],
    "lung cancer": ["lung", "cancer", "tumor", "tumour", "pulmonary", "malignancy"],
    
    "fracture": ["fracture", "bone", "broken", "break"],
    "bone": ["fracture", "bone", "skeleton"],
    
    "infection": ["infection", "inflammatory", "inflammation", "pneumonia"],
    "pneumonia": ["lung", "pneumonia", "infection", "inflammatory"],
    
    "effusion": ["effusion", "fluid", "pleural", "pericardial"],
    "pleural effusion": ["lung", "pleura", "effusion", "fluid"],
    
    "kidney": ["kidney", "renal"],
    "brain": ["brain", "cerebral", "intracranial"],
}


def extract_keywords(text: str) -> set:
    """Tokenize question into alphanumeric keywords (len > 2)."""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return set(token for token in tokens if len(token) > 2)


def extract_medical_entities(question: str) -> set:
    """Expand keywords using MEDICAL_TERM_MAPPING synonyms."""
    question_lower = question.lower()
    keywords = extract_keywords(question)
    related_terms = set(keywords)
    
    for term, related in MEDICAL_TERM_MAPPING.items():
        if term in question_lower:
            related_terms.update(related)
    
    for keyword in keywords:
        if keyword in MEDICAL_TERM_MAPPING:
            related_terms.update(MEDICAL_TERM_MAPPING[keyword])
    
    return related_terms


def label_matches_question(label: str, question_entities: set, question_text: str) -> bool:
    """Return True if a detection label aligns with question entities/text."""
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


def load_detection_info(image_folder: str, img_name: str, question_text: str = "") -> str:
    """Load detection.json and format boxes filtered by question entities."""
    try:
        img_dir = os.path.dirname(img_name) if "/" in img_name else os.path.splitext(img_name)[0]
        detection_paths = [
            os.path.join(image_folder, img_dir, "detection.json"),
            os.path.join(image_folder, os.path.dirname(img_dir), "detection.json"),
            os.path.join(image_folder, "detection.json"),
        ]
        
        detection_data = None
        for det_path in detection_paths:
            if os.path.exists(det_path):
                with open(det_path, 'r', encoding='utf-8') as f:
                    detection_data = json.load(f)
                break
        
        if not detection_data:
            return ""
        
        if not question_text:
            matched_detections = detection_data if isinstance(detection_data, list) else []
        else:
            question_entities = extract_medical_entities(question_text)
            matched_detections = []
            
            if isinstance(detection_data, list):
                for item in detection_data:
                    if isinstance(item, dict):
                        for label, bbox in item.items():
                            if len(bbox) >= 4:
                                if label_matches_question(label, question_entities, question_text):
                                    matched_detections.append({label: bbox})
        
        if not matched_detections:
            return ""
        
        detection_text = "Detection information:\n"
        for det in matched_detections:
            for label, bbox in det.items():
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                detection_text += f"- {label}: bounding box at ({x:.1f}, {y:.1f}) with size {w:.1f}x{h:.1f}\n"
        
        return detection_text.strip()
    
    except Exception as e:
        print(f"[Warning] Failed to load detection info for {img_name}: {e}")
        return ""


def format_structured_context(detection_info: str) -> str:
    """Passthrough wrapper for detection text."""
    if not detection_info:
        return ""
    return detection_info


def main():
    parser = argparse.ArgumentParser(description="InternVL3-8B VQA with detection context")
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
    
    detection_used_count = 0
    
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
        
        has_detection = False
        detection_context = ""
        if idx == 0:
            print("[DEBUG] Loading detection...")
        detection_info = load_detection_info(args.image_folder, image_file, question)
        if detection_info:
            detection_context = format_structured_context(detection_info)
            has_detection = True
            detection_used_count += 1
            question = detection_context + "\n\n" + question
            if is_binary:
                question += "\n\nPlease provide a direct answer: Yes or No."
            if idx == 0:
                print(f"[DEBUG] Detection loaded, context length: {len(detection_context)}")
        elif idx == 0:
            print("[DEBUG] No detection info")
        
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
                "has_detection_context": has_detection,
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
                "has_detection_context": has_detection,
                "metadata": {"error": str(e)}
            }
            results.append(result)
    
    with open(args.answers_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Used detection context: {detection_used_count}/{len(questions_chunk)} ({detection_used_count/len(questions_chunk)*100:.1f}%)")
    print(f"Done. Saved to: {args.answers_file}")


if __name__ == "__main__":
    main()

