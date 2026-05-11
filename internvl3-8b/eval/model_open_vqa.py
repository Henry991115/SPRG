"""Open-ended VQA inference (baseline)."""
import argparse
import math
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

from model_vqa_med import load_model_and_tokenizer, load_image




def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    if n <= 0:
        n = 1
    if len(lst) == 0:
        return []
    chunk_size = max(1, math.ceil(len(lst) / n))
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks"""
    if n <= 0:
        n = 1
    if k < 0:
        k = 0
    chunks = split_list(lst, n)
    return chunks[k] if k < len(chunks) else []


def generate_open_answer(model, tokenizer, pixel_values, question, device="cuda", 
                        temperature=0.2, top_p=None, num_beams=1, max_new_tokens=1024):
    """Generate an open-ended answer (optional sampling)."""
    formatted_question = f'<image>\n{question}'
    
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        num_beams=num_beams if temperature == 0 else 1,
    )
    
    generation_config = {k: v for k, v in generation_config.items() if v is not None}
    
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
    parser = argparse.ArgumentParser(description="InternVL3-8B open-ended VQA inference")
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL3_5-8B",
                       help="Model path or Hugging Face ID")
    parser.add_argument("--question-file", type=str, required=True,
                       help="Path to questions JSON or JSONL")
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
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p nucleus sampling (optional)")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Beam size when temperature=0 (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                       help="Max new tokens (default: 1024)")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    
    print("[DEBUG] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    print("[DEBUG] Model loaded")
    
    file_path = os.path.expanduser(args.question_file)
    if file_path.endswith('.jsonl'):
        # Handle JSONL files
        with open(file_path, 'r', encoding='utf-8') as file:
            questions = [json.loads(line) for line in file]
    else:
        # Handle JSON files
        with open(file_path, 'r', encoding='utf-8') as file:
            questions = json.load(file)
    
    original_count = len(questions)
    
    need_questions = []
    for i in questions:
        question_type = i.get("question_type", "")
        if not question_type or question_type == "type4_Knowledge" or question_type == "type_2":
            need_questions.append(i)
    questions = need_questions
    print(f"[DEBUG] Questions: {original_count} raw, {len(need_questions)} after filter")
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    print(f"Processing chunk: {len(questions)} questions")
    
    results = []
    ans_file = open(args.answers_file, "w", encoding='utf-8')
    
    for line in tqdm(questions, desc="infer"):
        question_type = line.get("question_type", "")
        if question_type and question_type not in ["type_2", "type4_Knowledge"]:
            continue
            
        image_file = line.get("img_name", line.get("image", line.get("image_id", "")))
        idx = line.get("qid", line.get("question_id", 0))
        question = line.get("question", line.get("text", ""))
        gt_ans = line.get("answer", line.get("ground_truth", ""))
        stru_ans = line.get("structured_answer", "")
        question_type = line.get("question_type", "")
        
        image_path = os.path.join(args.image_folder, image_file)
        
        try:
            pixel_values = load_image(image_path, input_size=448, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16)
            if args.device == "cuda":
                pixel_values = pixel_values.cuda()
            
            outputs = generate_open_answer(
                model, tokenizer, pixel_values, question, args.device,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens
            )
            
            ans_id = shortuuid.uuid()
            result = {
                "question_id": idx,
                "prompt": question,
                "model_answer": outputs,
                "ground_truth": gt_ans,
                "question_type": question_type,
                "structured_answer": stru_ans,
                "image_id": image_file,
                "answer_id": ans_id,
                "model_id": args.model_path,
                "metadata": {}
            }
            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            ans_file.flush()
            
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            import traceback
            traceback.print_exc()
            ans_id = shortuuid.uuid()
            result = {
                "question_id": idx,
                "prompt": question,
                "model_answer": f"Error: {str(e)}",
                "ground_truth": gt_ans,
                "question_type": question_type,
                "structured_answer": stru_ans,
                "image_id": image_file,
                "answer_id": ans_id,
                "model_id": args.model_path,
                "metadata": {"error": str(e)}
            }
            ans_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            ans_file.flush()
    
    ans_file.close()
    print(f"Done. Saved to: {args.answers_file}")


if __name__ == "__main__":
    main()

