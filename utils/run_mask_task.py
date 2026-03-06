from typing import List, Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image


def run_mask_task(
    model_dict: Dict[str, Any],
    frame_paths: List[str],
    question: str,
    crop_caption: str = "",
    crop_category: str = ""
) -> Tuple[str, List[np.ndarray]]:
    model = model_dict['model']
    model_type = model_dict['model_type']
    tokenizer = model_dict.get('tokenizer')
    processor = model_dict.get('processor')
    
    if model_type == "unipixel":
        from utils.unipixel_helper import run_unipixel_mask
        return run_unipixel_mask(model, processor, frame_paths, question)

    images = [Image.open(fp).convert('RGB') for fp in frame_paths]
    text_prompts = f"<image>{question}"
    
    if model_type in ["sa2va_qwen3", "sa2va_qwen2_5"]:
        input_dict = {
            'video': images,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'processor': processor,
        }
    else:
        input_dict = {
            'video': images,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }
    
    with torch.inference_mode():
        return_dict = model.predict_forward(**input_dict)
    
    answer = return_dict.get("prediction", "")
    answer = answer.replace('<|im_end|>', '').strip()
    masks = return_dict.get("prediction_masks", [])
    
    return answer, masks