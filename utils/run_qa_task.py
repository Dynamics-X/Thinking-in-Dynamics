from typing import List, Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
from colorama import Fore, init

init(autoreset=True)


def run_qa_task(
    model_dict: Dict[str, Any],
    frame_paths: List[str],
    question: str,
    options: List[str] = None
) -> str:
    model = model_dict['model']
    model_type = model_dict['model_type']
    tokenizer = model_dict.get('tokenizer')
    processor = model_dict.get('processor')
    device = model_dict['device']
    
    if options:
        options_text = "\n".join(options)
        base_question = f"{question}\n{options_text}\nAnswer with the option's letter from the given choices directly."
    else:
        base_question = question
    
    if model_type in ["sa2va", "sa2va_internvl3", "sa2va_qwen2_5", "sa2va_qwen3"]:
        return _run_sa2va_qa(model, tokenizer, processor, frame_paths, base_question, model_type)
    
    elif model_type in ["internvl3", "internvl3_5"]:
        return _run_internvl_qa(model, tokenizer, frame_paths, base_question)
    
    elif model_type == "qwen2_5_vl":
        return _run_qwen25_vl_qa(model, processor, frame_paths, base_question)
    
    elif model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        return _run_qwen3_vl_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "llava_onevision":
        return _run_llava_onevision_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "unipixel":
        return _run_unipixel_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "vst":
        return _run_vst_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "spatial_ssrl":
        return _run_spatial_ssrl_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "spatial_ladder":
        return _run_spatial_ladder_qa(model, processor, frame_paths, base_question)
    
    elif model_type == "spacer_sft":
        return _run_spacer_sft_qa(model, processor, frame_paths, base_question)
    
    else:
        print(f"{Fore.RED}ERROR: Unsupported model type for QA -> {model_type}")
        raise SystemExit(1)


def _run_sa2va_qa(model, tokenizer, processor, frame_paths, question, model_type):
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
    
    return answer


def _run_internvl_qa(model, tokenizer, frame_paths, question):
    from .video_utils import load_video_frames_for_internvl
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    pixel_values, num_patches_list = load_video_frames_for_internvl(frame_paths)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    full_question = video_prefix + question
    
    generation_config = dict(max_new_tokens=128, do_sample=False)
    
    response = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False
    )
    
    return response


def _run_qwen25_vl_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def _run_qwen3_vl_qa(model, processor, frame_paths, question):
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    video_metadata = [{"fps": 24.0, "total_num_frames": len(images)}]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        video_metadata=video_metadata
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def _run_llava_onevision_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    
    generate_kwargs = {k: v for k, v in inputs.items() if k not in ("second_per_grid_ts",)}
    
    generated_ids = model.generate(**generate_kwargs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def _run_vst_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    THINK_SYSTEM_PROMPT = "You are a helpful assistant. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here."
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": THINK_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    response = output_text[0]
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    return response


def _run_spatial_ssrl_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    format_prompt = "\n You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question + format_prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    response = output_text[0]
    import re
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1)
    
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    
    return response


def _run_spatial_ladder_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def _run_spacer_sft_qa(model, processor, frame_paths, question):
    from qwen_vl_utils import process_vision_info
    
    images = [Image.open(p).convert('RGB') for p in frame_paths]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": images},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def _run_unipixel_qa(model, processor, frame_paths: List[str], question: str) -> str:
    from utils.unipixel_helper import run_unipixel_qa
    
    response = run_unipixel_qa(model, processor, frame_paths, question)
    return response
