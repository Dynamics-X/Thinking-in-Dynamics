from typing import Dict, Any
import os
import re
import torch
from colorama import Fore, init
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor, AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
init(autoreset=True)


def is_flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def get_model_type(model_name: str) -> str:
    model_name_lower = model_name.lower()
    
    if "bytedance/sa2va" in model_name_lower:
        if "qwen3-vl" in model_name_lower:
            return "sa2va_qwen3"
        elif "qwen2_5-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower:
            return "sa2va_qwen2_5"
        elif "internvl3" in model_name_lower:
            return "sa2va_internvl3"
        else:
            return "sa2va"
    
    elif "unipixel" in model_name_lower:
        return "unipixel"
    
    elif re.search(r"opengvlab/internvl3[._]5", model_name_lower):
        return "internvl3_5"
    
    elif "opengvlab/internvl3" in model_name_lower:
        return "internvl3"
    
    elif re.search(r"qwen/qwen3-vl-\d+b-a\d+b", model_name_lower):
        return "qwen3_vl_moe"
    
    elif "qwen/qwen3-vl" in model_name_lower:
        return "qwen3_vl"
    
    elif "qwen/qwen2.5-vl" in model_name_lower:
        return "qwen2_5_vl"
    
    elif "llava-onevision" in model_name_lower or "llava-one-vision" in model_name_lower:
        return "llava_onevision"
    
    elif re.search(r"vst-\d+b", model_name_lower):
        return "vst"
    
    elif "spatial-ssrl" in model_name_lower:
        return "spatial_ssrl"
    
    elif "spatialladder" in model_name_lower:
        return "spatial_ladder"
    
    elif "spacer-sft" in model_name_lower:
        return "spacer_sft"
    else:
        print(f"{Fore.RED}ERROR: Unknown model type -> {model_name}")
        raise SystemExit(1)


def load_model(config: Dict) -> Dict[str, Any]:
    model_path = config['model']['model_path']
    model_name = config['model']['model_name']
    device = config['model']['device']
    
    model_type = get_model_type(model_name)
    print(f"Model type: {model_type}")
    
    torch_dtype_str = config['model'].get('torch_dtype', 'bfloat16')
    if torch_dtype_str == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, torch_dtype_str)

    use_flash_attn = config['model'].get('use_flash_attn', True)
    flash_attn_available = is_flash_attn_available()
    
    if use_flash_attn and not flash_attn_available:
        print(f"{Fore.YELLOW}WARN: Flash Attention requested but not installed, falling back to eager attention")
        use_flash_attn = False
    
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    trust_remote_code = config['model'].get('trust_remote_code', True)

    model = None
    tokenizer = None
    processor = None
    supports_mask = False
    
    local_files_only = True
    
    if model_type in ["sa2va", "sa2va_internvl3", "sa2va_qwen2_5", "sa2va_qwen3"]:
        supports_mask = True
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        ).eval().cuda()
        if model_type in ["sa2va_qwen3", "sa2va_qwen2_5"]:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=local_files_only)

    elif model_type in ["internvl3", "internvl3_5"]:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            local_files_only=local_files_only
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=local_files_only)

    elif model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)

    elif model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)

    elif model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)

    elif model_type == "llava_onevision":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)

    elif model_type == "vst":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256*28*28,
            max_pixels=1280*28*28,
            local_files_only=local_files_only
        )

    elif model_type == "spatial_ssrl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)

    elif model_type == "spatial_ladder":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    
    elif model_type == "spacer_sft":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map="auto",
            local_files_only=local_files_only
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    
    elif model_type == "unipixel":
        from utils.unipixel_helper import load_unipixel_model
        supports_mask = True
        model, processor = load_unipixel_model(
            model_name=model_path,
            device=device,
            local_files_only=local_files_only
        )
        tokenizer = None
    
    if model is None:
        print(f"{Fore.RED}ERROR: Fail to load model from {model_path}")
        raise SystemExit(1)
    
    if processor:
        print(f"{Fore.GREEN}Processor loaded successfully")
    else:
        print(f"{Fore.GREEN}Tokenizer loaded successfully")
    
    print(f"Precision: {torch_dtype_str}")
    print(f"Flash Attention: {attn_implementation} {f'{Fore.GREEN}(checked)' if attn_implementation == 'flash_attention_2' else f'{Fore.YELLOW}(fallback eager)'}")
    print(f"Supports Mask: {supports_mask}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'processor': processor,
        'model_type': model_type,
        'device': device,
        'supports_mask': supports_mask
    }

