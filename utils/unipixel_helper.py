import sys
import os
from pathlib import Path
from typing import List, Tuple, Any, Dict
import numpy as np
import torch
from PIL import Image
from colorama import Fore, init
init(autoreset=True)


def check_unipixel_installed():
    try:
        import unipixel
        return True
    except ImportError:
        return False


def setup_unipixel_path():
    bench_dir = Path(__file__).parent.parent
    unipixel_dir = bench_dir / "thirdparty" / "UniPixel"
    
    if not unipixel_dir.exists():
        print(f"{Fore.RED}ERROR: UniPixel repo not found -> {unipixel_dir}")
        raise SystemExit(1)
    
    unipixel_str = str(unipixel_dir)
    if unipixel_str not in sys.path:
        sys.path.insert(0, unipixel_str)


def load_unipixel_model(
    model_name: str, 
    device: str = "cuda",
    local_files_only: bool = True
):
    setup_unipixel_path()
    
    try:
        from unipixel.model.builder import build_model_with_cache
    except ImportError:
        try:
            from unipixel.model.builder import build_model
            print(f"{Fore.YELLOW}WARN: Use original build_model")
        except ImportError as e:
            print(f"{Fore.RED}ERROR: Can not import unipixel.model.builder.build_model")
            print(f"  Details: {e}")
            raise SystemExit(1)
        
        print(f"Loading model: {model_name}")
        model, processor = build_model(model_name)
        model = model.to(device)
        return model, processor
    
    print(f"Loading model: {model_name}")
    
    model, processor = build_model_with_cache(
        model_name,
        local_files_only=local_files_only
    )
    model = model.to(device)
    
    return model, processor


def run_unipixel_qa(
    model: Any,
    processor: Any,
    frame_paths: List[str],
    question: str
) -> str:
    from unipixel.dataset.utils import process_vision_info
    from unipixel.utils.io import load_image, load_video
    from unipixel.utils.transforms import get_sam2_transform
    
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    
    frames_list = [np.array(Image.open(p).convert('RGB')) for p in frame_paths]
    frames = np.stack(frames_list, axis=0)
    
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': frame_paths,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * max(1, int(16 / len(frame_paths)))
            },
            {
                'type': 'text',
                'text': question
            }
        ]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
    frames_tensor = torch.from_numpy(frames)
    data['frames'] = [sam2_transform(frames_tensor).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    
    model.seg = []
    
    with torch.no_grad():
        output_ids = model.generate(
            **data.to(device),
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512
        )
    
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    return response


def run_unipixel_mask(
    model: Any,
    processor: Any,
    frame_paths: List[str],
    question: str
) -> Tuple[str, List[np.ndarray]]:
    from unipixel.dataset.utils import process_vision_info
    from unipixel.utils.io import load_image, load_video
    from unipixel.utils.transforms import get_sam2_transform
    
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    
    frames_list = [np.array(Image.open(p).convert('RGB')) for p in frame_paths]
    frames = np.stack(frames_list, axis=0)
    
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': frame_paths,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * max(1, int(16 / len(frame_paths)))
            },
            {
                'type': 'text',
                'text': question
            }
        ]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
    frames_tensor = torch.from_numpy(frames)
    data['frames'] = [sam2_transform(frames_tensor).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    
    model.seg = []
    
    with torch.no_grad():
        output_ids = model.generate(
            **data.to(device),
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            repetition_penalty=None,
            max_new_tokens=512
        )
    
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    masks = []
    if len(model.seg) >= 1:
        for seg in model.seg:
            if isinstance(seg, np.ndarray):
                seg_array = seg
            else:
                seg_array = seg.cpu().numpy() if hasattr(seg, 'cpu') else np.array(seg)
            
            seg_array = np.squeeze(seg_array)
            
            if seg_array.ndim == 3:
                for i in range(seg_array.shape[0]):
                    masks.append(seg_array[i])
            elif seg_array.ndim == 2:
                masks.append(seg_array)
            else:
                print(f"{Fore.YELLOW}WARN: Unexpected UniPixel mask shape -> {seg_array.shape}")
    
    if len(masks) == 0:
        H, W = frames.shape[1:3]
        masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(len(frames_list))]
    
    return response, masks



