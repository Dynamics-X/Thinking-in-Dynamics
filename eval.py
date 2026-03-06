import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import OrderedDict, defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from colorama import Fore, init

from utils.load_config import load_config
from utils.load_model import load_model, get_model_type
from utils.load_tasks import load_all_tasks, get_task_statistics
from utils.cal_metrics import (
    compute_qa_accuracy,
    compute_jf_score,
    load_gt_masks,
    compute_category_metrics,
    fuzzy_matching
)
from utils.run_qa_task import run_qa_task
from utils.run_mask_task import run_mask_task
from utils.save_results import save_results

init(autoreset=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def verify_dataset_structure(datasets_dir: Path) -> bool:
    required_dirs = ['frames', 'masks', 'multi_json']
    
    for dir_name in required_dirs:
        dir_path = datasets_dir / dir_name
        if not dir_path.exists():
            print(f"{Fore.RED}ERROR: Directory {dir_path} not found")
            return False
        print(f"{Fore.GREEN}Found {dir_name} directory")
    
    return True


def validate_config(config: Dict) -> None:
    model_path = config.get('model', {}).get('model_path', '')
    if not model_path or model_path.strip() == '':
        print(f"{Fore.RED}ERROR: <model_path> is required in config.yaml")
        raise SystemExit(1)
    
    if not os.path.exists(model_path):
        print(f"{Fore.RED}ERROR: <model_path> does not exist -> {model_path}")
        raise SystemExit(1)
    
    model_name = config.get('model', {}).get('model_name', '')
    if not model_name or model_name.strip() == '':
        print(f"{Fore.RED}ERROR: <model_name> is required in config.yaml")
        raise SystemExit(1)
    
    task_type = config.get('task', '')
    valid_tasks = ['all', 'qa', 'mask']
    if task_type not in valid_tasks:
        print(f"{Fore.RED}ERROR: Unknown task type -> {task_type}")
        raise SystemExit(1)


def main():
    config = load_config("conf/config.yaml")
    
    validate_config(config)
    
    datasets_dir = Path(config['datasets']['datasets_path'])
    
    if not datasets_dir.exists():
        print(f"{Fore.RED}ERROR: Datasets directory not found -> {datasets_dir}")
        raise SystemExit(1)
    
    if not verify_dataset_structure(datasets_dir):
        print(f"{Fore.RED}ERROR: Datasets structure verification failed")
        raise SystemExit(1)
    
    model_path = config['model']['model_path']
    model_name = config['model']['model_name']
    task_type = config['task']
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Datasets Path: {datasets_dir}")
    print(f"  Model Path: {model_path}")
    print(f"  Model Name: {model_name}")
    print(f"  Task: {task_type}")
    print(f"  Device: {config['model']['device']}")
    print(f"{'='*60}")

    model_type = get_model_type(model_name)
    supports_mask = model_type in ["sa2va", "sa2va_internvl3", "sa2va_qwen2_5", "sa2va_qwen3", "unipixel"]
    
    if task_type in ['mask', 'all'] and not supports_mask:
        print(f"{Fore.RED}ERROR: {model_name} only supports QA task")
        raise SystemExit(1)
    
    if task_type == 'all':
        tasks = load_all_tasks(datasets_dir, 'all')
    elif task_type == 'qa':
        tasks = load_all_tasks(datasets_dir, 'qa')
    else:
        tasks = load_all_tasks(datasets_dir, 'mask')
    
    if not tasks:
        print(f"{Fore.RED}ERROR: No task found")
        raise SystemExit(1)
    
    stats = get_task_statistics(tasks)
    for category, counts in stats['by_category'].items():
        print(f"{category}:")
        if counts['qa'] > 0:
            print(f"  QA: {counts['qa']}")
        if counts['mask'] > 0:
            print(f"  Mask: {counts['mask']}")
    
    print(f"Loading model: {model_name}")
    model_dict = load_model(config)
    model = model_dict['model']
    loaded_model_type = model_dict['model_type']
    
    if 'processor' in model_dict:
        processor = model_dict['processor']
        tokenizer = None
    else:
        tokenizer = model_dict['tokenizer']
        processor = None
    
    print(f"{Fore.GREEN}Model loaded successfully")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    results_dir = Path(config['result_path'])
    model_dir_name = model_name.split('/')[-1]
    mask_output_dir = results_dir / model_dir_name / "mask_details"
    
    results = []
    category_results = defaultdict(lambda: {"qa": [], "mask": []})
    
    for task in tqdm(tasks, desc="Evaluating"):
        frame_paths = task["frame_paths"]
        result = task.copy()
        
        if task["is_segmentation"]:
            try:
                answer, pred_masks = run_mask_task(
                    model_dict=model_dict,
                    frame_paths=frame_paths,
                    question=task["question"],
                    crop_caption=task.get("crop_caption", ""),
                    crop_category=task.get("crop_category", "")
                )
                result["prediction"] = answer
                
                if task_type in ['mask', 'all']:
                    dataset = task["dataset"]
                    scene_name = task["scene_name"]
                    task_suffix = task["task_type"]
                    
                    mask_save_dir = mask_output_dir / dataset / f"{scene_name}_{task_suffix}_output"
                    mask_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    if isinstance(pred_masks, list) and len(pred_masks) > 0:
                        if isinstance(pred_masks[0], np.ndarray) and pred_masks[0].ndim == 3:
                            pred_masks = [pred_masks[0][i] for i in range(pred_masks[0].shape[0])]
                    
                    for idx, mask in enumerate(pred_masks):
                        if isinstance(mask, np.ndarray):
                            mask = np.squeeze(mask)
                            
                            if mask.ndim != 2:
                                print(f"{Fore.YELLOW}WARN: Mask shape is {mask.shape}, expected 2D, Skip frame {idx}")
                                continue
                            
                            if mask.dtype != np.uint8:
                                mask_img = (mask > 0.5).astype(np.uint8) * 255
                            else:
                                mask_img = mask
                            Image.fromarray(mask_img).save(mask_save_dir / f"frame_{idx:04d}.png")
                
                gt_masks = load_gt_masks(
                    task["mask_dir"],
                    task["object_id"],
                    len(task["frame_paths"])
                )
                
                boundary_th = config['evaluation'].get('boundary_threshold', 2)
                jf = compute_jf_score(pred_masks, gt_masks, boundary_th)
                
                result["J"] = jf["J"]
                result["F"] = jf["F"]
                result["J&F"] = jf["J&F"]
                
                category_results[task["category"]]["mask"].append(jf["J&F"])
                
            except Exception as e:
                import traceback
                print(f"{Fore.YELLOW}WARN: Failure in segmentation task, skip")
                print(f"  Details: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["J&F"] = 0.0
                category_results[task["category"]]["mask"].append(0.0)
        
        else:
            try:
                answer = run_qa_task(
                    model_dict=model_dict,
                    frame_paths=frame_paths,
                    question=task["question"],
                    options=task["options"]
                )
                result["prediction"] = answer
                
                accuracy = compute_qa_accuracy(answer, task["answer"])
                result["accuracy"] = accuracy
                
                category_results[task["category"]]["qa"].append(accuracy)
                
            except Exception as e:
                import traceback
                print(f"{Fore.YELLOW}WARN: Failure in QA task, skip")
                print(f"  Details: {e}")
                traceback.print_exc()
                result["prediction"] = ""
                result["accuracy"] = 0.0
                category_results[task["category"]]["qa"].append(0.0)
        
        results.append(result)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nEvaluation Results:")
    
    summary = OrderedDict()
    all_scores = []
    
    for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
        qa_scores = category_results[category]["qa"]
        mask_scores = category_results[category]["mask"]
        
        print(f"{category}:")
        
        if qa_scores and task_type in ['qa', 'all']:
            qa_acc = np.mean(qa_scores) * 100
            summary[f"{category}_QA_Accuracy"] = qa_acc
            all_scores.append(qa_acc)
            print(f"  QA Accuracy: {qa_acc:.2f}% ({len(qa_scores)} samples)")
        
        if mask_scores and task_type in ['mask', 'all']:
            mask_jf = np.mean(mask_scores) * 100
            summary[f"{category}_Mask_J&F"] = mask_jf
            all_scores.append(mask_jf)
            print(f"  Mask J&F: {mask_jf:.2f}% ({len(mask_scores)} samples)")
    
    if all_scores:
        overall = np.mean(all_scores)
        summary["Overall"] = overall
        print(f"  Overall Score: {overall:.2f}%")
    
    save_results(results, config, summary, task_type)

if __name__ == "__main__": 
    main()