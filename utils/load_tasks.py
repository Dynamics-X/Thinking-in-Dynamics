import json
import glob
from pathlib import Path
from typing import List, Dict, Any
from colorama import Fore, init

init(autoreset=True)


TASK_CATEGORIES = {
    "cameramask": "Camera-Object",
    "cameraqa": "Camera-Object",
    "objmask": "Inter-Object",
    "qa": "Inter-Object",
    "scenemask": "Object-Scene",
    "sceneqa": "Object-Scene",
}
QA_TASK_SUFFIXES = ["cameraqa", "qa", "sceneqa"]
MASK_TASK_SUFFIXES = ["cameramask", "objmask", "scenemask"]


def load_all_tasks(datasets_dir: Path, task_type: str = "all") -> List[Dict[str, Any]]:
    samples = []
    multi_json_dir = datasets_dir / "multi_json"
    
    if not multi_json_dir.exists():
        print(f"{Fore.YELLOW}WARN: Multi JSON directory not found -> {multi_json_dir}")
        return samples

    
    for dataset_dir in sorted(multi_json_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        json_files = sorted(dataset_dir.glob("*.json"))
        for json_file in json_files:
            if "frame_sampling" in json_file.name:
                continue

            filename = json_file.stem
            
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                print(f"{Fore.YELLOW}WARN: Skip invalid filename -> {filename}")
                continue
            
            scene_name = parts[0]
            task_suffix = parts[1]
            
            if task_suffix not in QA_TASK_SUFFIXES + MASK_TASK_SUFFIXES:
                print(f"{Fore.YELLOW}WARN: Unknown task type -> {task_suffix}")
                continue

            is_segmentation = task_suffix in MASK_TASK_SUFFIXES
            if task_type == "qa" and is_segmentation:
                continue
            elif task_type == "mask" and not is_segmentation:
                continue

            category = TASK_CATEGORIES.get(task_suffix, "Unknown")

            frames_dir = datasets_dir / "frames" / dataset_name / scene_name
            if not frames_dir.exists():
                print(f"{Fore.YELLOW}WARN: Frames directory not found -> {frames_dir}")
                continue

            frame_paths = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
            if not frame_paths:
                frame_paths = sorted(glob.glob(str(frames_dir / "frame_*.png")))
            if not frame_paths:
                frame_paths = sorted(glob.glob(str(frames_dir / "*.jpg")))
            if not frame_paths:
                frame_paths = sorted(glob.glob(str(frames_dir / "*.png")))

            if not frame_paths:
                print(f"{Fore.YELLOW}WARN: No frames found in {frames_dir}")
                continue

            mask_dir = datasets_dir / "masks" / dataset_name / scene_name

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
            except Exception as e:
                print(f"{Fore.YELLOW}WARN: Can not load {json_file}")
                print(f"  Details: {e}")
                continue

            if is_segmentation:
                for obj_id, obj_data in task_data.items():
                    sample = {
                        "scene_name": scene_name,
                        "dataset": dataset_name,
                        "task_type": task_suffix,
                        "category": category,
                        "is_segmentation": True,
                        "frame_paths": frame_paths,
                        "mask_dir": str(mask_dir),
                        "question": obj_data.get("question", ""),
                        "crop_caption": obj_data.get("crop_caption", ""),
                        "crop_category": obj_data.get("crop_category", ""),
                        "options": None,
                        "answer": None,
                        "object_id": obj_id,
                    }
                    samples.append(sample)
            else:
                for idx, qa_item in enumerate(task_data):
                    sample = {
                        "scene_name": scene_name,
                        "dataset": dataset_name,
                        "task_type": task_suffix,
                        "category": category,
                        "is_segmentation": False,
                        "frame_paths": frame_paths,
                        "mask_dir": None,
                        "question": qa_item.get("question", ""),
                        "options": qa_item.get("options", []),
                        "answer": qa_item.get("answer", ""),
                        "object_id": None,
                        "question_idx": idx,
                    }
                    samples.append(sample)
    
    qa_count = sum(1 for s in samples if not s["is_segmentation"])
    mask_count = sum(1 for s in samples if s["is_segmentation"])
    
    print(f"Loaded {len(samples)} tasks in total")
    print(f"  QA tasks: {qa_count}")
    print(f"  Mask tasks: {mask_count}")
    
    return samples


def get_task_statistics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "total": len(tasks),
        "qa": 0,
        "mask": 0,
        "by_category": {},
        "by_dataset": {},
    }
    
    for task in tasks:
        if task["is_segmentation"]:
            stats["mask"] += 1
        else:
            stats["qa"] += 1
        
        category = task["category"]
        if category not in stats["by_category"]:
            stats["by_category"][category] = {"qa": 0, "mask": 0}
        
        if task["is_segmentation"]:
            stats["by_category"][category]["mask"] += 1
        else:
            stats["by_category"][category]["qa"] += 1
        
        dataset = task["dataset"]
        if dataset not in stats["by_dataset"]:
            stats["by_dataset"][dataset] = 0
        stats["by_dataset"][dataset] += 1
    
    return stats