from typing import List, Dict
from pathlib import Path
import numpy as np
from PIL import Image
from colorama import Fore, init
init(autoreset=True)


def fuzzy_matching(pred: str) -> str:
    pred = pred.strip()
    if not pred:
        return ""
    
    first_char = pred.split()[0] if pred.split() else pred[0]
    
    return first_char.rstrip('.,;:!?').upper()


def compute_qa_accuracy(pred: str, gt: str) -> float:
    pred_letter = fuzzy_matching(pred)
    gt_letter = gt.strip().upper()
    
    return 1.0 if pred_letter == gt_letter else 0.0


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: int = 2) -> float:
    try:
        from scipy import ndimage
        
        pred_boundary = pred_mask ^ ndimage.binary_erosion(pred_mask, iterations=bound_th)
        gt_boundary = gt_mask ^ ndimage.binary_erosion(gt_mask, iterations=bound_th)
        
        if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
            return 1.0
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return 0.0
        
        pred_dilated = ndimage.binary_dilation(pred_boundary, iterations=bound_th)
        gt_dilated = ndimage.binary_dilation(gt_boundary, iterations=bound_th)
        
        precision = np.logical_and(pred_boundary, gt_dilated).sum() / pred_boundary.sum()
        recall = np.logical_and(gt_boundary, pred_dilated).sum() / gt_boundary.sum()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
        
    except Exception as e:
        print(f"{Fore.YELLOW}WARN: Can not compute boundary F1")
        print(f"  Details: {e}")
        return 0.0


def load_gt_masks(mask_dir: str, object_id: str, n_frames: int) -> List[np.ndarray]:
    masks = []
    
    obj_num = object_id.split("_")[-1]
    obj_suffix = f"obj{obj_num}"
    
    mask_dir = Path(mask_dir)
    
    for i in range(n_frames):
        possible_names = [
            f"frame_{i:04d}_{obj_suffix}.png",
            f"frame_{i:04d}_{obj_suffix}.jpg",
        ]
        
        mask_path = None
        for name in possible_names:
            path = mask_dir / name
            if path.exists():
                mask_path = path
                break
        
        if mask_path:
            try:
                mask = np.array(Image.open(mask_path).convert('L'))
                masks.append(mask)
            except Exception as e:
                print(f"{Fore.YELLOW}WARN: Can not load mask -> {mask_path}")
                print(f"  Details: {e}")
                if masks:
                    masks.append(np.zeros_like(masks[-1]))
                else:
                    masks.append(np.zeros((480, 640), dtype=np.uint8))
        else:
            if masks:
                masks.append(np.zeros_like(masks[-1]))
            else:
                masks.append(np.zeros((480, 640), dtype=np.uint8))
    
    return masks


def compute_jf_score(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                     bound_th: int = 2) -> Dict[str, float]:
    if not pred_masks or not gt_masks:
        return {"J": 0.0, "F": 0.0, "J&F": 0.0}
    
    j_scores = []
    f_scores = []
    
    n_frames = min(len(pred_masks), len(gt_masks))
    
    for i in range(n_frames):
        pred = pred_masks[i] if i < len(pred_masks) else np.zeros_like(gt_masks[0])
        gt = gt_masks[i]
        
        if pred.ndim > 2:
            pred = pred.squeeze()
        if gt.ndim > 2:
            gt = gt.squeeze()
        
        pred_binary = (pred > 0.5).astype(bool)
        gt_binary = (gt > 0).astype(bool)
        
        if pred_binary.shape != gt_binary.shape:
            pred_pil = Image.fromarray(pred_binary.astype(np.uint8) * 255)
            pred_pil = pred_pil.resize((gt_binary.shape[1], gt_binary.shape[0]), Image.NEAREST)
            pred_binary = np.array(pred_pil) > 127
        
        j_scores.append(compute_iou(pred_binary, gt_binary))
        f_scores.append(compute_boundary_f1(pred_binary, gt_binary, bound_th))
    
    j_mean = np.mean(j_scores) if j_scores else 0.0
    f_mean = np.mean(f_scores) if f_scores else 0.0
    
    return {
        "J": j_mean,
        "F": f_mean,
        "J&F": (j_mean + f_mean) / 2
    }



def compute_category_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    from collections import defaultdict
    
    category_results = defaultdict(lambda: {"qa": [], "mask": []})
    
    for result in results:
        category = result.get("category", "Unknown")
        
        if result.get("is_segmentation"):
            jf_score = result.get("J&F", 0.0)
            category_results[category]["mask"].append(jf_score)
        else:
            accuracy = result.get("accuracy", 0.0)
            category_results[category]["qa"].append(accuracy)
    
    metrics = {}
    for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
        qa_scores = category_results[category]["qa"]
        mask_scores = category_results[category]["mask"]
        
        metrics[category] = {}
        
        if qa_scores:
            metrics[category]["QA_Accuracy"] = np.mean(qa_scores) * 100
            metrics[category]["QA_Count"] = len(qa_scores)
        
        if mask_scores:
            metrics[category]["Mask_J&F"] = np.mean(mask_scores) * 100
            metrics[category]["Mask_Count"] = len(mask_scores)
    
    return metrics
