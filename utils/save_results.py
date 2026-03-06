import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from colorama import Fore, init
init(autoreset=True)


def save_results(results: List[Dict], config: Dict, summary: Dict, task_type: str):
    results_dir = Path(config['result_path'])
    model_name = config['model']['model_name']
    
    dir_name = model_name.split('/')[-1]
    model_results_dir = results_dir / dir_name
    
    qa_details_dir = model_results_dir / "qa_details"
    mask_details_dir = model_results_dir / "mask_details"
    
    if task_type in ['qa', 'all']:
        qa_details_dir.mkdir(parents=True, exist_ok=True)
    if task_type in ['mask', 'all']:
        mask_details_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSave results to {model_results_dir}")
    
    qa_results_by_scene = defaultdict(list)
    mask_results_by_scene = defaultdict(list)
    
    for result in results:
        dataset = result['dataset']
        scene_name = result['scene_name']
        task_suffix = result['task_type']
        
        if result['is_segmentation']:
            mask_results_by_scene[(dataset, scene_name, task_suffix)].append(result)
        else:
            qa_results_by_scene[(dataset, scene_name, task_suffix)].append(result)
    
    if task_type in ['qa', 'all']:
        for (dataset, scene_name, task_suffix), task_results in qa_results_by_scene.items():
            scene_dir = qa_details_dir / dataset
            scene_dir.mkdir(parents=True, exist_ok=True)
            
            details_file = scene_dir / f"{scene_name}_{task_suffix}_details.json"
            
            qa_details = []
            for r in task_results:
                qa_details.append({
                    "question": r['question'],
                    "options": r['options'],
                    "answer": r['answer'],
                    "model_predict": r.get('prediction', ''),
                    "accuracy": r.get('accuracy', 0.0)
                })
            
            with open(details_file, 'w', encoding='utf-8') as f:
                json.dump(qa_details, f, indent=2, ensure_ascii=False)
        
        print(f"  QA details saved to {qa_details_dir}")
    
    if task_type in ['mask', 'all']:
        print(f"  Mask details saved to {mask_details_dir}")
    
    overall_file = model_results_dir / "overall.txt"
    
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"  Model: {model_name}\n")
        f.write(f"  Task: {task_type}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for category in ["Camera-Object", "Inter-Object", "Object-Scene"]:
            f.write(f"{category}:\n")
            
            qa_key = f"{category}_QA_Accuracy"
            mask_key = f"{category}_Mask_J&F"
            
            if qa_key in summary:
                f.write(f"  QA Accuracy: {summary[qa_key]:.2f}%\n")
            
            if mask_key in summary:
                f.write(f"  Mask J&F: {summary[mask_key]:.2f}%\n")
            
            f.write("\n")
        
        if "Overall" in summary:
            f.write("=" * 60 + "\n")
            f.write(f"  Overall Score: {summary['Overall']:.2f}%\n")
            f.write("=" * 60 + "\n")
    
    print(f"  Overall results saved to: {overall_file}")