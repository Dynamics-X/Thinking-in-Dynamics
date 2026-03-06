import os
import argparse
import subprocess
from pathlib import Path
from colorama import Fore, init

from utils.load_model import get_model_type
from utils.load_config import get_download_path

init(autoreset=True)


def download_model(model_name: str, local_dir: str):
    model_type = get_model_type(model_name)

    os.makedirs(local_dir, exist_ok=True)
    
    cmd = [
        "hf",
        "download",
        model_name,
        "--local-dir", local_dir
    ]
    
    try:
        print(f"{Fore.CYAN}> {' '.join(cmd)}")
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        print(f"{Fore.GREEN}{model_name} downloaded successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}ERROR: Fail to download {model_name}")
        if e.stderr:
            try:
                print(f"{Fore.RED}{e.stderr}")
            except:
                print(f"{Fore.RED}ERROR: Output contains undecodable characters")
        raise SystemExit(1)
    except FileNotFoundError:
        print(f"{Fore.RED}ERROR: hf command not found")
        raise SystemExit(1)


def load_model_list():
    model_list_path = Path(__file__).parent / "conf" / "model_list.txt"
    
    if not model_list_path.exists():
        print(f"{Fore.RED}ERROR: model_list.txt not found -> {model_list_path}")
        raise SystemExit(1)
    
    models = []
    with open(model_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                models.append(line)
    
    if not models:
        print(f"{Fore.RED}ERROR: No model found in model_list.txt")
        raise SystemExit(1)
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models to local directory",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Single model name on HuggingFace'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch download models from conf/model_list.txt'
    )
    
    args = parser.parse_args()
    
    if args.batch and args.model:
        print(f"{Fore.RED}ERROR: Cannot use --model and --batch together")
        raise SystemExit(1)
    
    if not args.batch and not args.model:
        print(f"{Fore.RED}ERROR: Please specify --model or --batch")
        raise SystemExit(1)
    
    if args.batch:
        models = load_model_list()
    else:
        models = [args.model]
    
    download_path = get_download_path()
    if not download_path:
        print(f"{Fore.RED}ERROR: <download_path> not specified in config.yaml")
        raise SystemExit(1)
    
    print(f"Download directory: {download_path}")
    print(f"Models to download: {len(models)}")
    print()
    
    for i, model_name in enumerate(models, 1):
        print(f"{'='*60}")
        print(f"[{i}/{len(models)}] Processing: {model_name}")
        print(f"{'='*60}")
        
        model_dir_name = model_name.replace('/', '--')
        local_dir = os.path.join(download_path, model_dir_name)
        
        try:
            download_model(model_name, local_dir)
            print()
        except SystemExit:
            print(f"{Fore.YELLOW}WARN: Skip {model_name}")
            print()
            continue
    
    print(f"{Fore.GREEN}All models downloaded successfully")



if __name__ == "__main__":
    main()
