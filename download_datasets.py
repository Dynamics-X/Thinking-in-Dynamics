from pathlib import Path
from huggingface_hub import snapshot_download
from colorama import Fore, init

from utils.load_config import load_config

init(autoreset=True)


def download_dataset(repo_name: str, local_dir: Path):
    print(f"Download datasets from {repo_name}")
    print(f"Datasets save to {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="dataset",
            local_dir=str(local_dir),
            ignore_patterns=[".gitattributes"]
        )
        print(f"\n{Fore.GREEN}Datasets downloaded successfully")
        return True
    except Exception as e:
        print(f"\n{Fore.RED}ERROR: Failed to download datasets")
        print(f"  Details: {e}")
        return False


def main():
    config = load_config()
    
    repo_name = config['datasets']['repo_name']
    datasets_path = Path(config['datasets']['datasets_path'])
    
    if download_dataset(repo_name, datasets_path):
        pass
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
