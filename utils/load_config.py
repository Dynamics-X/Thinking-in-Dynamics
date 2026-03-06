import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from colorama import Fore, init

init(autoreset=True)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"{Fore.RED}ERROR: config.yaml not found -> {config_path}")
        raise SystemExit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_download_path() -> Optional[str]:
    try:
        config = load_config()
        return config.get('model', {}).get('download_path', None)
    except SystemExit:
        return None
