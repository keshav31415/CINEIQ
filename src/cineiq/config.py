import yaml
from pathlib import Path

def get_config():
    config_path = Path("config.yaml")
    if not config_path.exists():
        # Fallback for when running from a different directory
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
