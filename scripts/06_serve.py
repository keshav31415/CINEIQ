import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

import uvicorn
from cineiq.config import get_config

def main():
    cfg = get_config()
    uvicorn.run(
        "cineiq.api.app:app",
        host=cfg['api']['host'],
        port=cfg['api']['port'],
        reload=True,
    )

if __name__ == "__main__":
    main()
