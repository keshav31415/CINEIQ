"""Launch the CINEIQ Streamlit dashboard.

Run: uv run python scripts/07_dashboard.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard = Path(__file__).parent.parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard)],
        check=True,
    )
