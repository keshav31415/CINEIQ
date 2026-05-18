"""
Download pre-trained CINEIQ artifacts so the project runs without
re-executing the training pipeline (scripts 01-05).

Usage:
    python scripts/00_download_artifacts.py

After the host uploads share.zip to a GitHub Release, update ARTIFACTS_URL
below with the actual download URL.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── Set this to your GitHub Release download URL after uploading share.zip ────
ARTIFACTS_URL = (
    "https://github.com/keshav31415/cineiq/releases/latest/download/share.zip"
)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent


def _progress(count, block_size, total_size):
    if total_size <= 0:
        print(f"\r  Downloaded {count * block_size // 1_048_576} MB...", end="", flush=True)
        return
    pct = min(100, count * block_size * 100 // total_size)
    filled = pct // 5
    bar = "#" * filled + "-" * (20 - filled)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


def download(url: str, dest: Path):
    print(f"Downloading artifacts from:\n  {url}\n")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Upload share.zip to a GitHub Release and set ARTIFACTS_URL in this script.")
        sys.exit(1)
    print()  # newline after progress bar


def extract(zip_path: Path):
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue

            # Map zip paths to project paths:
            #   share/models/<file>  -->  models/<file>
            #   share/cineiq.db      -->  data/processed/cineiq.db
            if name.startswith("share/models/"):
                rel = name[len("share/models/"):]
                target = ROOT / "models" / rel
            elif name in ("share/cineiq.db", "share\\cineiq.db"):
                target = ROOT / "data" / "processed" / "cineiq.db"
            else:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  extracted  {target.relative_to(ROOT)}")

    zip_path.unlink()
    print("\nDone. All artifacts are in place.")
    print("Start the server:      python run_api.py")
    print("Start the dashboard:   streamlit run dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained CINEIQ artifacts")
    parser.add_argument("--url", default=ARTIFACTS_URL, help="Direct URL to share.zip")
    args = parser.parse_args()

    zip_path = ROOT / "share.zip"

    # Skip download if already present (e.g. manual copy)
    if zip_path.exists():
        print("share.zip found locally, skipping download.")
    else:
        download(args.url, zip_path)

    extract(zip_path)


if __name__ == "__main__":
    main()
