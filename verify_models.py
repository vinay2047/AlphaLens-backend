"""
verify_models.py – Build-time model integrity check.

Run during Render build to ensure PyTorch model files are not corrupted
by Git LFS pointer substitution or CRLF line-ending conversion.

If a file is corrupted, it is re-downloaded from the GitHub raw URL.
"""

import os
import sys
import zipfile
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/vinay2047/AlphaLens-backend/main"

MODEL_FILES = [
    {
        "path": "services/shadow_portfolio/saved_models/ppo_shadow_portfolio.zip",
        "check": "zip",  # validate as a zip archive
    },
    {
        "path": "services/shadow_portfolio/saved_models/vec_normalize.pkl",
        "check": "pkl",  # just check it's not a git-lfs pointer
    },
]


def is_lfs_pointer(filepath: str) -> bool:
    """Check if a file is a Git LFS pointer (tiny text file) instead of actual content."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(20)
        return header.startswith(b"version https://git-lfs")
    except Exception:
        return False


def is_valid_zip(filepath: str) -> bool:
    """Check if a file is a valid zip archive."""
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.testzip()
        return True
    except Exception:
        return False


def download_file(url: str, dest: str) -> bool:
    """Download a file from a URL."""
    print(f"  Downloading from: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Downloaded: {os.path.getsize(dest)} bytes")
        return True
    except Exception as e:
        print(f"  Download FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("  Model Integrity Check")
    print("=" * 60)

    all_ok = True

    for entry in MODEL_FILES:
        path = entry["path"]
        check = entry["check"]
        print(f"\nChecking: {path}")

        if not os.path.exists(path):
            print(f"  MISSING — will download")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"{REPO_RAW}/{path}"
            if not download_file(url, path):
                all_ok = False
            continue

        size = os.path.getsize(path)
        print(f"  Size: {size} bytes")

        # Check for Git LFS pointer files (typically ~130 bytes of text)
        if is_lfs_pointer(path):
            print(f"  CORRUPTED: Git LFS pointer detected (not actual content)")
            url = f"{REPO_RAW}/{path}"
            if not download_file(url, path):
                all_ok = False
            continue

        # Validate zip files
        if check == "zip" and not is_valid_zip(path):
            print(f"  CORRUPTED: Not a valid zip archive")
            url = f"{REPO_RAW}/{path}"
            if not download_file(url, path):
                all_ok = False
            continue

        print(f"  OK")

    print("\n" + "=" * 60)
    if all_ok:
        print("  All model files verified successfully.")
    else:
        print("  WARNING: Some files could not be fixed.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
