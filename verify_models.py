"""
verify_models.py – Build-time model integrity check.

Run during Render build to ensure PyTorch model files are not corrupted
by Git LFS pointer substitution or CRLF line-ending conversion.

If a file is corrupted, it is reconstructed from the base64-encoded
model_b64.py (plain text, immune to git binary corruption).
"""

import os
import sys
import base64
import zipfile


MODEL_DIR = os.path.join("services", "shadow_portfolio", "saved_models")

MODEL_FILES = [
    {
        "filename": "ppo_shadow_portfolio.zip",
        "check": "zip",
    },
    {
        "filename": "vec_normalize.pkl",
        "check": "pkl",
    },
]


def is_lfs_pointer(filepath: str) -> bool:
    """Check if a file is a Git LFS pointer instead of actual content."""
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


def restore_from_b64(filename: str, dest: str) -> bool:
    """Restore a file from the base64-encoded model_b64.py."""
    b64_path = os.path.join(MODEL_DIR, "model_b64.py")
    if not os.path.exists(b64_path):
        print(f"  ERROR: {b64_path} not found — run encode_models.py locally first")
        return False

    try:
        # Import the model data module
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_b64", b64_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if filename not in mod.MODEL_DATA:
            print(f"  ERROR: '{filename}' not found in model_b64.py")
            return False

        raw = base64.b64decode(mod.MODEL_DATA[filename])
        with open(dest, "wb") as f:
            f.write(raw)

        print(f"  Restored from base64: {len(raw)} bytes")
        return True
    except Exception as e:
        print(f"  Restore FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("  Model Integrity Check")
    print("=" * 60)

    all_ok = True

    for entry in MODEL_FILES:
        filename = entry["filename"]
        check = entry["check"]
        filepath = os.path.join(MODEL_DIR, filename)

        print(f"\nChecking: {filepath}")

        needs_restore = False

        if not os.path.exists(filepath):
            print(f"  MISSING")
            needs_restore = True
        else:
            size = os.path.getsize(filepath)
            print(f"  Size: {size} bytes")

            if is_lfs_pointer(filepath):
                print(f"  CORRUPTED: Git LFS pointer detected")
                needs_restore = True
            elif check == "zip" and not is_valid_zip(filepath):
                print(f"  CORRUPTED: Not a valid zip archive")
                needs_restore = True
            else:
                print(f"  OK")

        if needs_restore:
            print(f"  Restoring from base64 backup...")
            if not restore_from_b64(filename, filepath):
                all_ok = False

    # Final verification
    print("\n" + "-" * 60)
    print("  Final verification:")
    for entry in MODEL_FILES:
        filepath = os.path.join(MODEL_DIR, entry["filename"])
        if entry["check"] == "zip":
            ok = is_valid_zip(filepath)
        else:
            ok = os.path.exists(filepath) and not is_lfs_pointer(filepath)
        status = "OK" if ok else "FAILED"
        print(f"  {entry['filename']}: {status}")

    print("=" * 60)
    if not all_ok:
        print("  WARNING: Some files could not be restored.")
        sys.exit(1)
    else:
        print("  All model files verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
