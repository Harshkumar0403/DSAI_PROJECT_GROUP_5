"""
download_dataset.py
────────────────────────────────────────────────────────────────────────────────
Downloads the Forest Loss Detection dataset from Kaggle.

Dataset : suranjandas1990/forest-loss-dataset
Size    : ~3–5 GB (7,500 patches × 3 splits × 6 channels)

USAGE
─────
    python download_dataset.py

PREREQUISITES
─────────────
1. Install kagglehub:
       pip install kagglehub

2. Place your kaggle.json API token in one of:
       Linux/Mac : ~/.kaggle/kaggle.json
       Windows   : C:\\Users\\<username>\\.kaggle\\kaggle.json

   To get kaggle.json:
       → Go to https://www.kaggle.com/settings
       → Scroll to "API" section
       → Click "Create New Token"
       → Move the downloaded kaggle.json to the path above

   Set permissions (Linux/Mac only):
       chmod 600 ~/.kaggle/kaggle.json

3. Accept the dataset license on Kaggle (if prompted):
       https://www.kaggle.com/datasets/suranjandas1990/forest-loss-dataset
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import shutil
import argparse


# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_HANDLE = "suranjandas1990/forest-loss-dataset"
DEFAULT_DEST   = os.path.join(os.path.dirname(__file__), "dataset")

EXPECTED_REGIONS = [
    "Meghalaya_2021_2023",
    "Nagaland_2021_2023"
]
EXPECTED_SUBDIRS = ["t1", "t2", "label"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_kagglehub():
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("[ERROR] kagglehub is not installed.")
        print("        Run: pip install kagglehub")
        sys.exit(1)


def check_credentials():
    kaggle_json_paths = [
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json"),
    ]
    # Also accept env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[INFO] Using KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return True

    for path in kaggle_json_paths:
        if os.path.isfile(path):
            print(f"[INFO] Found Kaggle credentials at: {path}")
            # Warn if permissions are too open (Linux/Mac)
            if sys.platform != "win32":
                mode = oct(os.stat(path).st_mode)[-3:]
                if mode != "600":
                    print(f"[WARN] kaggle.json permissions are {mode}. "
                          f"Recommended: 600")
                    print("       Run: chmod 600 ~/.kaggle/kaggle.json")
            return True

    print("[ERROR] Kaggle credentials not found.")
    print("        Option 1: Place kaggle.json at ~/.kaggle/kaggle.json")
    print("        Option 2: Set environment variables:")
    print("                  export KAGGLE_USERNAME=your_username")
    print("                  export KAGGLE_KEY=your_api_key")
    sys.exit(1)


def download_with_kagglehub(dest_dir):
    kagglehub = check_kagglehub()
    check_credentials()

    print(f"\n[INFO] Downloading dataset: {DATASET_HANDLE}")
    print(f"[INFO] This may take several minutes depending on your connection.\n")

    try:
        # Download to kagglehub cache first
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"\n[INFO] Downloaded to cache: {path}")

        # Copy to project dataset directory
        if os.path.abspath(path) != os.path.abspath(dest_dir):
            print(f"[INFO] Copying to project directory: {dest_dir}")
            if os.path.exists(dest_dir):
                print(f"[WARN] Destination already exists. Overwriting: {dest_dir}")
                shutil.rmtree(dest_dir)
            shutil.copytree(path, dest_dir)
            print(f"[INFO] Dataset copied to: {dest_dir}")
        else:
            print(f"[INFO] Dataset already at destination: {dest_dir}")

        return dest_dir

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\n[INFO] Trying fallback: kaggle CLI...")
        return download_with_cli(dest_dir)


def download_with_cli(dest_dir):
    """Fallback: use kaggle CLI if kagglehub fails."""
    if shutil.which("kaggle") is None:
        print("[ERROR] kaggle CLI not found. Install it with: pip install kaggle")
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)
    cmd = (f"kaggle datasets download "
           f"-d {DATASET_HANDLE} "
           f"-p {dest_dir} "
           f"--unzip")

    print(f"[INFO] Running: {cmd}")
    ret = os.system(cmd)

    if ret != 0:
        print("[ERROR] kaggle CLI download failed.")
        print("[INFO]  Check that you have accepted the dataset license at:")
        print(f"        https://www.kaggle.com/datasets/{DATASET_HANDLE}")
        sys.exit(1)

    return dest_dir


def verify_dataset(dest_dir):
    """Check that expected directory structure is present."""
    print(f"\n[INFO] Verifying dataset structure at: {dest_dir}")
    all_ok    = True
    total_files = 0

    for region in EXPECTED_REGIONS:
        for subdir in EXPECTED_SUBDIRS:
            path = os.path.join(dest_dir, region, subdir)
            if not os.path.isdir(path):
                print(f"  [MISSING]  {region}/{subdir}/")
                all_ok = False
            else:
                n_files = len([
                    f for f in os.listdir(path)
                    if f.endswith(".npy")
                ])
                total_files += n_files
                print(f"  [OK]  {region}/{subdir}/  ({n_files} patches)")

    print(f"\n[INFO] Total .npy files found: {total_files}")

    if total_files == 0:
        print("[WARN] No .npy files found. The dataset may not have extracted correctly.")
        all_ok = False

    if all_ok:
        print("[SUCCESS] Dataset structure verified successfully.\n")
    else:
        print("[WARN] Some directories are missing. "
              "The download may be incomplete.\n")

    return all_ok


def print_usage_example(dest_dir):
    print("─" * 60)
    print("  NEXT STEPS — use this path in your notebook:")
    print("─" * 60)
    print(f"""
  root_dir = "{os.path.abspath(dest_dir)}"

  # Example usage
  import numpy as np
  import os

  t1    = np.load(os.path.join(root_dir,
              "Meghalaya_2021_2023/t1/<filename>.npy"))
  t2    = np.load(os.path.join(root_dir,
              "Meghalaya_2021_2023/t2/<filename>.npy"))
  label = np.load(os.path.join(root_dir,
              "Meghalaya_2021_2023/label/<filename>.npy"))

  print("T1 shape   :", t1.shape)    # (6, 256, 256)
  print("T2 shape   :", t2.shape)    # (6, 256, 256)
  print("Label shape:", label.shape) # (256, 256)
""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download the Forest Loss Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--dest", type=str, default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Skip download and only verify existing dataset structure"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Forest Loss Detection — Dataset Downloader")
    print("=" * 60)
    print(f"  Dataset : {DATASET_HANDLE}")
    print(f"  Dest    : {os.path.abspath(args.dest)}")
    print("=" * 60)

    if args.verify_only:
        verify_dataset(args.dest)
        return

    # Download
    dest = download_with_kagglehub(args.dest)

    # Verify
    verify_dataset(dest)

    # Usage hint
    print_usage_example(dest)


if __name__ == "__main__":
    main()
