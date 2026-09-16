"""Download SEN2SR pretrained weights from HuggingFace via mlstac.

Run once before inference:
    python scripts/download_sen2sr_weights.py

Weights are saved to models/pretrained/sen2sr_rgbn_x4/ (gitignored).
Re-running is idempotent -- mlstac skips files that already exist.
"""
import pathlib
import sys

try:
    import mlstac
except ImportError:
    sys.exit("mlstac not installed. Run: pip install mlstac")

WEIGHTS_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
    "/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)
OUTPUT_DIR = pathlib.Path("models/pretrained/sen2sr_rgbn_x4")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Downloading SEN2SRLite weights -> {OUTPUT_DIR.resolve()} ...")
mlstac.download(file=WEIGHTS_URL, output_dir=str(OUTPUT_DIR))

files = sorted(OUTPUT_DIR.iterdir())
print(f"Done. {len(files)} files:")
for f in files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:40s}  {size_kb:8.1f} KB")
