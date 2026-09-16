import os

MAX_UPLOAD_MB = 25
MAX_PIXELS = 4096 * 4096
MIN_DIMENSION = 128
MIN_BANDS = 3
PATCH_SIZE = 128
OVERLAP = 32
SCALE = 4

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "models/pretrained/sen2sr_rgbn_x4")
DATA_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH = os.environ.get("DB_PATH", "terraresolve.db")
