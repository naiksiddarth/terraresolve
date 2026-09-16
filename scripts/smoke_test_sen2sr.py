import sys, torch, numpy as np
sys.path.insert(0, '.')

import terraresolve.models  # noqa: F401 — side-effect: registers all models via decorators
from terraresolve.registry import MODEL_REGISTRY
from terraresolve.engine.inference import run_inference

model = MODEL_REGISTRY.build(
    'sen2sr',
    weights_dir='models/pretrained/sen2sr_rgbn_x4',
    device='cuda'
)
print("Registered models:", MODEL_REGISTRY.names())
print("Model type:", type(model).__name__)
print("USES_INTERNAL_TILING:", model.USES_INTERNAL_TILING)
print()

all_ok = True
for h, w in [(64, 64), (128, 128), (256, 200)]:
    scene = (np.random.rand(4, h, w) * 0.1).astype(np.float32)
    sr = run_inference(model, scene, scale=4, device='cuda')
    expected = (4, h * 4, w * 4)
    ok = sr.shape == expected
    all_ok = all_ok and ok
    status = "PASS" if ok else f"FAIL (expected {expected})"
    print(f"  Input ({h}x{w}) -> SR {sr.shape}  [{status}]")

print()
print("=== ALL PASSED ===" if all_ok else "=== FAILURES DETECTED ===")
