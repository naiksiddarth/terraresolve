"""
Benchmark: how long does one training step + one epoch estimate take?
Run BEFORE downloading any data.
"""
import sys, time, torch
sys.path.insert(0, '.')

import terraresolve.models  # noqa
from terraresolve.registry import MODEL_REGISTRY

print("=== Timing benchmark on RTX 5050 ===")
model = MODEL_REGISTRY.build("sen2sr", weights_dir="models/pretrained/sen2sr_rgbn_x4", device="cuda")

inner = None
for attr in ['_inner', 'model', 'net', 'backbone']:
    if hasattr(model, attr):
        inner = getattr(model, attr)
        break
if inner is None:
    inner = model

# Unfreeze all for benchmark
for p in inner.parameters():
    p.requires_grad_(True)

inner.train()
optimizer = torch.optim.Adam(inner.parameters(), lr=1e-4)

# Simulate one training step (LR 64x64 -> HR 256x256)
BATCH_SIZE = 4
LR_PATCH = 64
HR_PATCH = 256
N_WARMUP = 3
N_TIMED = 20

print(f"Batch size: {BATCH_SIZE}, LR patch: {LR_PATCH}x{LR_PATCH}")
print(f"Warming up ({N_WARMUP} steps)...")

for i in range(N_WARMUP):
    lr = torch.randn(BATCH_SIZE, 4, LR_PATCH, LR_PATCH, device='cuda')
    hr = torch.randn(BATCH_SIZE, 4, HR_PATCH, HR_PATCH, device='cuda')
    optimizer.zero_grad()
    out = inner(lr)
    loss = torch.nn.functional.l1_loss(out, hr)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
print(f"Timing {N_TIMED} steps...")
t0 = time.perf_counter()

for i in range(N_TIMED):
    lr = torch.randn(BATCH_SIZE, 4, LR_PATCH, LR_PATCH, device='cuda')
    hr = torch.randn(BATCH_SIZE, 4, HR_PATCH, HR_PATCH, device='cuda')
    optimizer.zero_grad()
    out = inner(lr)
    loss = torch.nn.functional.l1_loss(out, hr)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

step_time = elapsed / N_TIMED
samples_per_sec = BATCH_SIZE / step_time

print(f"\n=== Results ===")
print(f"Avg step time: {step_time*1000:.1f}ms per step (batch={BATCH_SIZE})")
print(f"Throughput: {samples_per_sec:.1f} samples/sec")
print(f"\n=== Epoch estimates ===")
for n_samples in [500, 1000, 5000]:
    steps = n_samples // BATCH_SIZE
    epoch_sec = steps * step_time
    print(f"  {n_samples} samples ({steps} steps): {epoch_sec:.1f}s = {epoch_sec/60:.1f} min/epoch")

print(f"\n=== Realistic epochs in 2h with 500 samples ===")
steps_500 = 500 // BATCH_SIZE
epoch_500_sec = steps_500 * step_time
epochs_in_2h = (2 * 3600) / epoch_500_sec
print(f"  {epoch_500_sec:.1f}s per epoch → ~{int(epochs_in_2h)} epochs in 2 hours")
