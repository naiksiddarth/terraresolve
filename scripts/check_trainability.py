"""
Corrected trainability check: load via trainable_model(), not compiled_model().
compiled_model() explicitly freezes all params (lines 40-41 in load.py).
trainable_model() loads the same weights but does NOT freeze them.
"""
import sys, time, torch, pathlib
import safetensors.torch

sys.path.insert(0, '.')
from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel

WEIGHTS_DIR = pathlib.Path("models/pretrained/sen2sr_rgbn_x4")
DEVICE = "cuda"

print("=== Loading CNNSR via trainable_model path ===")
sr_model_weights = safetensors.torch.load_file(WEIGHTS_DIR / "model.safetensor")
cnnsr = CNNSR(4, 4, 24, 4, True, False, 6)
cnnsr.load_state_dict(sr_model_weights)
cnnsr = cnnsr.to(DEVICE)
# NOTE: NOT calling requires_grad_(False) — weights are trainable by default after load

print(f"CNNSR type: {type(cnnsr)}")
all_params = list(cnnsr.parameters())
grad_params = [p for p in all_params if p.requires_grad]
print(f"Total params: {sum(p.numel() for p in all_params):,}")
print(f"Trainable: {sum(p.numel() for p in grad_params):,}")

# Also load HardConstraint to understand if it blocks backprop
hc_weights = safetensors.torch.load_file(WEIGHTS_DIR / "hard_constraint.safetensor")
hard_constraint = HardConstraint(
    low_pass_mask=hc_weights["weights"].to(DEVICE), device=DEVICE
)
hc_params = list(hard_constraint.parameters())
print(f"HardConstraint params: {sum(p.numel() for p in hc_params)} (should be ~filter weights, not trained)")

# Test 1: CNNSR alone (no HardConstraint) — does backprop work?
print("\n=== Test 1: CNNSR forward+backward (without HardConstraint) ===")
cnnsr.train()
optimizer = torch.optim.Adam(cnnsr.parameters(), lr=1e-4)

dummy_lr = torch.randn(1, 4, 64, 64, device=DEVICE)
dummy_hr = torch.randn(1, 4, 256, 256, device=DEVICE)

optimizer.zero_grad()
out = cnnsr(dummy_lr)
print(f"Output shape: {out.shape}")
loss = torch.nn.functional.l1_loss(out, dummy_hr)
print(f"Loss: {loss.item():.6f}")
loss.backward()

grad_norms = []
for name, p in cnnsr.named_parameters():
    if p.grad is not None:
        grad_norms.append((name, p.grad.norm().item()))

print(f"\nParams with gradients: {len(grad_norms)}/{len(list(cnnsr.parameters()))}")
print("First 8 grad norms:")
for name, norm in grad_norms[:8]:
    print(f"  {name}: {norm:.6f}")
    
if all(n > 0 for _, n in grad_norms):
    print("\n✅ GRADIENTS FLOW CORRECTLY through CNNSR")
elif len(grad_norms) > 0:
    zero = sum(1 for _, n in grad_norms if n == 0)
    print(f"\n⚠️  {zero}/{len(grad_norms)} params have zero gradients")
else:
    print("\n❌ NO GRADIENTS")

# Test 2: Timing benchmark
print("\n=== Test 2: Training step timing ===")
BATCH = 4
N_WARMUP, N_TIMED = 3, 20

for _ in range(N_WARMUP):
    lr = torch.randn(BATCH, 4, 64, 64, device=DEVICE)
    hr = torch.randn(BATCH, 4, 256, 256, device=DEVICE)
    optimizer.zero_grad()
    loss = torch.nn.functional.l1_loss(cnnsr(lr), hr)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_TIMED):
    lr = torch.randn(BATCH, 4, 64, 64, device=DEVICE)
    hr = torch.randn(BATCH, 4, 256, 256, device=DEVICE)
    optimizer.zero_grad()
    loss = torch.nn.functional.l1_loss(cnnsr(lr), hr)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

step_ms = (elapsed / N_TIMED) * 1000
sps = BATCH / (elapsed / N_TIMED)
print(f"Step time: {step_ms:.1f}ms (batch={BATCH}) → {sps:.1f} samples/sec")

for n in [500, 1000, 5000]:
    steps = n // BATCH
    epoch_s = steps * (step_ms / 1000)
    epochs_2h = (7200) / epoch_s
    print(f"  {n} samples: {epoch_s:.1f}s/epoch → ~{int(epochs_2h)} epochs in 2h")
