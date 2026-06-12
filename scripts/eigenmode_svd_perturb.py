"""SVD direction perturbation experiment.

Tests whether dominant singular vectors of GAN weights encode
physically meaningful structural features.

For each of the top-k singular vectors, perturb W → W + α·u
and observe how generated topology changes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = "checkpoints/quickstart/pretrained_generator.pt"
OUT_DIR = "results/eigenmode"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load model ─────────────────────────────────────────────────
from main.model.topologygan import TopologyGANGenerator

state = torch.load(PRETRAINED, map_location=device, weights_only=False)
if "generator_state_dict" in state:
    state = state["generator_state_dict"]
gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                            variant="unet", height=64, width=128).to(device)
gen.load_state_dict(state, strict=False)
gen.eval()

# ── fixed input ────────────────────────────────────────────────
from main.utils.data_loader import create_dataloaders
_, _, test_loader = create_dataloaders(
    "data/synthetic_train.npy", height=64, width=128, batch_size=16, num_workers=0)
batch = next(iter(test_loader))
conditions = batch[0][:4].to(device)
nz = gen.nz
torch.manual_seed(42)
z = torch.randn(4, nz, device=device)

# ── baseline generation ────────────────────────────────────────
with torch.no_grad():
    baseline = gen(conditions, z=z).cpu()

# ── find conv layers ───────────────────────────────────────────
conv_layers = {}
for name, module in gen.named_modules():
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        conv_layers[name] = module

target_layers = [n for n in conv_layers if any(k in n for k in ['d2', 'd3', 'e5'])]
print(f"Target layers: {target_layers}")

# ── SVD perturbation ───────────────────────────────────────────
N_VECTORS = 3
ALPHAS = [-0.3, -0.1, 0.0, 0.1, 0.3]
N_LAYERS = len(target_layers)
N_COLS = len(ALPHAS) + 1  # +1 for baseline

results = {}

for layer_name in target_layers:
    module = conv_layers[layer_name]
    W = module.weight.data.clone()  # [out, in, kh, kw]
    W_flat = W.view(W.size(0), -1).cpu().numpy()  # [out, in*kh*kw]

    # SVD
    U, S, Vt = np.linalg.svd(W_flat, full_matrices=False)

    print(f"\n{layer_name}: shape={list(W.shape)}, S range=[{S[0]:.0f},{S[-1]:.0f}]")
    print(f"  Top-5 singular values: {S[:5]}")
    print(f"  Energy ratio: top1={S[0]**2/np.sum(S**2):.3f}, top3={np.sum(S[:3]**2)/np.sum(S**2):.3f}")

    fig, axes = plt.subplots(4, N_COLS, figsize=(N_COLS * 3.2, 10))

    for row in range(4):
        # Baseline column
        axes[row, 0].imshow(baseline[row, 0].numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
        if row == 0: axes[row, 0].set_title('Baseline', fontsize=9)
        axes[row, 0].axis('off')

    for vec_idx in range(N_VECTORS):
        u = U[:, vec_idx]  # left singular vector [out_dim]
        u_tensor = torch.tensor(u, dtype=torch.float32, device=device)

        # Build perturbation: W' = W + α·u ⊗ 1
        for col, alpha in enumerate(ALPHAS):
            if alpha == 0.0:
                # skip, same as baseline
                for row in range(4):
                    axes[row, col+1].axis('off')
                continue

            # Apply perturbation
            delta = alpha * u_tensor.view(-1, 1, 1, 1).expand_as(W)
            module.weight.data = W + delta

            with torch.no_grad():
                perturbed = gen(conditions, z=z).cpu()

            col_idx = col + 1
            for row in range(4):
                ax = axes[row, col_idx]
                diff = (perturbed[row, 0] - baseline[row, 0]).abs().mean().item()
                ax.imshow(perturbed[row, 0].numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
                if row == 0:
                    ax.set_title(f'α={alpha:+0.1f}\n|Δ|={diff:.4f}', fontsize=8)
                ax.axis('off')

        # Restore original weight
        module.weight.data = W

    # Column labels for first vector only
    axes[0, 0].set_ylabel(f'u₁ (σ₁={S[0]:.0f})', fontsize=10, fontweight='bold')
    if N_VECTORS > 1:
        pass  # labels are in the subplot titles

    plt.suptitle(f'{layer_name}: SVD Direction Perturbation', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f"{OUT_DIR}/perturb_{layer_name.replace('.','_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

    results[layer_name] = {
        'singular_values': S.tolist(),
        'energy_top1': float(S[0]**2 / np.sum(S**2)),
        'energy_top3': float(np.sum(S[:3]**2) / np.sum(S**2)),
    }

# ── summary ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SVD Energy Concentration by Layer")
print(f"{'='*60}")
print(f"{'Layer':<25} {'Top-1 Energy':>14} {'Top-3 Energy':>14}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<25} {r['energy_top1']:>14.3f} {r['energy_top3']:>14.3f}")

import json
with open(f"{OUT_DIR}/svd_energy.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT_DIR}/svd_energy.json")
print("Done: SVD perturbation experiment")
