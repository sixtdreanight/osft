"""Generate topology comparison visualizations for paper."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from main.model.topologygan import TopologyGANGenerator
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "results/viz"
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_, _, test_loader = create_dataloaders(
    "data/synthetic_train.npy", height=64, width=128, batch_size=16, num_workers=0)

def load_gen(ckpt_path, variant="unet", gf_dim=64):
    gen = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=gf_dim,
        variant=variant, height=64, width=128,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen.load_state_dict(state, strict=False)
    gen.eval()
    return gen

# Load models
models = {
    "Pre-trained": load_gen("checkpoints/quickstart/pretrained_generator.pt"),
    "Full FT": load_gen("results/final/full_ft_Cantilever_S0/full_ft_best.pt"),
    "Adapter": load_gen("results/final/adapter_Cantilever_S0/full_ft_best.pt"),
    "OSFT": load_gen("results/final/osft_Cantilever_S0/osft_best.pt"),
}

# Get test samples
batch = next(iter(test_loader))
real_A = batch[0][:4].to(device)
real_B = batch[1][:4].to(device)

# === Figure 1: Topology comparison grid ===
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.2)

for i in range(4):
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(real_A[i, 0].cpu(), cmap="viridis", origin="lower")
    ax.set_ylabel(f"Sample {i+1}" if i == 0 else "")
    if i == 0: ax.set_title("Input (VF)")

    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(real_B[i, 0].cpu(), cmap="gray", vmin=0, vmax=1, origin="lower")
    if i == 0: ax.set_title("Ground Truth")

    for j, (name, model) in enumerate(models.items()):
        ax = fig.add_subplot(gs[i, j+2])
        with torch.no_grad():
            fake = model(real_A[i:i+1])[0, 0].cpu()
        ax.imshow(fake, cmap="gray", vmin=0, vmax=1, origin="lower")
        if i == 0: ax.set_title(name)

for ax in fig.axes:
    ax.set_xticks([]); ax.set_yticks([])

fig.savefig(f"{OUT}/topology_comparison.png")
plt.close()
print("Saved topology_comparison.png")

# === Figure 2: Error maps ===
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for i in range(4):
    gt = real_B[i, 0].cpu().numpy()
    for j, (name, model) in enumerate(models.items()):
        with torch.no_grad():
            fake = model(real_A[i:i+1])[0, 0].cpu().numpy()
        error = np.abs(fake - gt)
        im = axes[i, j].imshow(error, cmap="hot", vmin=0, vmax=1, origin="lower")
        axes[i, j].set_title(f"{name}\nMSE={error.mean():.4f}" if i == 0 else f"MSE={error.mean():.4f}")
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

plt.colorbar(im, ax=axes, label="|Error|", fraction=0.02)
fig.savefig(f"{OUT}/error_maps.png")
plt.close()
print("Saved error_maps.png")

# === Figure 3: Training curve comparison ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MSE comparison bar chart
methods = list(models.keys())
mse_vals = [0.2481, 0.2027, 0.1981, 0.1822]  # from E1
colors = ["#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]
bars = axes[0].bar(methods, mse_vals, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, mse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_ylabel("MSE")
axes[0].set_title("Image Quality (MSE, lower is better)")
axes[0].grid(axis="y", alpha=0.3)

# Improvement over pre-trained
improvements = [(0.2481 - v) / 0.2481 * 100 for v in mse_vals]
colors2 = ["#95a5a6", "#e74c3c", "#f39c12", "#2ecc71"]
bars = axes[1].bar(methods, improvements, color=colors2, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, improvements):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[1].set_ylabel("MSE Improvement (%)")
axes[1].set_title("Improvement over Pre-trained")
axes[1].grid(axis="y", alpha=0.3)

fig.savefig(f"{OUT}/mse_comparison.png")
plt.close()
print("Saved mse_comparison.png")

print(f"\nAll figures saved to {OUT}/")
