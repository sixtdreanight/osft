"""Physics Latent Probing: Can z-dimensions control physical properties?

Hypothesis: Certain latent dimensions naturally encode physical parameters
(VF, compliance). If true, we can build a Physics-Conditioned Generator that
explicitly separates physics control from detail generation — achieving the
same goal as OSFT (protected physics + adaptable details) without SVD.

Experiment:
1. Sample z vectors from the physics-pretrained GAN
2. Generate topologies
3. Train a probe MLP: z → physical properties (VF)
4. If probe accuracy is high, the latent space has physics structure
5. Identify which z dimensions are most physics-predictive
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

from main.model.topologygan import TopologyGANGenerator
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 128
OUT = "results/inverted_osft"
os.makedirs(OUT, exist_ok=True)


class PhysicsProbe(nn.Module):
    """Small MLP that predicts physical properties from z."""
    def __init__(self, nz=100, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # Predict VF
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def load_gen(path):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=64,
        variant="unet", height=H, width=W,
    ).to(device)
    gen.load_state_dict(state, strict=False)
    gen.eval()
    return gen


def collect_data(gen, data_path, n_samples=500):
    """Generate (z, VF) pairs."""
    _, _, tl = create_dataloaders(
        data_path, height=H, width=W, batch_size=8, num_workers=0)

    zs, vfs = [], []
    with torch.no_grad():
        for batch in tl:
            if len(zs) >= n_samples:
                break
            cond = batch[0].to(device)
            B = cond.size(0)
            z = torch.randn(B, gen.nz, device=device)
            out = gen(cond, z=z)
            zs.append(z.cpu())
            vfs.append(out.mean(dim=[1, 2, 3]).cpu())

    return torch.cat(zs, dim=0)[:n_samples], torch.cat(vfs, dim=0)[:n_samples]


def train_probe(z_train, vf_train, z_test, vf_test, epochs=200):
    """Train physics probe and return per-dimension importance."""
    probe = PhysicsProbe(nz=z_train.size(1)).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    z_train, vf_train = z_train.to(device), vf_train.to(device)
    z_test, vf_test = z_test.to(device), vf_test.to(device)

    for ep in range(epochs):
        probe.train()
        opt.zero_grad()
        pred = probe(z_train)
        loss = crit(pred, vf_train)
        loss.backward()
        opt.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred_test = probe(z_test)
        test_loss = crit(pred_test, vf_test).item()
        # R² score
        ss_res = ((vf_test - pred_test) ** 2).sum()
        ss_tot = ((vf_test - vf_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Per-dimension importance via gradient magnitude
    per_dim_importance = []
    for dim in range(z_train.size(1)):
        z_pert = z_test.clone()
        z_pert[:, dim] += 0.1  # Perturb this dimension
        with torch.no_grad():
            delta = torch.abs(probe(z_pert) - probe(z_test)).mean()
        per_dim_importance.append(float(delta))

    return test_loss, float(r2), per_dim_importance


def main():
    print("Physics Latent Probing Experiment")
    print("=" * 60)

    # Load physics-pretrained GAN
    gen = load_gen("checkpoints/physics_pretrain/best.pt")

    # Collect data
    print("Collecting (z, VF) pairs...")
    z_all, vf_all = collect_data(gen, "data/cantilever_physics_train.npy", n_samples=500)
    print(f"  Collected {len(z_all)} samples, VF range=[{vf_all.min():.3f}, {vf_all.max():.3f}]")

    # Split
    n_train = int(len(z_all) * 0.8)
    z_train, z_test = z_all[:n_train], z_all[n_train:]
    vf_train, vf_test = vf_all[:n_train], vf_all[n_train:]

    # Train probe
    print("\nTraining physics probe (z → VF)...")
    test_loss, r2, importance = train_probe(z_train, vf_train, z_test, vf_test)
    print(f"  Test MSE: {test_loss:.6f}")
    print(f"  R²: {r2:.4f}")

    # Find top physics-controlling dimensions
    top_dims = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 physics-controlling z dimensions:")
    for rank, (dim, imp) in enumerate(top_dims):
        bar = "█" * int(imp / max(importance) * 30)
        print(f"    #{rank+1}: dim {dim:3d} | importance={imp:.6f} {bar}")

    # Key finding
    print(f"\n  {'='*60}")
    if r2 > 0.5:
        print(f"  ✅ R²={r2:.3f} > 0.5: Latent physics structure CONFIRMED")
        print(f"     z-dimensions CAN control physical properties!")
        print(f"     → Physics-Conditioned Generator is viable")
    elif r2 > 0.2:
        print(f"  ⚠️  R²={r2:.3f} > 0.2: Weak latent physics structure")
        print(f"     Some signal exists but not strong enough")
    else:
        print(f"  ❌ R²={r2:.3f} < 0.2: No latent physics structure")
        print(f"     Current training doesn't encode physics in latent space")

    # Save
    with open(f"{OUT}/latent_probe.json", "w") as f:
        json.dump({
            "test_mse": test_loss,
            "r2": r2,
            "top_dims": [(d, imp) for d, imp in top_dims[:20]],
        }, f, indent=2)
    print(f"\nSaved to {OUT}/latent_probe.json")


if __name__ == "__main__":
    main()
