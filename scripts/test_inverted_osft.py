#!/usr/bin/env python3
"""Test the inverted OSFT hypothesis.

1. SVD-eigenmode correlation: Random-pretrained vs Physics-pretrained
2. OSFT fine-tuning on MBB Beam: Random-pretrained vs Physics-pretrained
3. Cross-condition transfer comparison
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr

from main.model.topologygan import TopologyGANGenerator
from main.osft.decomposer import SVDWeightDecomposer
from main.osft.subspace_layers import apply_osft_to_generator
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 128
OUT = "results/inverted_osft"
os.makedirs(OUT, exist_ok=True)


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


# ═══════════════════════════════════════════════════════════════
# Part 1: SVD-Eigenmode correlation comparison
# ═══════════════════════════════════════════════════════════════

def compute_eigenmode_correlation(gen, data_path, label):
    """Measure Pearson r between SVD perturbation fields and beam eigenmodes."""
    # Analytical cantilever eigenmodes
    beta_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])
    sigma = (np.cosh(beta_L) + np.cos(beta_L)) / (np.sinh(beta_L) + np.sin(beta_L))
    x = np.linspace(0, 1.0, W)
    y = np.linspace(0, 1.0, H)
    X, Y = np.meshgrid(x, y)
    eigenmodes = []
    for n in range(4):
        bx = beta_L[n] * X
        s = sigma[n]
        phi = np.cosh(bx) - np.cos(bx) - s * (np.sinh(bx) - np.sin(bx))
        eigenmodes.append((phi / (np.abs(phi).max() + 1e-12)).astype(np.float32))

    # Fixed input
    _, _, tl = create_dataloaders(
        data_path, height=H, width=W, batch_size=4, num_workers=0)
    batch = next(iter(tl))
    conds = batch[0].to(device)
    B = conds.size(0)

    z = torch.randn(B, gen.nz, device=device)
    with torch.no_grad():
        baseline = gen(conds, z=z).cpu()

    # Target conv layers
    target_layers = {}
    for name, mod in gen.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            if any(k in name for k in ["d2", "d3", "e5"]):
                target_layers[name] = mod

    results = []
    for lname, mod in target_layers.items():
        W = mod.weight.data.clone()
        W2d = W.view(W.size(0), -1).cpu().numpy()
        U, S, _ = np.linalg.svd(W2d, full_matrices=False)

        for vi in range(3):
            u = torch.tensor(U[:, vi], dtype=torch.float32, device=device)
            delta = 0.2 * u.view(-1, 1, 1, 1).expand_as(W)
            mod.weight.data = W + delta
            with torch.no_grad():
                pert = gen(conds, z=z).cpu()
            mod.weight.data = W

            diff = (pert - baseline).abs().mean(dim=0)[0].numpy()
            for mi, em in enumerate(eigenmodes):
                r, p = pearsonr(diff.ravel(), em.ravel())
                results.append({
                    "layer": lname, "svd_vec": vi + 1, "eigenmode": mi + 1,
                    "pearson_r": float(r), "p_value": float(p),
                })

    all_abs = sorted([abs(r["pearson_r"]) for r in results], reverse=True)
    best = max(results, key=lambda x: abs(x["pearson_r"]))

    print(f"\n  {label}:")
    print(f"    Best: {best['layer']} SVD#{best['svd_vec']} vs Mode#{best['eigenmode']}, "
          f"r={best['pearson_r']:.4f}")
    print(f"    Top-5 |r|: {[f'{v:.3f}' for v in all_abs[:5]]}")
    print(f"    Mean top-5 |r|: {np.mean(all_abs[:5]):.4f}")

    return {
        "best_r": float(best["pearson_r"]),
        "top5_mean_r": float(np.mean(all_abs[:5])),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════
# Part 2: OSFT cross-condition transfer
# ═══════════════════════════════════════════════════════════════

def osft_finetune(gen, train_data, target_data, name, epochs=50):
    """OSFT fine-tune and evaluate on target condition."""
    # SVD decompose
    decomp = SVDWeightDecomposer(energy_threshold=0.80)
    decomp.decompose_model(gen, verbose=False)
    apply_osft_to_generator(gen, decomp.results)

    trainable = [p for p in gen.parameters() if p.requires_grad]
    print(f"\n  {name}: {sum(p.numel() for p in trainable)} trainable params")

    train_loader, _, test_loader = create_dataloaders(
        target_data, height=H, width=W, batch_size=4, num_workers=0)

    opt = optim.Adam(trainable, lr=1e-4)
    crit = nn.L1Loss()

    best_loss = float("inf")
    for ep in range(epochs):
        gen.train()
        tl = 0.0
        for batch in train_loader:
            c, t = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            l = crit(gen(c), t)
            l.backward()
            opt.step()
            tl += l.item()
        avg = tl / len(train_loader)
        if avg < best_loss:
            best_loss = avg
            torch.save({"generator_state_dict": gen.state_dict()},
                       f"{OUT}/{name}_best.pt")
        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1}: loss={avg:.6f}")

    # Evaluate
    from main.eval.metrics import compute_all_image_metrics
    gen.eval()
    ms = {}
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            if n >= 20:
                break
            c, t = batch[0].to(device), batch[1]
            o = gen(c).cpu()
            for i in range(o.size(0)):
                m = compute_all_image_metrics(o[i:i+1], t[i:i+1])
                for k, v in m.items():
                    ms[k] = ms.get(k, 0.0) + v
                n += 1

    metrics = {k: v / n for k, v in ms.items()}
    print(f"    MSE={metrics.get('mse', 0):.4f} SSIM={metrics.get('ssim', 0):.4f} "
          f"IOU={metrics.get('iou', 0):.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    RANDOM_CKPT = "checkpoints/quickstart/pretrained_generator.pt"
    PHYSICS_CKPT = "checkpoints/physics_pretrain/best.pt"
    CANTILEVER_DATA = "data/cantilever_physics_train.npy"
    MBB_DATA = "data/mbb_beam_train.npy"

    print("=" * 70)
    print("INVERTED OSFT HYPOTHESIS TEST")
    print("=" * 70)

    # Part 1: Eigenmode correlation
    print("\n--- Part 1: SVD-Eigenmode Correlation ---")

    if os.path.exists(RANDOM_CKPT):
        gen_random = load_gen(RANDOM_CKPT)
        corr_random = compute_eigenmode_correlation(
            gen_random, CANTILEVER_DATA, "Random-Pretrained")
    else:
        print("  Random-Pretrained checkpoint not found, skipping")
        corr_random = None

    if os.path.exists(PHYSICS_CKPT):
        gen_physics = load_gen(PHYSICS_CKPT)
        corr_physics = compute_eigenmode_correlation(
            gen_physics, CANTILEVER_DATA, "Physics-Pretrained")
    else:
        print("  Physics-Pretrained checkpoint not found, skipping")
        corr_physics = None

    # Part 2: OSFT cross-condition transfer
    print("\n--- Part 2: OSFT Cross-Condition Transfer (Cantilever → MBB) ---")

    transfer_results = {}

    if os.path.exists(RANDOM_CKPT) and os.path.exists(MBB_DATA):
        print("\n  Random-Pretrained → MBB Beam (OSFT):")
        gen = load_gen(RANDOM_CKPT)
        transfer_results["Random-OSFT"] = osft_finetune(
            gen, CANTILEVER_DATA, MBB_DATA, "random_osft_mbb")

    if os.path.exists(PHYSICS_CKPT) and os.path.exists(MBB_DATA):
        print("\n  Physics-Pretrained → MBB Beam (OSFT):")
        gen = load_gen(PHYSICS_CKPT)
        transfer_results["Physics-OSFT"] = osft_finetune(
            gen, CANTILEVER_DATA, MBB_DATA, "physics_osft_mbb")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if corr_random and corr_physics:
        print(f"\nEigenmode Correlation:")
        print(f"  Random-Pretrained:  |r|={corr_random['best_r']:.4f} "
              f"(top5 mean={corr_random['top5_mean_r']:.4f})")
        print(f"  Physics-Pretrained: |r|={corr_physics['best_r']:.4f} "
              f"(top5 mean={corr_physics['top5_mean_r']:.4f})")
        delta = corr_physics["best_r"] - corr_random["best_r"]
        print(f"  Δ|r| = {delta:+.4f}")

    if transfer_results:
        print(f"\nCross-Condition Transfer (→ MBB Beam):")
        for name, m in transfer_results.items():
            print(f"  {name}: MSE={m.get('mse',0):.4f} SSIM={m.get('ssim',0):.4f} "
                  f"IOU={m.get('iou',0):.4f}")

    # Save
    report = {
        "eigenmode_correlation": {
            "random": corr_random,
            "physics": corr_physics,
        },
        "transfer_results": transfer_results,
    }
    # Strip detailed results for clean JSON
    if corr_random:
        report["eigenmode_correlation"]["random"] = {
            "best_r": corr_random["best_r"],
            "top5_mean_r": corr_random["top5_mean_r"],
            "top5": sorted(
                [abs(r["pearson_r"]) for r in corr_random["results"]],
                reverse=True,
            )[:5],
        }
    if corr_physics:
        report["eigenmode_correlation"]["physics"] = {
            "best_r": corr_physics["best_r"],
            "top5_mean_r": corr_physics["top5_mean_r"],
            "top5": sorted(
                [abs(r["pearson_r"]) for r in corr_physics["results"]],
                reverse=True,
            )[:5],
        }

    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUT}/report.json")


if __name__ == "__main__":
    main()
