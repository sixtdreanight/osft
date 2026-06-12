"""Taylor Expansion GAN: Hierarchical residual generation.

G_0: Zero-order (base structure) — full generator
G_1: First-order correction — small residual generator
G_2: Second-order correction — tiny residual generator

Training:
  1. Train G_0 on all SIMP data
  2. Freeze G_0, train G_1 on residual (target - G_0)
  3. Freeze G_0+G_1, train G_2 on residual (target - G_0 - G_1)

Fine-tuning on new condition:
  - Freeze G_0 (physics preserved by construction)
  - Optionally freeze G_1
  - Only train G_2 (lightweight adaptation)
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from main.model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 128
DATA = "data/cantilever_physics_train.npy"
OUT_DIR = "checkpoints/taylor_gan"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 8
LR = 2e-4
LAMBDA_L1 = 100.0
NZ = 100


def build_full_generator():
    """G_0: Full-size base generator."""
    return TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=64,
        variant="unet", height=H, width=W,
    ).to(device)


def build_correction_generator(gf_dim=16):
    """G_k (k>0): Small correction generator."""
    return TopologyGANGenerator(
        input_c_dim=3 + 1,  # conditions + previous cumulative output
        output_c_dim=1, gf_dim=gf_dim,
        variant="unet", height=H, width=W,
    ).to(device)


def build_discriminator():
    return TopologyGANDiscriminator(
        condition_dim=3, output_c_dim=1, df_dim=16,
        height=H, width=W,
    ).to(device)


def train_stage(G, corrections, D, train_loader, test_loader,
                stage_name, n_epochs=100, freeze_previous=True):
    """Train one Taylor stage. corrections = list of previous G_k."""

    # Freeze previous corrections
    for g in corrections:
        g.eval()
        for p in g.parameters():
            p.requires_grad = False

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    crit_l1 = nn.L1Loss()
    crit_mse = nn.MSELoss()

    best_l1 = float("inf")
    t_start = time.time()

    for ep in range(n_epochs):
        G.train(); D.train()

        for batch in train_loader:
            cond = batch[0].to(device)
            real_B = batch[1].to(device)
            B = cond.size(0)

            # Compute cumulative output from previous stages
            with torch.no_grad():
                cumulative = torch.zeros(B, 1, H, W, device=device)
                for gi, g in enumerate(corrections):
                    g_in = cond if gi == 0 else torch.cat([cond, cumulative], dim=1)
                    z = torch.randn(B, g.nz, device=device)
                    cumulative = cumulative + g(g_in, z=z)

                residual = real_B - cumulative  # what G should learn

            # ── Train D ──
            z = torch.randn(B, G.nz, device=device)
            # G_0 takes conditions only; G_k (k>0) takes [conditions, cumulative]
            if len(corrections) == 0:
                g_input = cond
            else:
                g_input = torch.cat([cond, cumulative], dim=1)
            with torch.no_grad():
                fake_B = cumulative + G(g_input, z=z)

            d_real, _ = D(torch.cat([cond, real_B], dim=1))
            d_fake, _ = D(torch.cat([cond, fake_B.detach()], dim=1))
            loss_D = (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2)) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train G ──
            z = torch.randn(B, G.nz, device=device)
            if len(corrections) == 0:
                g_input = cond
            else:
                g_input = torch.cat([cond, cumulative], dim=1)
            correction = G(g_input, z=z)
            fake_B = cumulative + correction
            d_fake, _ = D(torch.cat([cond, fake_B], dim=1))

            loss_G_gan = torch.mean((d_fake - 1) ** 2) * 0.5
            loss_L1 = crit_l1(correction, residual) * LAMBDA_L1
            loss_G = loss_G_gan + loss_L1
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if (ep + 1) % 20 == 0:
            # Eval
            G.eval()
            with torch.no_grad():
                batch = next(iter(test_loader))
                c, t = batch[0].to(device), batch[1].to(device)
                Bt = c.size(0)

                cum = torch.zeros(Bt, 1, H, W, device=device)
                for gi, g in enumerate(corrections):
                    g_input = c if gi == 0 else torch.cat([c, cum], dim=1)
                    cum = cum + g(g_input, z=torch.randn(Bt, g.nz, device=device))
                if len(corrections) == 0:
                    g_input = c
                else:
                    g_input = torch.cat([c, cum], dim=1)
                z = torch.randn(Bt, G.nz, device=device)
                out = cum + G(g_input, z=z)
                l1 = crit_l1(out, t).item()

                if len(corrections) == 0:
                    mse = crit_mse(out, t).item()
                else:
                    mse = crit_mse(G(g_input, z=z), t - cum).item()

            elapsed = time.time() - t_start
            n_corrections = len(corrections)
            order_name = f"G_{n_corrections}"
            print(f"  [{order_name}] Epoch {ep+1:3d} | G={loss_G.item():.2f} "
                  f"D={loss_D.item():.3f} L1={l1:.6f} MSE_res={mse:.6f} | {elapsed:.0f}s")

            if l1 < best_l1:
                best_l1 = l1
                torch.save({
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                }, f"{OUT_DIR}/{stage_name}.pt")

            G.train()

    elapsed = (time.time() - t_start) / 60
    print(f"  [{stage_name}] Done: {elapsed:.1f} min, best L1={best_l1:.6f}\n")
    return best_l1


def compare_on_test(G0, G1, G2, test_loader):
    """Compare cumulative output at each order."""
    from main.eval.metrics import compute_all_image_metrics

    crit_l1 = nn.L1Loss()
    crit_mse = nn.MSELoss()

    results = {}
    for label, gens in [("G_0 only", [G0]),
                         ("G_0 + G_1", [G0, G1]),
                         ("G_0 + G_1 + G_2", [G0, G1, G2])]:
        all_metrics = {}
        n = 0
        with torch.no_grad():
            for batch in test_loader:
                if n >= 20: break
                c, t = batch[0].to(device), batch[1]
                Bt = c.size(0)

                cum = torch.zeros(Bt, 1, H, W, device=device)
                for gi, g in enumerate(gens):
                    if gi == 0:
                        g_input = c  # G_0 takes only conditions
                    else:
                        g_input = torch.cat([c, cum], dim=1)
                    cum = cum + g(g_input, z=torch.randn(Bt, g.nz, device=device))

                for i in range(Bt):
                    m = compute_all_image_metrics(
                        cum[i:i+1].cpu(), t[i:i+1])
                    for k, v in m.items():
                        all_metrics[k] = all_metrics.get(k, 0.0) + v
                    n += 1

        avg = {k: v/n for k, v in all_metrics.items()}
        results[label] = avg
        print(f"  {label}: MSE={avg.get('mse', 0):.4f} "
              f"SSIM={avg.get('ssim', 0):.4f} IOU={avg.get('iou', 0):.4f}")

    return results


def main():
    print("=" * 60)
    print("Taylor Expansion GAN")
    print("=" * 60)

    train_loader, _, test_loader = create_dataloaders(
        DATA, height=H, width=W, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Test samples: {len(test_loader.dataset)}")

    # ── Stage 0: Base generator G_0 ──
    print("\n--- Stage 0: G_0 (Zero-order base) ---")
    G0 = build_full_generator()
    D0 = build_discriminator()
    train_stage(G0, [], D0, train_loader, test_loader,
                "G0_base", n_epochs=100)

    # Reload best G_0
    state = torch.load(f"{OUT_DIR}/G0_base.pt", map_location=device,
                       weights_only=False)
    G0.load_state_dict(state["generator_state_dict"])
    G0.eval()

    # ── Stage 1: First correction G_1 ──
    print("\n--- Stage 1: G_1 (First-order correction) ---")
    G1 = build_correction_generator(gf_dim=16)
    D1 = build_discriminator()
    train_stage(G1, [G0], D1, train_loader, test_loader,
                "G1_correction", n_epochs=100)

    state = torch.load(f"{OUT_DIR}/G1_correction.pt", map_location=device,
                       weights_only=False)
    G1.load_state_dict(state["generator_state_dict"])
    G1.eval()

    # ── Stage 2: Second correction G_2 ──
    print("\n--- Stage 2: G_2 (Second-order correction) ---")
    G2 = build_correction_generator(gf_dim=8)
    D2 = build_discriminator()
    train_stage(G2, [G0, G1], D2, train_loader, test_loader,
                "G2_correction", n_epochs=100)

    state = torch.load(f"{OUT_DIR}/G2_correction.pt", map_location=device,
                       weights_only=False)
    G2.load_state_dict(state["generator_state_dict"])
    G2.eval()

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("TAYLOR ORDER COMPARISON")
    print("=" * 60)
    results = compare_on_test(G0, G1, G2, test_loader)

    # ── Parameter count ──
    print("\nParameter counts:")
    for name, gen in [("G_0", G0), ("G_1", G1), ("G_2", G2)]:
        n_params = sum(p.numel() for p in gen.parameters())
        print(f"  {name}: {n_params:,}")

    total = sum(sum(p.numel() for p in g.parameters())
                for g in [G0, G1, G2])
    print(f"  Total: {total:,}")

    with open(f"{OUT_DIR}/taylor_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/taylor_results.json")


if __name__ == "__main__":
    main()
