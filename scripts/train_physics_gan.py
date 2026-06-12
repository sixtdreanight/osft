#!/usr/bin/env python3
"""Train TopologyGAN from scratch on SIMP cantilever data with physics loss.

Key: the GAN must generate physically-compliant topologies.
Loss = GAN + L1_reconstruction + lambda_vf * VF_constraint

This is the "physics-aware pretraining" for the inverted OSFT hypothesis.
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
OUT_DIR = "checkpoints/physics_pretrain"
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparams
N_EPOCHS = 100
BATCH_SIZE = 8
LR_G = 2e-4
LR_D = 2e-4
LAMBDA_L1 = 100.0   # L1 reconstruction weight
LAMBDA_VF = 10.0     # Volume fraction constraint weight
NZ = 100             # Noise dimension


def build_generator():
    return TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=64,
        variant="unet", height=H, width=W,
    ).to(device)


def build_discriminator():
    return TopologyGANDiscriminator(
        condition_dim=3, output_c_dim=1, df_dim=16,
        height=H, width=W,
    ).to(device)


def gan_loss(d_real, d_fake):
    """LSGAN loss."""
    real_loss = torch.mean((d_real - 1.0) ** 2)
    fake_loss = torch.mean(d_fake ** 2)
    return (real_loss + fake_loss) * 0.5


def vf_loss(output, target):
    """Volume fraction constraint: penalize VF mismatch."""
    vf_out = output.mean(dim=[1, 2, 3])
    vf_tgt = target.mean(dim=[1, 2, 3])
    return torch.mean((vf_out - vf_tgt) ** 2)


def main():
    print("Building models...")
    G = build_generator()
    D = build_discriminator()

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion_l1 = nn.L1Loss()

    train_loader, val_loader, test_loader = create_dataloaders(
        DATA, height=H, width=W, batch_size=BATCH_SIZE, num_workers=0)

    print(f"Training on {len(train_loader.dataset)} samples, {N_EPOCHS} epochs")
    print(f"Loss weights: L1={LAMBDA_L1}, VF={LAMBDA_VF}")

    history = {"g_loss": [], "d_loss": [], "l1_loss": [], "vf_loss": []}
    best_l1 = float("inf")
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        G.train()
        D.train()
        epoch_g, epoch_d, epoch_l1, epoch_vf = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            cond = batch[0].to(device)      # [B, 3, H, W]
            real_B = batch[1].to(device)     # [B, 1, H, W]
            B = cond.size(0)

            # ── Train Discriminator ──
            z = torch.randn(B, NZ, device=device)
            with torch.no_grad():
                fake_B = G(cond, z=z)

            d_real, _ = D(torch.cat([cond, real_B], dim=1))
            d_fake, _ = D(torch.cat([cond, fake_B.detach()], dim=1))
            loss_D = gan_loss(d_real, d_fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──
            z = torch.randn(B, NZ, device=device)
            fake_B = G(cond, z=z)
            d_fake, _ = D(torch.cat([cond, fake_B], dim=1))

            loss_G_gan = torch.mean((d_fake - 1.0) ** 2) * 0.5
            loss_L1 = criterion_l1(fake_B, real_B) * LAMBDA_L1
            loss_VF = vf_loss(fake_B, real_B) * LAMBDA_VF
            loss_G = loss_G_gan + loss_L1 + loss_VF

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()
            epoch_l1 += loss_L1.item()
            epoch_vf += loss_VF.item()
            n_batches += 1

        avg_g = epoch_g / n_batches
        avg_d = epoch_d / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_vf = epoch_vf / n_batches

        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["l1_loss"].append(avg_l1)
        history["vf_loss"].append(avg_vf)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
                  f"G={avg_g:.4f} D={avg_d:.4f} "
                  f"L1={avg_l1:.4f} VF={avg_vf:.4f} | "
                  f"{elapsed:.0f}s")

        # Save best model
        if avg_l1 < best_l1:
            best_l1 = avg_l1
            torch.save({
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "history": history,
            }, f"{OUT_DIR}/best.pt")

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch,
                "generator_state_dict": G.state_dict(),
                "discriminator_state_dict": D.state_dict(),
                "history": history,
            }, f"{OUT_DIR}/epoch_{epoch+1}.pt")

    # Final save
    torch.save({
        "epoch": N_EPOCHS,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "history": history,
    }, f"{OUT_DIR}/final.pt")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTraining complete: {elapsed:.1f} min, best L1={best_l1:.6f}")

    with open(f"{OUT_DIR}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Quick eval on test set
    print("\nQuick evaluation on test set...")
    G.eval()
    test_l1, test_vf_err = 0.0, 0.0
    n_test = 0
    with torch.no_grad():
        for batch in test_loader:
            if n_test >= 20:
                break
            cond = batch[0].to(device)
            real_B = batch[1].to(device)
            z = torch.randn(cond.size(0), NZ, device=device)
            fake_B = G(cond, z=z)
            test_l1 += criterion_l1(fake_B, real_B).item()
            test_vf_err += vf_loss(fake_B, real_B).item()
            n_test += 1
    print(f"Test L1: {test_l1/n_test:.6f}, VF err: {test_vf_err/n_test:.6f}")


if __name__ == "__main__":
    main()
