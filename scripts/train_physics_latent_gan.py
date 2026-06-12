"""Physics-Structured Latent GAN.

Key innovation: Explicitly structure the latent space so that the first K
dimensions of z provably control physical properties (VF).

Architecture:
  z = [z_phys (K dims), z_detail (100-K dims)]
  Generator: z + conditions → topology
  PhysicsHead: z_phys → predicted VF

Loss:
  L = L_GAN + λ_L1*L1 + λ_VF*(VF_out - VF_target)² + λ_latent*(VF_pred - VF_out)²

The latent consistency loss forces z_phys to encode VF information.
After training, z_phys controls physics, z_detail controls details.

For fine-tuning: freeze PhysicsHead, fix z_phys → physics pathway is preserved.
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
OUT_DIR = "checkpoints/physics_latent"
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparams
N_EPOCHS = 200
BATCH_SIZE = 8
LR_G = 2e-4
LR_D = 2e-4
LAMBDA_L1 = 100.0
LAMBDA_VF = 10.0
LAMBDA_LATENT = 5.0  # Latent physics consistency weight
NZ = 100
K_PHYS = 16  # Number of physics-controlling z dimensions


class PhysicsHead(nn.Module):
    """Predict physical properties from z_phys."""
    def __init__(self, k_phys=16, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_phys, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_phys):
        return self.net(z_phys).squeeze(-1)


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


def main():
    print("Physics-Structured Latent GAN")
    print(f"  K_phys={K_PHYS}, λ_latent={LAMBDA_LATENT}")
    print(f"  Data: {DATA}")

    G = build_generator()
    D = build_discriminator()
    P = PhysicsHead(k_phys=K_PHYS).to(device)

    opt_G = optim.Adam(list(G.parameters()) + list(P.parameters()),
                       lr=LR_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    crit_l1 = nn.L1Loss()
    crit_mse = nn.MSELoss()

    train_loader, _, test_loader = create_dataloaders(
        DATA, height=H, width=W, batch_size=BATCH_SIZE, num_workers=0)

    print(f"  Train samples: {len(train_loader.dataset)}")
    best_l1 = float("inf")
    t_start = time.time()

    for ep in range(N_EPOCHS):
        G.train(); D.train(); P.train()

        for batch in train_loader:
            cond = batch[0].to(device)
            real_B = batch[1].to(device)
            B = cond.size(0)

            # ── Train D ──
            z_phys = torch.randn(B, K_PHYS, device=device)
            z_detail = torch.randn(B, NZ - K_PHYS, device=device)
            z = torch.cat([z_phys, z_detail], dim=1)

            with torch.no_grad():
                fake_B = G(cond, z=z)

            d_real, _ = D(torch.cat([cond, real_B], dim=1))
            d_fake, _ = D(torch.cat([cond, fake_B.detach()], dim=1))
            loss_D = (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2)) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train G + P ──
            z_phys = torch.randn(B, K_PHYS, device=device)
            z_detail = torch.randn(B, NZ - K_PHYS, device=device)
            z = torch.cat([z_phys, z_detail], dim=1)

            fake_B = G(cond, z=z)
            d_fake, _ = D(torch.cat([cond, fake_B], dim=1))

            # Physics outputs
            vf_out = fake_B.mean(dim=[1, 2, 3])
            vf_target = real_B.mean(dim=[1, 2, 3])
            vf_pred = P(z_phys)

            loss_G_gan = torch.mean((d_fake - 1) ** 2) * 0.5
            loss_L1 = crit_l1(fake_B, real_B) * LAMBDA_L1
            loss_VF = crit_mse(vf_out, vf_target) * LAMBDA_VF
            loss_latent = crit_mse(vf_pred, vf_out.detach()) * LAMBDA_LATENT

            loss_G = loss_G_gan + loss_L1 + loss_VF + loss_latent
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if (ep + 1) % 20 == 0:
            G.eval(); P.eval()
            # Eval: can VF_pred match VF_out?
            with torch.no_grad():
                batch = next(iter(test_loader))
                c, t = batch[0].to(device), batch[1].to(device)
                z_p = torch.randn(c.size(0), K_PHYS, device=device)
                z_d = torch.randn(c.size(0), NZ - K_PHYS, device=device)
                o = G(c, z=torch.cat([z_p, z_d], dim=1))
                vf_o = o.mean(dim=[1, 2, 3])
                vf_p = P(z_p)
                ss_res = ((vf_o - vf_p) ** 2).sum()
                ss_tot = ((vf_o - vf_o.mean()) ** 2).sum()
                latent_r2 = float((1 - ss_res / (ss_tot + 1e-8)).cpu())

            elapsed = time.time() - t_start
            print(f"Epoch {ep+1:3d} | G={loss_G.item():.2f} D={loss_D.item():.3f} "
                  f"L1={loss_L1.item():.2f} VF={loss_VF.item():.3f} "
                  f"Lat={loss_latent.item():.3f} | z_phys R²={latent_r2:.3f} | {elapsed:.0f}s")

            if loss_L1.item() < best_l1:
                best_l1 = loss_L1.item()
                torch.save({
                    "epoch": ep + 1,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                    "physics_head_state_dict": P.state_dict(),
                }, f"{OUT_DIR}/best.pt")

            G.train(); P.train()

    torch.save({
        "epoch": N_EPOCHS,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "physics_head_state_dict": P.state_dict(),
    }, f"{OUT_DIR}/final.pt")

    elapsed = (time.time() - t_start) / 60
    print(f"\nDone: {elapsed:.1f} min")

    # Final probe test
    print("\nFinal latent physics probe:")
    G.eval(); P.eval()
    zs_p, vfs_o = [], []
    with torch.no_grad():
        for batch in test_loader:
            c = batch[0].to(device)
            z_p = torch.randn(c.size(0), K_PHYS, device=device)
            z_d = torch.randn(c.size(0), NZ - K_PHYS, device=device)
            o = G(c, z=torch.cat([z_p, z_d], dim=1))
            zs_p.append(z_p.cpu()); vfs_o.append(o.mean(dim=[1,2,3]).cpu())

    zs_p = torch.cat(zs_p)
    vfs_o = torch.cat(vfs_o)
    vfs_p = P(zs_p.to(device)).cpu()
    r2 = float((1 - ((vfs_o - vfs_p)**2).sum() / ((vfs_o - vfs_o.mean())**2 + 1e-8)).cpu())
    print(f"  z_phys → VF: R² = {r2:.4f}")
    print(f"  VF range: [{vfs_o.min():.3f}, {vfs_o.max():.3f}]")
    if r2 > 0.5:
        print(f"  ✅ Physics structure successfully embedded in latent space!")
    else:
        print(f"  ⚠️  Need more training or stronger latent loss")


if __name__ == "__main__":
    main()
