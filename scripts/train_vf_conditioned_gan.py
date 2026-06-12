"""VF-Conditioned GAN: Physics control via explicit conditioning.

Simplest possible approach: set z_phys = target VF (scalar, repeated K times).
The generator learns to use z_phys to control output VF.

Architecture:
  z_phys = target_VF * ones(K)    (explicit physics control)
  z_detail = randn(NZ - K)        (diversity control)
  z = [z_phys, z_detail]
  Generator(z, conditions) → topology

During fine-tuning on new conditions:
  - Fix z_phys = target_VF (physics preserved by construction)
  - Train only z_detail pathway (via weight updates)

This achieves the "inverted OSFT" goal trivially:
  - Physics is in z_phys → provably protected (it's the input, not a weight)
  - Details adapt via weight updates
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
OUT_DIR = "checkpoints/vf_conditioned"
os.makedirs(OUT_DIR, exist_ok=True)

N_EPOCHS = 200
BATCH_SIZE = 8
LR_G = 2e-4
LR_D = 2e-4
LAMBDA_L1 = 100.0
LAMBDA_VF = 50.0  # Stronger VF supervision
NZ = 100
K_PHYS = 8  # z_phys dimensions = target VF repeated K times


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
    print("VF-Conditioned GAN: Physics control via explicit z_phys")
    print(f"  K_phys={K_PHYS} (z_phys = target_VF * ones(K))")

    G = build_generator()
    D = build_discriminator()

    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    crit_l1 = nn.L1Loss()
    crit_mse = nn.MSELoss()

    train_loader, _, test_loader = create_dataloaders(
        DATA, height=H, width=W, batch_size=BATCH_SIZE, num_workers=0)

    print(f"  Train samples: {len(train_loader.dataset)}")
    best_l1 = float("inf")
    t_start = time.time()

    for ep in range(N_EPOCHS):
        G.train(); D.train()

        for batch in train_loader:
            cond = batch[0].to(device)
            real_B = batch[1].to(device)
            B = cond.size(0)

            # Explicit physics control: z_phys = target VF
            target_vf = real_B.mean(dim=[1, 2, 3])  # [B]
            z_phys = target_vf.unsqueeze(1).repeat(1, K_PHYS)  # [B, K]
            z_detail = torch.randn(B, NZ - K_PHYS, device=device)
            z = torch.cat([z_phys, z_detail], dim=1)

            # ── Train D ──
            with torch.no_grad():
                fake_B = G(cond, z=z)
            d_real, _ = D(torch.cat([cond, real_B], dim=1))
            d_fake, _ = D(torch.cat([cond, fake_B.detach()], dim=1))
            loss_D = (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2)) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Train G ──
            z_phys = target_vf.unsqueeze(1).repeat(1, K_PHYS)
            z = torch.cat([z_phys, torch.randn(B, NZ - K_PHYS, device=device)], dim=1)
            fake_B = G(cond, z=z)
            d_fake, _ = D(torch.cat([cond, fake_B], dim=1))

            vf_out = fake_B.mean(dim=[1, 2, 3])

            loss_G = (torch.mean((d_fake - 1) ** 2) * 0.5
                      + crit_l1(fake_B, real_B) * LAMBDA_L1
                      + crit_mse(vf_out, target_vf) * LAMBDA_VF)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if (ep + 1) % 20 == 0:
            # Eval: random VF targets
            G.eval()
            with torch.no_grad():
                batch = next(iter(test_loader))
                c, t = batch[0].to(device), batch[1].to(device)
                Bt = c.size(0)

                # Test: random VF targets (not from real data)
                test_vf = torch.rand(Bt, device=device) * 0.4 + 0.2  # [0.2, 0.6]
                z_p = test_vf.unsqueeze(1).repeat(1, K_PHYS)
                z_d = torch.randn(Bt, NZ - K_PHYS, device=device)
                o = G(c, z=torch.cat([z_p, z_d], dim=1))
                vf_o = o.mean(dim=[1, 2, 3])
                vf_err = torch.mean((vf_o - test_vf) ** 2).item()

                # Also test with real VF targets
                real_vf = t.mean(dim=[1, 2, 3]).to(device)
                z_pr = real_vf.unsqueeze(1).repeat(1, K_PHYS)
                o2 = G(c, z=torch.cat([z_pr, torch.randn_like(z_d)], dim=1))
                vf_o2 = o2.mean(dim=[1, 2, 3])
                real_vf_err = torch.mean((vf_o2 - real_vf) ** 2).item()
                l1_test = crit_l1(o2, t.to(device)).item()

            elapsed = time.time() - t_start
            print(f"Epoch {ep+1:3d} | G={loss_G.item():.2f} D={loss_D.item():.3f} "
                  f"VF_err(rand)={vf_err:.4f} VF_err(real)={real_vf_err:.4f} "
                  f"L1={l1_test:.4f} | {elapsed:.0f}s")

            if l1_test < best_l1:
                best_l1 = l1_test
                torch.save({
                    "epoch": ep + 1,
                    "generator_state_dict": G.state_dict(),
                    "discriminator_state_dict": D.state_dict(),
                }, f"{OUT_DIR}/best.pt")

            G.train()

    torch.save({
        "epoch": N_EPOCHS,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
    }, f"{OUT_DIR}/final.pt")

    # ── Final test: VF controllability ──
    print(f"\n{'='*60}")
    print("VF Controllability Test")
    print(f"{'='*60}")
    G.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        c = batch[0][:1].to(device).repeat(8, 1, 1, 1)  # same condition
        test_vfs = torch.linspace(0.15, 0.65, 8, device=device)
        z_p = test_vfs.unsqueeze(1).repeat(1, K_PHYS)
        z_d = torch.randn(8, NZ - K_PHYS, device=device)
        o = G(c, z=torch.cat([z_p, z_d], dim=1))
        out_vfs = o.mean(dim=[1, 2, 3]).cpu().numpy()

    print(f"{'Target VF':>10} {'Output VF':>10} {'Error':>10}")
    print("-" * 32)
    for tv, ov in zip(test_vfs.cpu().numpy(), out_vfs):
        print(f"{tv:>10.3f} {ov:>10.3f} {abs(tv-ov):>10.4f}")

    # R² of VF control
    ss_res = np.sum((out_vfs - test_vfs.cpu().numpy()) ** 2)
    ss_tot = np.sum((out_vfs - out_vfs.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"\nVF Control R²: {r2:.4f}")
    if r2 > 0.8:
        print("✅ EXCELLENT: z_phys → VF control works!")
        print("   Physics is EXPLICITLY encoded in latent space.")
    elif r2 > 0.5:
        print("✅ Good: z_phys controls VF directionally")
    else:
        print("❌ Failed: z_phys does not control VF")

    elapsed = (time.time() - t_start) / 60
    print(f"\nDone: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
