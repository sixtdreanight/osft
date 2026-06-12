"""E_EXP1: Curriculum learning for physics loss.

Standard OSFT applies full physics loss from epoch 1, which may disrupt
topology before GAN stabilizes. Curriculum learning phases in physics:
  Phase 1 (warmup): pure GAN+L1, no physics
  Phase 2 (transition): physics weight linearly increases
  Phase 3 (converge): full physics, fixed

Hypothesis: reduces seed variance (CV from 4.5% → <2%) and improves final MSE.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np

from main.model.topologygan import TopologyGANGenerator
from main.model.physics_loss import PhysicsConstraintLoss
from main.osft.config import OSFTConfig
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "data/synthetic_train.npy"
PRETRAINED = "checkpoints/quickstart/pretrained_generator.pt"
RESULTS = "results/phase2"
os.makedirs(RESULTS, exist_ok=True)


def load_gen():
    state = torch.load(PRETRAINED, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    return gen


class CurriculumOSFTTrainer(OSFTTrainer):
    """OSFT trainer with phased physics loss weight."""

    def __init__(self, config, generator, curriculum_config=None):
        super().__init__(config, generator=generator)
        self.curriculum = curriculum_config or {}
        self.physics_weight = 0.0

    def set_physics_weight(self, weight):
        self.physics_weight = weight

    def train_step(self, batch, train_d=True):
        from torch.amp import autocast, GradScaler
        if not hasattr(self, 'scaler'):
            self.scaler = GradScaler("cuda", enabled=self.cfg.use_amp)

        real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
        bc = batch[2].to(self.device) if len(batch) > 2 else None
        load_x = batch[3].to(self.device) if len(batch) > 3 else None
        load_y = batch[4].to(self.device) if len(batch) > 4 else None
        metrics = {}

        if train_d:
            with autocast("cuda", enabled=self.cfg.use_amp):
                fake_B = self.generator(real_A)
                real_AB = torch.cat([real_A, bc, load_x, load_y, real_B], dim=1)
                fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B.detach()], dim=1)
                _, d_real = self.discriminator(real_AB)
                _, d_fake = self.discriminator(fake_AB)
                d_loss = self.bce_loss(d_real, torch.ones_like(d_real)) + \
                         self.bce_loss(d_fake, torch.zeros_like(d_fake))
            self.d_optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
            metrics["D_loss"] = d_loss.item()

        with autocast("cuda", enabled=self.cfg.use_amp):
            fake_B = self.generator(real_A)
            fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B], dim=1)
            _, d_fake = self.discriminator(fake_AB)
            g_gan = self.bce_loss(d_fake, torch.ones_like(d_fake))
            g_l1 = self.l1_loss(fake_B, real_B)
            phys = self.physics_loss(fake_B, real_B, bc, load_x, load_y)
            g_loss = (self.cfg.lambda_gan * g_gan +
                      self.cfg.lambda_l1 * g_l1 +
                      self.physics_weight * phys["total"])

        self.g_optimizer.zero_grad()
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        metrics.update({"G_loss": g_loss.item(), "G_gan": g_gan.item(),
                        "G_l1": g_l1.item(), "phys_weight": self.physics_weight})
        self.global_step += 1
        return metrics

    def train(self, train_loader, val_loader, n_epochs=50, resume=False):
        self._clear_interrupt()
        start_epoch = 0
        best_val_mse = float("inf")

        warmup = self.curriculum.get("warmup_epochs", 20)
        ramp = self.curriculum.get("ramp_epochs", 20)
        target_weight = self.curriculum.get("target_weight", 1.0)

        for epoch in range(start_epoch, n_epochs):
            # Curriculum: phase in physics loss
            if epoch < warmup:
                w = 0.0
            elif epoch < warmup + ramp:
                w = target_weight * (epoch - warmup) / ramp
            else:
                w = target_weight
            self.set_physics_weight(w)

            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            if (epoch + 1) % 5 == 0 or epoch < 10:
                print(f"  Epoch {epoch+1}/{n_epochs} ({elapsed:.1f}s) "
                      f"G={train_metrics['G_loss']:.2f} w={w:.2f}")

            if (epoch + 1) % self.cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader)
                if val_metrics["val_mse"] < best_val_mse:
                    best_val_mse = val_metrics["val_mse"]
                    self.save_checkpoint(epoch=epoch + 1, best=True)

            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch=epoch + 1)

            if self._should_stop():
                break

        print(f"  Best val_mse={best_val_mse:.6f}")
        return self.generator


def main():
    train_loader, val_loader, test_loader = create_dataloaders(
        DATA, height=64, width=128, batch_size=16, num_workers=0)

    results = {}
    configs = {
        "standard (w=1.0)": {"warmup": 0, "ramp": 0, "target": 1.0},
        "curriculum (w=0→1)": {"warmup": 20, "ramp": 20, "target": 1.0},
        "curriculum (w=0→1, fast)": {"warmup": 10, "ramp": 10, "target": 1.0},
    }

    for cfg_name, curr in configs.items():
        print(f"\n{'='*60}")
        print(f"Curriculum: {cfg_name}")
        print(f"  warmup={curr['warmup']}, ramp={curr['ramp']}, target={curr['target']}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        gen = load_gen()
        os_cfg = OSFTConfig(
            n_epochs=50,
            checkpoint_dir=f"{RESULTS}/curriculum_{cfg_name.replace(' ', '_')}",
            lr=1e-4, eval_every=5, save_every=999,
        )
        trainer = CurriculumOSFTTrainer(os_cfg, generator=gen,
                                         curriculum_config={
                                             "warmup_epochs": curr["warmup"],
                                             "ramp_epochs": curr["ramp"],
                                             "target_weight": curr["target"],
                                         })
        trainer.apply_svd_decomposition()
        trainer.g_optimizer = optim.Adam(
            [p for p in gen.parameters() if p.requires_grad],
            lr=os_cfg.lr, betas=(os_cfg.beta1, os_cfg.beta2))
        trainer.train(train_loader, val_loader, n_epochs=50)

        m = evaluate_model(gen, test_loader, device)
        tp = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        total = sum(p.numel() for p in gen.parameters())
        m["trainable_pct"] = 100 * tp / total
        results[cfg_name] = m
        del gen, trainer
        torch.cuda.empty_cache()

    # Print comparison
    print("\n" + "=" * 70)
    print("E_EXP1: Curriculum Learning Results")
    print("=" * 70)
    print(f"{'Config':<35} {'MSE':>10} {'SSIM':>8} {'IOU':>8} {'PSNR':>8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<35} {m['mse']:>10.6f} {m['ssim']:>8.4f} "
              f"{m['iou']:>8.4f} {m.get('psnr',0):>8.2f}")

    # Compute improvement
    base_mse = results["standard (w=1.0)"]["mse"]
    for name in results:
        if name != "standard (w=1.0)":
            delta = (base_mse - results[name]["mse"]) / base_mse * 100
            print(f"  {name}: MSE {delta:+.1f}% vs standard")

    with open(f"{RESULTS}/curriculum_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS}/curriculum_results.json")


if __name__ == "__main__":
    main()
