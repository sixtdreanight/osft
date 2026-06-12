"""Full fine-tuning baseline trainer.

Standard GAN fine-tuning where all generator and discriminator parameters
are trainable. Used as the quality upper bound for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import os
import time
from collections import defaultdict

from ..osft.config import OSFTConfig
from ..osft.checkpoint import CheckpointMixin
from ..model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from ..model.physics_loss import PhysicsConstraintLoss
from ..eval.metrics import compute_all_image_metrics
from ..utils.logger import ExperimentLogger


class FullFinetuneTrainer(CheckpointMixin):
    """Standard full fine-tuning trainer for TopologyGAN."""

    def __init__(self, config: OSFTConfig, generator: Optional[nn.Module] = None,
                 discriminator: Optional[nn.Module] = None):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = ExperimentLogger(config.checkpoint_dir, "full_finetune")
        self._ckpt_prefix = "full_ft"

        if generator is not None:
            self.generator = generator.to(self.device)
        else:
            self.generator = TopologyGANGenerator(
                input_c_dim=config.input_c_dim,
                output_c_dim=config.output_c_dim,
                gf_dim=config.gf_dim,
                variant=config.generator_variant,
                height=config.img_height,
                width=config.img_width,
            ).to(self.device)

        if discriminator is not None:
            self.discriminator = discriminator.to(self.device)
        else:
            self.discriminator = TopologyGANDiscriminator(
                condition_dim=config.condition_dim,
                output_c_dim=config.output_c_dim,
                df_dim=config.df_dim,
                height=config.img_height,
                width=config.img_width,
            ).to(self.device)

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.physics_loss = PhysicsConstraintLoss(
            lambda_vf=config.lambda_vf,
            lambda_comp=config.lambda_comp,
        )

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )

        self.scaler = GradScaler("cuda", enabled=config.use_amp)
        self.global_step = 0

        self._register_signal_handlers()

    def load_pretrained_generator(self, checkpoint_path: str):
        """Load pre-trained generator weights.

        Handles both raw state_dict and wrapped checkpoint formats.
        Also loads discriminator if available in the checkpoint.
        """
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "generator_state_dict" in state:
            gen_state = state["generator_state_dict"]
            disc_state = state.get("discriminator_state_dict", None)
        elif isinstance(state, dict) and "model_state_dict" in state:
            gen_state = state["model_state_dict"]
            disc_state = None
        else:
            gen_state = state
            disc_state = None
        self.generator.load_state_dict(gen_state, strict=False)
        self.logger.info(f"Loaded pre-trained generator from {checkpoint_path}")
        if disc_state is not None:
            self.discriminator.load_state_dict(disc_state, strict=False)
            self.logger.info(f"Loaded pre-trained discriminator from {checkpoint_path}")

    def train_step(self, batch: Tuple[torch.Tensor, ...], train_d: bool = True) -> Dict[str, float]:
        """Single training step."""
        real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
        bc = batch[2].to(self.device) if len(batch) > 2 else None
        load_x = batch[3].to(self.device) if len(batch) > 3 else None
        load_y = batch[4].to(self.device) if len(batch) > 4 else None
        metrics = {}

        # Train Discriminator
        if train_d:
            with autocast("cuda", enabled=self.cfg.use_amp):
                fake_B = self.generator(real_A)
                real_AB = torch.cat([real_A, bc, load_x, load_y, real_B], dim=1)
                fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B.detach()], dim=1)
                _, d_real_logits = self.discriminator(real_AB)
                _, d_fake_logits = self.discriminator(fake_AB)
                d_loss = self.bce_loss(d_real_logits, torch.ones_like(d_real_logits)) + \
                          self.bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))

            self.d_optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
            metrics["D_loss"] = d_loss.item()

        # Train Generator
        with autocast("cuda", enabled=self.cfg.use_amp):
            fake_B = self.generator(real_A)
            fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B], dim=1)
            _, d_fake_logits = self.discriminator(fake_AB)

            g_gan = self.bce_loss(d_fake_logits, torch.ones_like(d_fake_logits))
            g_l1 = self.l1_loss(fake_B, real_B)
            phys = self.physics_loss(fake_B, real_B, bc, load_x, load_y)

            g_loss = (
                self.cfg.lambda_gan * g_gan +
                self.cfg.lambda_l1 * g_l1 +
                phys["total"]
            )

        self.g_optimizer.zero_grad()
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        metrics.update({
            "G_loss": g_loss.item(),
            "G_gan": g_gan.item(),
            "G_l1": g_l1.item(),
        })
        self.global_step += 1
        return metrics

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        epoch_metrics = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            train_d = (batch_idx % self.cfg.gradient_accumulation_steps == 0)
            metrics = self.train_step(batch, train_d=train_d)
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        self.logger.log_metrics(epoch_metrics, epoch, prefix="train")
        return dict(epoch_metrics)

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on validation/test set with full metric suite."""
        self.generator.eval()

        metric_names = ["mse", "mae", "psnr", "ssim", "iou", "lpips", "vfae"]
        totals = {k: 0.0 for k in metric_names}
        n_samples = 0

        for batch in dataloader:
            real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
            fake_B = self.generator(real_A)

            m = compute_all_image_metrics(fake_B, real_B)
            batch_size = real_A.size(0)
            for k in metric_names:
                totals[k] += m[k] * batch_size
            n_samples += batch_size

        metrics = {f"val_{k}": totals[k] / max(n_samples, 1) for k in metric_names}
        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: Optional[int] = None,
        resume: bool = False,
        resume_from: Optional[str] = None,
    ):
        """Full training loop with checkpoint/resume support.

        Args:
            train_loader: Training data
            val_loader: Validation data
            n_epochs: Number of epochs (default: from config)
            resume: If True, auto-resume from latest checkpoint
            resume_from: Explicit checkpoint path to resume from
        """
        self._clear_interrupt()
        epochs = n_epochs or self.cfg.n_epochs

        # Resume from checkpoint
        start_epoch = 0
        if resume or resume_from:
            start_epoch = self.resume(resume_from)
            if start_epoch > 0:
                self.logger.info(f"Resumed from epoch {start_epoch}")

        best_val_mse = float("inf")

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                f"G_loss: {train_metrics['G_loss']:.4f}, "
                f"D_loss: {train_metrics.get('D_loss', 0):.4f}"
            )

            # Evaluate
            if (epoch + 1) % self.cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader)
                if val_metrics["val_mse"] < best_val_mse:
                    best_val_mse = val_metrics["val_mse"]
                    self.save_checkpoint(epoch=epoch + 1, best=True)
                    self.logger.info(f"New best model! val_mse={best_val_mse:.6f}")

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch=epoch + 1)

            # Handle interrupt
            if self._should_stop():
                self.logger.info(f"Graceful shutdown at epoch {epoch+1}. "
                                 f"Resume with: train(resume=True)")
                break

        self.logger.info("Training completed.")
        return self.generator
