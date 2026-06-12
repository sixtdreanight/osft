"""OSFT Fine-Tuning Trainer.

Complete training loop for Orthogonal Subspace Fine-Tuning of TopologyGAN
with GAN loss, L1 reconstruction loss, physics constraints, subspace
regularization, checkpoint/resume, and spectral tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import os
import time
import numpy as np
from collections import defaultdict

from .config import OSFTConfig
from .checkpoint import CheckpointMixin
from .decomposer import SVDWeightDecomposer
from .subspace_layers import (
    OrthogonalSubspaceConv2d,
    OrthogonalSubspaceConvTranspose2d,
    OrthogonalSubspaceLinear,
    apply_osft_to_generator,
)
from ..model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from ..model.physics_loss import PhysicsConstraintLoss
from ..eval.metrics import compute_all_image_metrics
from ..utils.logger import ExperimentLogger


class OSFTTrainer(CheckpointMixin):
    """Orthogonal Subspace Fine-Tuning Trainer for TopologyGAN."""

    def __init__(self, config: OSFTConfig, generator: Optional[nn.Module] = None,
                 discriminator: Optional[nn.Module] = None):
        self.cfg = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger = ExperimentLogger(config.checkpoint_dir, "osft_trainer")
        self._ckpt_prefix = "osft"

        self.generator = (generator if generator is not None
                          else self._build_generator()).to(self.device)
        self.discriminator = (discriminator if discriminator is not None
                              else self._build_discriminator()).to(self.device)

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.physics_loss = PhysicsConstraintLoss(
            lambda_vf=config.lambda_vf,
            lambda_comp=config.lambda_comp,
        )

        self.g_optimizer = optim.Adam(
            self._get_trainable_params(self.generator),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )

        self.scaler = GradScaler("cuda", enabled=config.use_amp)
        self.global_step = 0

        self._register_signal_handlers()

    def _build_generator(self) -> nn.Module:
        cfg = self.cfg
        return TopologyGANGenerator(
            input_c_dim=cfg.input_c_dim,
            output_c_dim=cfg.output_c_dim,
            gf_dim=cfg.gf_dim,
            variant=cfg.generator_variant,
            height=cfg.img_height,
            width=cfg.img_width,
        )

    def _build_discriminator(self) -> nn.Module:
        cfg = self.cfg
        return TopologyGANDiscriminator(
            condition_dim=cfg.condition_dim,
            output_c_dim=cfg.output_c_dim,
            df_dim=cfg.df_dim,
            height=cfg.img_height,
            width=cfg.img_width,
        )

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

    def apply_svd_decomposition(self, energy_threshold: Optional[float] = None):
        """Perform SVD decomposition and apply OSFT layers to generator."""
        threshold = energy_threshold or self.cfg.energy_threshold
        decomposer = SVDWeightDecomposer(energy_threshold=threshold)
        decomposer.decompose_model(self.generator, target_layers=self.cfg.target_layers)

        summary = decomposer.summary()
        self.logger.info(f"SVD decomposition: {summary}")

        apply_osft_to_generator(self.generator, decomposer.results)
        self._decomposer = decomposer

        self.g_optimizer = optim.Adam(
            self._get_trainable_params(self.generator),
            lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2),
        )
        return decomposer

    def _get_trainable_params(self, model: nn.Module):
        return [p for p in model.parameters() if p.requires_grad]

    def _compute_osft_losses(self) -> Dict[str, torch.Tensor]:
        loss_orth = torch.tensor(0.0, device=self.device)
        loss_ksv = torch.tensor(0.0, device=self.device)
        count = 0

        for module in self.generator.modules():
            if isinstance(module, (OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d, OrthogonalSubspaceLinear)):
                loss_orth += module.orthogonality_loss()
                loss_ksv += module.singular_value_constraint()
                count += 1

        if count > 0:
            loss_orth /= count
            loss_ksv /= count

        return {"L_orth": loss_orth, "L_ksv": loss_ksv}

    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        train_d: bool = True,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: (real_A, real_B, bc, load_x, load_y)
            train_d: Whether to train discriminator this step.
        """
        real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
        bc = batch[2].to(self.device) if len(batch) > 2 else None
        load_x = batch[3].to(self.device) if len(batch) > 3 else None
        load_y = batch[4].to(self.device) if len(batch) > 4 else None

        metrics = {}

        # === Train Discriminator ===
        if train_d:
            with autocast("cuda", enabled=self.cfg.use_amp):
                fake_B = self.generator(real_A)

                real_AB = torch.cat([real_A, bc, load_x, load_y, real_B], dim=1)
                fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B.detach()], dim=1)

                _, d_real_logits = self.discriminator(real_AB)
                _, d_fake_logits = self.discriminator(fake_AB)

                d_loss_real = self.bce_loss(d_real_logits, torch.ones_like(d_real_logits))
                d_loss_fake = self.bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))
                d_loss = d_loss_real + d_loss_fake

            self.d_optimizer.zero_grad()
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
            self.scaler.update()

            metrics["D_loss"] = d_loss.item()
            metrics["D_real"] = d_loss_real.item()
            metrics["D_fake"] = d_loss_fake.item()

        # === Train Generator ===
        with autocast("cuda", enabled=self.cfg.use_amp):
            fake_B = self.generator(real_A)

            fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B], dim=1)
            _, d_fake_logits = self.discriminator(fake_AB)

            g_gan = self.bce_loss(d_fake_logits, torch.ones_like(d_fake_logits))
            g_l1 = self.l1_loss(fake_B, real_B)
            phys = self.physics_loss(fake_B, real_B, bc, load_x, load_y)
            osft_reg = self._compute_osft_losses()

            g_loss = (
                self.cfg.lambda_gan * g_gan +
                self.cfg.lambda_l1 * g_l1 +
                phys["total"] +
                self.cfg.lambda_orth * osft_reg["L_orth"] +
                self.cfg.lambda_ksv * osft_reg["L_ksv"]
            )

        self.g_optimizer.zero_grad()
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        metrics.update({
            "G_loss": g_loss.item(),
            "G_gan": g_gan.item(),
            "G_l1": g_l1.item(),
            "G_vf": phys.get("vf_loss", torch.tensor(0.0)).item(),
            "G_comp": phys.get("comp_loss", torch.tensor(0.0)).item(),
            "G_orth": osft_reg["L_orth"].item(),
            "G_ksv": osft_reg["L_ksv"].item(),
        })

        self.global_step += 1
        return metrics

    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
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

        metrics = {f"val_{k}": totals[k] / n_samples for k in metric_names}
        self.logger.log_metrics(metrics, self.global_step, prefix="eval")
        return metrics

    # ============================================================
    # Tracking hooks for spectral analysis (E2-E7 experiments)
    # ============================================================

    def enable_tracking(self):
        """Enable comprehensive tracking for experiments."""
        self._svd_snapshots = []
        self._grad_projection_history = []
        self._betti_history = []
        self._feature_snapshots = {}
        self._tracking_enabled = True

    def _extra_checkpoint_state(self) -> dict:
        """Extra state for checkpoint resume (tracking data)."""
        extra = {}
        if getattr(self, '_decomposer', None) is not None:
            extra["decomposer_results"] = self._decomposer.results
        if getattr(self, '_tracking_enabled', False):
            extra["svd_snapshots"] = getattr(self, '_svd_snapshots', [])
            extra["grad_projection_history"] = getattr(self, '_grad_projection_history', [])
            extra["betti_history"] = getattr(self, '_betti_history', [])
        return extra

    def snapshot_svd(self):
        """Record current singular value distribution for SVD dynamics (E6)."""
        from ..eval.spectral import SingularValueAnalyzer
        analyzer = SingularValueAnalyzer(self.generator)
        snapshot = {
            "epoch": getattr(self, '_current_epoch', 0),
            "spectra": analyzer.layer_spectra(),
            "ranks": analyzer.compute_all_ranks(),
        }
        self._svd_snapshots.append(snapshot)
        return snapshot

    def snapshot_gradient_projection(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Record gradient projection ratios for gradient flow analysis (E4).

        Computes η = ||G_res||_F^2 / ||G_phy||_F^2 for each OSFT layer.
        """
        from ..eval.spectral import GradientProjectionAnalyzer
        decomp = getattr(self, '_decomposer', None)
        if decomp is None:
            return {}
        analyzer = GradientProjectionAnalyzer(decomp.results)
        ratios = analyzer.snapshot_gradients(
            self.generator,
            lambda m, b: self._physics_grad_loss(m, b),
            batch,
            self.device,
        )
        avg_eta = analyzer.average_ratio()
        self._grad_projection_history.append({
            "epoch": getattr(self, '_current_epoch', 0),
            "layer_ratios": ratios,
            "avg_eta": avg_eta,
        })
        return {"avg_eta": avg_eta}

    def _physics_grad_loss(self, model: nn.Module, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Physics-only loss for gradient projection analysis."""
        real_A = batch[0].to(self.device)
        real_B = batch[1].to(self.device)
        fake_B = model(real_A)
        phys = self.physics_loss(
            fake_B, real_B,
            batch[2].to(self.device) if len(batch) > 2 else None,
            batch[3].to(self.device) if len(batch) > 3 else None,
            batch[4].to(self.device) if len(batch) > 4 else None,
        )
        return phys["total"]

    def snapshot_betti(self, dataloader, max_batches: int = 5) -> Dict[str, float]:
        """Evaluate Betti number preservation (E2, E3)."""
        from ..eval.spectral import TopologyFeatureAnalyzer
        self.generator.eval()
        all_results = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
                fake_B = self.generator(real_A)
                result = TopologyFeatureAnalyzer.batch_betti_analysis(fake_B, real_B)
                all_results.append(result)

        avg = {}
        for key in all_results[0]:
            avg[key] = float(np.mean([r[key] for r in all_results]))
        self._betti_history.append(avg)
        return avg

    def get_tracking_report(self) -> Dict:
        """Compile all tracking data into a report."""
        report = {}

        if self._svd_snapshots:
            if len(self._svd_snapshots) >= 2:
                first = self._svd_snapshots[0]["ranks"]
                last = self._svd_snapshots[-1]["ranks"]
                eff_rank_changes = {}
                for name in first:
                    if name in last:
                        delta = last[name]["effective_rank"] - first[name]["effective_rank"]
                        eff_rank_changes[name] = delta
                report["svd_effective_rank_change"] = eff_rank_changes
                report["svd_n_snapshots"] = len(self._svd_snapshots)

        if self._grad_projection_history:
            report["grad_projection_final_eta"] = self._grad_projection_history[-1]["avg_eta"]
            report["grad_projection_history"] = [
                {"epoch": h["epoch"], "eta": h["avg_eta"]}
                for h in self._grad_projection_history
            ]

        if self._betti_history:
            report["betti_final"] = self._betti_history[-1]

        return report

    # ============================================================
    # Full training loop with resume support
    # ============================================================

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: Optional[int] = None,
        resume: bool = False,
        resume_from: Optional[str] = None,
        track_svd_every: int = 0,
        track_grad_every: int = 0,
        track_betti_every: int = 0,
    ):
        """Full training loop with checkpoint/resume and spectral tracking.

        Args:
            train_loader: Training data
            val_loader: Validation data
            n_epochs: Number of epochs (default: from config)
            resume: If True, auto-resume from latest checkpoint
            resume_from: Explicit checkpoint path to resume from
            track_svd_every: Snapshot SVD every N epochs (0=disabled, E6)
            track_grad_every: Snapshot gradient projection every N epochs (0=disabled, E4)
            track_betti_every: Evaluate Betti numbers every N epochs (0=disabled, E2/E3)
        """
        self._clear_interrupt()
        epochs = n_epochs or self.cfg.n_epochs

        # Resume from checkpoint
        start_epoch = 0
        if resume or resume_from:
            info = self.load_checkpoint(resume_from) if resume_from else {}
            if not resume_from:
                start_epoch = self.resume()
            else:
                start_epoch = info.get("epoch", 0)
            if start_epoch > 0:
                self.logger.info(f"Resumed from epoch {start_epoch}")
                # Restore tracking state if available
                state = torch.load(resume_from or os.path.join(
                    self.cfg.checkpoint_dir, f"{self._ckpt_prefix}_latest.pt"),
                    map_location=self.device, weights_only=False)
                if "svd_snapshots" in state:
                    self._svd_snapshots = state["svd_snapshots"]
                if "grad_projection_history" in state:
                    self._grad_projection_history = state["grad_projection_history"]
                if "betti_history" in state:
                    self._betti_history = state["betti_history"]

        best_val_mse = float("inf")
        self.enable_tracking()

        for epoch in range(start_epoch, epochs):
            self._current_epoch = epoch
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) - "
                f"G_loss: {train_metrics['G_loss']:.4f}, "
                f"D_loss: {train_metrics.get('D_loss', 0):.4f}, "
                f"G_l1: {train_metrics['G_l1']:.4f}"
            )

            # Evaluate
            if (epoch + 1) % self.cfg.eval_every == 0:
                val_metrics = self.evaluate(val_loader)
                if val_metrics["val_mse"] < best_val_mse:
                    best_val_mse = val_metrics["val_mse"]
                    self.save_checkpoint(epoch=epoch + 1, best=True)
                    self.logger.info(f"New best model! val_mse={best_val_mse:.6f}")

            # Periodic spectral tracking
            if track_svd_every > 0 and (epoch + 1) % track_svd_every == 0:
                self.snapshot_svd()

            if track_grad_every > 0 and (epoch + 1) % track_grad_every == 0:
                batch = next(iter(train_loader))
                self.snapshot_gradient_projection(batch)

            if track_betti_every > 0 and (epoch + 1) % track_betti_every == 0:
                self.snapshot_betti(val_loader)

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch=epoch + 1)

            # Handle graceful shutdown
            if self._should_stop():
                self.logger.info(
                    f"Interrupt at epoch {epoch+1}. State saved to latest.pt. "
                    f"Resume with: train(resume=True)"
                )
                break

        self.logger.info("Training completed.")
        return self.generator
