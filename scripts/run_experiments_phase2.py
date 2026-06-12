#!/usr/bin/env python3
"""
Phase 2 Experiments: E3 (Betti ablation), E11 (FEM validation), E_Jac (Jacobian rank),
FIX1 (layered tau), FIX2 (gradient orthogonal projection).

Usage:
  python scripts/run_experiments_phase2.py --exp E3
  python scripts/run_experiments_phase2.py --exp E11
  python scripts/run_experiments_phase2.py --exp E_Jac
  python scripts/run_experiments_phase2.py --exp FIX1
  python scripts/run_experiments_phase2.py --exp FIX2
  python scripts/run_experiments_phase2.py --exp all
"""

import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from main.model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from main.model.physics_loss import PhysicsConstraintLoss
from main.osft.config import OSFTConfig
from main.osft.decomposer import SVDWeightDecomposer
from main.osft.subspace_layers import apply_osft_to_generator
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.eval.metrics import evaluate_model, compute_all_image_metrics
from main.eval.fem_validator import FEMValidator
from main.eval.spectral import (
    SingularValueAnalyzer, TopologyFeatureAnalyzer, JacobianAnalyzer,
)
from main.utils.data_loader import create_dataloaders
from main.utils.logger import ExperimentLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = "data/synthetic_train.npy"
PRETRAINED = "checkpoints/quickstart/pretrained_generator.pt"
RESULTS = "results/phase2"
os.makedirs(RESULTS, exist_ok=True)


# ============================================================
# Shared
# ============================================================

def load_pretrained(path=PRETRAINED):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    gen.eval()
    return gen


def make_loaders():
    return create_dataloaders(DATA, height=64, width=128, batch_size=16, num_workers=0)


# ============================================================
# E3: Betti Number Ablation — "Which subspace carries topology?"
# ============================================================

def run_E3():
    """E3: Generate images from W_full, W_main, W_res and compare Betti numbers.

    Validates Claim 1: principal subspace preserves topology.
    """
    logger = ExperimentLogger(RESULTS, "E3_betti_ablation")
    logger.info("E3: Betti Number Subspace Ablation")

    gen = load_pretrained()
    _, _, test_loader = make_loaders()

    # SVD decomposition
    decomposer = SVDWeightDecomposer(energy_threshold=0.80)
    decomposer.decompose_model(gen, verbose=True)

    # Collect fixed noise z for fair comparison
    torch.manual_seed(42)
    z_list = []
    conditions_list = []
    for batch in test_loader:
        cond = batch[0].to(device)
        conditions_list.append(cond)
        noise = torch.randn(cond.size(0), 100, device=device)
        z_list.append(noise)
        if len(conditions_list) * cond.size(0) >= 64:
            break
    conditions = torch.cat(conditions_list, dim=0)[:64]
    z = torch.cat(z_list, dim=0)[:64]

    # --- 1. Full weights (baseline) ---
    gen_full = load_pretrained()
    gen_full.eval()
    nz = gen_full.nz  # noise dimension
    z = torch.randn(conditions.size(0), nz, device=device)
    with torch.no_grad():
        out_full = gen_full(conditions, z=z)

    # --- 2. Principal subspace only (Wr) ---
    gen_main = load_pretrained()
    gen_main.eval()
    for name, module in gen_main.named_modules():
        if name in decomposer.results and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            Wr = decomposer.results[name]["Wr"]
            with torch.no_grad():
                module.weight.data = Wr.view(module.weight.shape)
    with torch.no_grad():
        out_main = gen_main(conditions, z=z)

    # --- 3. Residual subspace only (dW) ---
    gen_res = load_pretrained()
    gen_res.eval()
    for name, module in gen_res.named_modules():
        if name in decomposer.results and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            dW = decomposer.results[name]["dW"]
            with torch.no_grad():
                module.weight.data = dW.view(module.weight.shape)
    with torch.no_grad():
        out_res = gen_res(conditions, z=z)

    # --- Compute Betti numbers ---
    full_betti = TopologyFeatureAnalyzer.batch_betti_analysis(out_full, out_full)
    main_vs_full = TopologyFeatureAnalyzer.batch_betti_analysis(out_main, out_full)
    res_vs_full = TopologyFeatureAnalyzer.batch_betti_analysis(out_res, out_full)

    results = {
        "full_betti": full_betti,
        "main_vs_full": main_vs_full,
        "res_vs_full": res_vs_full,
        "MSE_main_vs_full": float(nn.functional.mse_loss(out_main, out_full).item()),
        "MSE_res_vs_full": float(nn.functional.mse_loss(out_res, out_full).item()),
        "MSE_res_vs_main": float(nn.functional.mse_loss(out_res, out_main).item()),
    }

    # Also scan multiple tau values
    tau_results = []
    for tau in [0.10, 0.30, 0.50, 0.65, 0.80, 0.90, 0.99]:
        decomposer_tau = SVDWeightDecomposer(energy_threshold=tau)
        gen_tau_main = load_pretrained()
        gen_tau_main.eval()
        decomposer_tau.decompose_model(gen_tau_main, verbose=False)
        for name, module in gen_tau_main.named_modules():
            if name in decomposer_tau.results and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                Wr = decomposer_tau.results[name]["Wr"]
                module.weight.data = Wr.view(module.weight.shape)
        with torch.no_grad():
            out_tau = gen_tau_main(conditions, z=z)
        betti_tau = TopologyFeatureAnalyzer.batch_betti_analysis(out_tau, out_full)
        mse_tau = float(nn.functional.mse_loss(out_tau, out_full).item())
        tau_results.append({
            "tau": tau,
            "beta0_preservation": betti_tau["beta0_preservation"],
            "beta1_preservation": betti_tau["beta1_preservation"],
            "mse_vs_full": mse_tau,
        })
        del gen_tau_main
        torch.cuda.empty_cache()

    results["tau_scan"] = tau_results

    # Save
    logger.save_results_table(results, "E3_betti_ablation.json")
    _print_betti_results(results, tau_results)
    return results


def _print_betti_results(results, tau_results):
    print("\n" + "=" * 70)
    print("E3: Betti Number Subspace Ablation")
    print("=" * 70)
    print(f"\n{'Subspace':<20} {'β0 Pres.':>12} {'β1 Pres.':>12}")
    print("-" * 46)
    for key in ["full_betti", "main_vs_full", "res_vs_full"]:
        b = results[key]
        label = {"full_betti": "Full (self)",
                 "main_vs_full": "Main vs Full",
                 "res_vs_full": "Residual vs Full"}[key]
        print(f"{label:<20} {b['beta0_preservation']:>12.4f} {b['beta1_preservation']:>12.4f}")
    print(f"\nMSE Main vs Full: {results['MSE_main_vs_full']:.6f}")
    print(f"MSE Res vs Full:  {results['MSE_res_vs_full']:.6f}")
    print(f"MSE Res vs Main:  {results['MSE_res_vs_main']:.6f}")

    print(f"\n{'τ':>8} {'β0 Pres.':>12} {'β1 Pres.':>12} {'MSE':>12}")
    print("-" * 46)
    for r in tau_results:
        print(f"{r['tau']:>8.2f} {r['beta0_preservation']:>12.4f} "
              f"{r['beta1_preservation']:>12.4f} {r['mse_vs_full']:>12.6f}")
    print("=" * 70)


# ============================================================
# E11: FEM Physical Validation
# ============================================================

def run_E11():
    """E11: Compute real FEM compliance for generated topologies.

    Compare OSFT vs Full FT vs Pre-trained on physical fidelity.
    """
    logger = ExperimentLogger(RESULTS, "E11_fem_validation")
    logger.info("E11: FEM Physical Validation")

    train_loader, val_loader, test_loader = make_loaders()
    fem_val = FEMValidator(64, 128)

    # Load models
    models = {
        "Pre-trained": load_pretrained(),
        "Full FT": _load_best_gen("results/final/full_ft_Cantilever_S0/full_ft_best.pt"),
        "OSFT": _load_best_gen("results/final/osft_Cantilever_S0/osft_best.pt"),
    }

    results = {}
    for name, model in models.items():
        model.eval()
        logger.info(f"Validating {name}...")
        batch_metrics = []
        n_batches = 0
        max_samples = 32  # FEM is slow

        with torch.no_grad():
            for batch in test_loader:
                if n_batches * batch[0].size(0) >= max_samples:
                    break
                real_A = batch[0].to(device)
                real_B = batch[1].to(device)
                bc = batch[2].to(device)
                lx = batch[3].to(device)
                ly = batch[4].to(device)

                fake_B = model(real_A)
                m = fem_val.validate_batch(fake_B, real_B, bc, lx, ly, max_samples=4)
                batch_metrics.append(m)
                n_batches += 1

        # Aggregate
        agg = {}
        valid = [m for m in batch_metrics if not np.isnan(m.get("compliance_error", float("nan")))]
        if valid:
            for k in valid[0]:
                vals = [m[k] for m in valid if not np.isnan(m.get(k, float("nan")))]
                agg[k] = float(np.mean(vals))
        else:
            agg = {k: float("nan") for k in batch_metrics[0]} if batch_metrics else {}
        results[name] = agg
        logger.info(f"  {name}: compliance_error={agg.get('compliance_error', float('nan')):.4f}, "
                    f"vf_error={agg.get('vf_error', float('nan')):.4f}")

    logger.save_results_table(results, "E11_fem_validation.json")
    _print_fem_results(results)
    return results


def _load_best_gen(path):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    gen.eval()
    return gen


def _print_fem_results(results):
    print("\n" + "=" * 70)
    print("E11: FEM Physical Validation")
    print("=" * 70)
    print(f"{'Model':<15} {'Comp. Error':>14} {'VF Error':>12} {'Comp. Fake':>14} {'Comp. Real':>14}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<15} {m.get('compliance_error', float('nan')):>14.4f} "
              f"{m.get('vf_error', float('nan')):>12.4f} "
              f"{m.get('compliance_fake', float('nan')):>14.2f} "
              f"{m.get('compliance_real', float('nan')):>14.2f}")
    print("=" * 70)


# ============================================================
# E_Jac: Jacobian Effective Rank — Diversity Preservation
# ============================================================

def run_E_Jac():
    """E_Jac: Compare Jacobian manifold dimension across fine-tuning methods.

    Validates Claim 3: OSFT preserves generative diversity.
    """
    logger = ExperimentLogger(RESULTS, "E_Jac_jacobian")
    logger.info("E_Jac: Jacobian Manifold Dimension")

    _, _, test_loader = make_loaders()

    models = {
        "Pre-trained": load_pretrained(),
        "Full FT": _load_best_gen("results/final/full_ft_Cantilever_S0/full_ft_best.pt"),
        "OSFT": _load_best_gen("results/final/osft_Cantilever_S0/osft_best.pt"),
    }

    results = {}
    for name, model in models.items():
        model.eval()
        logger.info(f"Analyzing {name}...")
        ja = JacobianAnalyzer(model)
        dim = ja.manifold_dimension_metrics(test_loader, device, n_samples=32)
        results[name] = dim
        logger.info(f"  Eff.Rank={dim['effective_rank']:.1f}, "
                    f"Stable Rank={dim['stable_rank']:.1f}, "
                    f"Part.Ratio={dim['participation_ratio']:.1f}")

    logger.save_results_table(results, "E_Jac_jacobian.json")
    _print_jacobian_results(results)
    return results


def _print_jacobian_results(results):
    print("\n" + "=" * 70)
    print("E_Jac: Jacobian Manifold Dimension")
    print("=" * 70)
    print(f"{'Model':<15} {'Eff. Rank':>12} {'Stable Rank':>14} {'Part. Ratio':>14}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<15} {m['effective_rank']:>12.1f} "
              f"{m['stable_rank']:>14.1f} {m['participation_ratio']:>14.1f}")
    print("=" * 70)


# ============================================================
# FIX1: Layered τ Strategy — Recover β1
# ============================================================

def run_FIX1():
    """FIX1: Use different tau for encoder vs decoder layers.

    Encoder: high tau (strong protection). Decoder: low tau (preserve holes).
    """
    logger = ExperimentLogger(RESULTS, "FIX1_layered_tau")
    logger.info("FIX1: Layered τ Strategy")

    train_loader, val_loader, test_loader = make_loaders()

    configs = {
        "baseline τ=0.80": {"energy_threshold": 0.80, "target_layers": None},
        "uniform τ=0.30": {"energy_threshold": 0.30, "target_layers": None},
        "layered τ (enc=0.85, dec=0.30)": {
            "energy_threshold": 0.85,
            "target_layers": ["e1", "e2", "e3"],
            "dec_threshold": 0.30,
            "dec_layers": ["d1", "d2", "d3", "d4", "d5", "d6"],
        },
    }

    results = {}
    torch.manual_seed(42)
    np.random.seed(42)

    for cfg_name, cfg_params in configs.items():
        logger.info(f"\n--- {cfg_name} ---")
        gen = load_pretrained()

        if "dec_threshold" in cfg_params:
            # Layered tau: different thresholds for encoder/decoder
            decomposer_enc = SVDWeightDecomposer(energy_threshold=cfg_params["energy_threshold"])
            decomposer_enc.decompose_model(gen, target_layers=cfg_params["target_layers"],
                                           verbose=False)
            decomposer_dec = SVDWeightDecomposer(energy_threshold=cfg_params["dec_threshold"])
            decomposer_dec.decompose_model(gen, target_layers=cfg_params["dec_layers"],
                                           verbose=False)
            # Merge: use enc results for enc layers, dec results for dec layers
            merged = {}
            merged.update(decomposer_enc.results)
            merged.update(decomposer_dec.results)
            apply_osft_to_generator(gen, merged)
            # Summary
            enc_summary = decomposer_enc.summary()
            dec_summary = decomposer_dec.summary()
            logger.info(f"  Encoder (τ={cfg_params['energy_threshold']}): "
                        f"rank={enc_summary.get('avg_rank_ratio', 0):.2%}")
            logger.info(f"  Decoder (τ={cfg_params['dec_threshold']}): "
                        f"rank={dec_summary.get('avg_rank_ratio', 0):.2%}")
        else:
            decomposer = SVDWeightDecomposer(energy_threshold=cfg_params["energy_threshold"])
            decomposer.decompose_model(gen, target_layers=cfg_params["target_layers"],
                                       verbose=False)
            apply_osft_to_generator(gen, decomposer.results)

        trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        total = sum(p.numel() for p in gen.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        # Train
        os_cfg = OSFTConfig(
            n_epochs=50, checkpoint_dir=f"{RESULTS}/FIX1_{cfg_name.replace(' ', '_')}",
            lr=1e-4, eval_every=5, save_every=999,  # only save best
        )
        trainer = OSFTTrainer(os_cfg, generator=gen)
        trainer.g_optimizer = optim.Adam(
            [p for p in gen.parameters() if p.requires_grad],
            lr=os_cfg.lr, betas=(os_cfg.beta1, os_cfg.beta2))
        trainer.train(train_loader, val_loader, n_epochs=50)

        metrics = evaluate_model(gen, test_loader, device)
        # Compute Betti
        betti = _eval_betti(gen, test_loader, device, max_batches=10)
        metrics.update(betti)
        metrics["trainable_params"] = trainable
        metrics["trainable_pct"] = 100 * trainable / total
        results[cfg_name] = metrics
        logger.info(f"  MSE={metrics['mse']:.4f}, SSIM={metrics['ssim']:.4f}, "
                    f"β0={betti['beta0_preservation']:.3f}, β1={betti['beta1_preservation']:.3f}")

    logger.save_results_table(results, "FIX1_layered_tau.json")
    _print_fix1_results(results)
    return results


def _print_fix1_results(results):
    print("\n" + "=" * 80)
    print("FIX1: Layered τ Strategy")
    print("=" * 80)
    header = f"{'Config':<30} {'MSE':>8} {'SSIM':>8} {'IOU':>8} {'β0':>8} {'β1':>8} {'Train%':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<30} {m['mse']:>8.4f} {m['ssim']:>8.4f} {m['iou']:>8.4f} "
              f"{m['beta0_preservation']:>8.3f} {m['beta1_preservation']:>8.3f} "
              f"{m['trainable_pct']:>7.1f}%")
    print("=" * 80)


def _eval_betti(model, dataloader, device, max_batches=10):
    model.eval()
    all_b0, all_b1 = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ra, rb = batch[0].to(device), batch[1].to(device)
            fb = model(ra)
            b = TopologyFeatureAnalyzer.batch_betti_analysis(fb, rb)
            all_b0.append(b["beta0_preservation"])
            all_b1.append(b["beta1_preservation"])
    return {"beta0_preservation": float(np.mean(all_b0)),
            "beta1_preservation": float(np.mean(all_b1))}


# ============================================================
# FIX2: Gradient Orthogonal Projection
# ============================================================

def run_FIX2():
    """FIX2: Explicitly project full gradient onto residual subspace before update.

    Instead of relying on optimizer to clip updates to dW, actively project
    the full gradient G_phy → G_projected = G_phy - U @ U^T @ G_phy @ V @ V^T,
    then update dW.
    """
    logger = ExperimentLogger(RESULTS, "FIX2_grad_projection")
    logger.info("FIX2: Gradient Orthogonal Projection OSFT")

    train_loader, val_loader, test_loader = make_loaders()

    torch.manual_seed(42)
    np.random.seed(42)

    # --- Baseline: Standard OSFT ---
    logger.info("\n--- Standard OSFT ---")
    gen_base = load_pretrained()
    cfg_base = OSFTConfig(
        n_epochs=50, checkpoint_dir=f"{RESULTS}/FIX2_baseline",
        lr=1e-4, eval_every=5, save_every=999,
    )
    base_trainer = OSFTTrainer(cfg_base, generator=gen_base)
    base_trainer.apply_svd_decomposition()
    base_trainer.g_optimizer = optim.Adam(
        [p for p in gen_base.parameters() if p.requires_grad],
        lr=cfg_base.lr, betas=(cfg_base.beta1, cfg_base.beta2))
    base_trainer.train(train_loader, val_loader, n_epochs=50)
    base_metrics = evaluate_model(gen_base, test_loader, device)
    logger.info(f"  Baseline OSFT: MSE={base_metrics['mse']:.4f}, SSIM={base_metrics['ssim']:.4f}")

    # --- Gradient Projection OSFT ---
    logger.info("\n--- Gradient Projection OSFT ---")
    gen_proj = load_pretrained()
    cfg_proj = OSFTConfig(
        n_epochs=50, checkpoint_dir=f"{RESULTS}/FIX2_projection",
        lr=1e-4, eval_every=5, save_every=999,
    )
    proj_trainer = GradientProjectedOSFTTrainer(cfg_proj, generator=gen_proj)
    proj_trainer.apply_svd_decomposition()
    proj_trainer.g_optimizer = optim.Adam(
        [p for p in gen_proj.parameters() if p.requires_grad],
        lr=cfg_proj.lr, betas=(cfg_proj.beta1, cfg_proj.beta2))
    proj_trainer.train(train_loader, val_loader, n_epochs=50)
    proj_metrics = evaluate_model(gen_proj, test_loader, device)
    logger.info(f"  Projection OSFT: MSE={proj_metrics['mse']:.4f}, SSIM={proj_metrics['ssim']:.4f}")

    results = {
        "baseline_OSFT": base_metrics,
        "projection_OSFT": proj_metrics,
    }
    logger.save_results_table(results, "FIX2_grad_projection.json")
    _print_fix2_results(results)
    return results


class GradientProjectedOSFTTrainer(OSFTTrainer):
    """OSFT trainer with explicit gradient projection onto residual subspace.

    After computing the full gradient, projects it:
      G_proj = G - Ur @ Ur.T @ G @ Vr @ Vr.T

    This ensures ONLY residual subspace components receive gradient updates,
    regardless of which subspace the gradient naturally lies in.
    """

    def train_step(self, batch, train_d=True):
        real_A, real_B = batch[0].to(self.device), batch[1].to(self.device)
        bc = batch[2].to(self.device) if len(batch) > 2 else None
        load_x = batch[3].to(self.device) if len(batch) > 3 else None
        load_y = batch[4].to(self.device) if len(batch) > 4 else None

        from torch.amp import autocast, GradScaler
        if not hasattr(self, 'scaler'):
            self.scaler = GradScaler("cuda", enabled=self.cfg.use_amp)
        metrics = {}

        # --- Discriminator (unchanged) ---
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

        # --- Generator with gradient projection ---
        with autocast("cuda", enabled=self.cfg.use_amp):
            fake_B = self.generator(real_A)
            fake_AB = torch.cat([real_A, bc, load_x, load_y, fake_B], dim=1)
            _, d_fake = self.discriminator(fake_AB)
            g_gan = self.bce_loss(d_fake, torch.ones_like(d_fake))
            g_l1 = self.l1_loss(fake_B, real_B)
            phys = self.physics_loss(fake_B, real_B, bc, load_x, load_y)
            g_loss = self.cfg.lambda_gan * g_gan + self.cfg.lambda_l1 * g_l1 + phys["total"]

        self.g_optimizer.zero_grad()
        self.scaler.scale(g_loss).backward()

        # --- KEY: Project gradients onto residual subspace ---
        if hasattr(self, '_decomposer') and self._decomposer is not None:
            self._project_gradients_to_residual()

        self.scaler.step(self.g_optimizer)
        self.scaler.update()

        metrics.update({"G_loss": g_loss.item(), "G_gan": g_gan.item(), "G_l1": g_l1.item()})
        self.global_step += 1
        return metrics

    def _project_gradients_to_residual(self):
        """Project layer gradients onto residual subspace.

        For each decomposed layer with residual basis Unr, Vnr:
          grad_2d ← grad_2d - Ur @ Ur.T @ grad_2d @ Vr @ Vr.T
        which is equivalent to:
          grad_proj ← (I - Ur@Ur.T) @ grad_2d @ (I - Vr@Vr.T) + Ur@Ur.T @ grad_2d @ Vr@Vr.T
        Wait, we want to REMOVE principal component, so:
          grad_res = grad - (Ur @ Ur.T @ grad @ Vr @ Vr.T)
        """
        from main.osft.subspace_layers import (
            OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d,
        )
        _OSFT_TYPES = (OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d)

        if not hasattr(self, '_decomposer'):
            return

        for name, module in self.generator.named_modules():
            if not isinstance(module, _OSFT_TYPES) or name not in self._decomposer.results:
                continue
            if not hasattr(module, '_weight_ref') or module._weight_ref.grad is None:
                continue

            d = self._decomposer.results[name]
            Ur = d["Ur"].to(self.device)  # [m, r]
            Vr = d["Vr"].to(self.device)  # [n, r]

            grad = module._weight_ref.grad  # [m, n] or similar
            orig_shape = grad.shape

            if grad.dim() == 4:  # Conv weight: [out, in, kh, kw]
                m, n_chan, kh, kw = grad.shape
                grad_2d = grad.view(m, n_chan * kh * kw)
            elif grad.dim() == 2:
                grad_2d = grad
            else:
                continue

            # Project: grad_res = grad - Ur @ Ur.T @ grad @ Vr @ Vr.T
            # This removes the principal subspace component
            grad_proj = Ur.T @ grad_2d @ Vr  # [r, r] — principal component
            grad_principal = Ur @ grad_proj @ Vr.T  # back to full space
            grad_residual = grad_2d - grad_principal  # only residual

            module._weight_ref.grad.copy_(grad_residual.view(orig_shape))


def _print_fix2_results(results):
    print("\n" + "=" * 70)
    print("FIX2: Gradient Orthogonal Projection")
    print("=" * 70)
    print(f"{'Method':<25} {'MSE':>10} {'SSIM':>8} {'IOU':>8} {'PSNR':>8}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<25} {m['mse']:>10.6f} {m['ssim']:>8.4f} {m['iou']:>8.4f} "
              f"{m.get('psnr', 0):>8.2f}")
    bas = results.get("baseline_OSFT", {})
    proj = results.get("projection_OSFT", {})
    if bas and proj:
        mse_delta = (bas["mse"] - proj["mse"]) / bas["mse"] * 100
        ssim_delta = (proj["ssim"] - bas["ssim"]) / (bas["ssim"] + 1e-8) * 100
        print(f"\nImprovement: MSE {mse_delta:+.1f}%, SSIM {ssim_delta:+.1f}%")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

EXP_MAP = {
    "E3": run_E3,
    "E11": run_E11,
    "E_Jac": run_E_Jac,
    "FIX1": run_FIX1,
    "FIX2": run_FIX2,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Experiments")
    parser.add_argument("--exp", type=str, required=True,
                        help="E3, E11, E_Jac, FIX1, FIX2, or all")
    args = parser.parse_args()

    if args.exp == "all":
        exps = list(EXP_MAP.keys())
    else:
        exps = [e.strip() for e in args.exp.split(",") if e.strip() in EXP_MAP]

    if not exps:
        print(f"Unknown experiment: {args.exp}")
        print(f"Available: {list(EXP_MAP.keys())}")
        sys.exit(1)

    for exp_name in exps:
        print(f"\n{'#' * 70}")
        print(f"# {exp_name}")
        print(f"{'#' * 70}")
        t0 = time.time()
        try:
            EXP_MAP[exp_name]()
            elapsed = (time.time() - t0) / 60
            print(f"{exp_name} completed in {elapsed:.1f} min")
        except Exception as e:
            print(f"{exp_name} FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
