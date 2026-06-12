#!/usr/bin/env python3
"""
Quickstart: end-to-end smoke test of the OSFT pipeline.

Runs:
  1. Pre-train a small TopologyGAN on synthetic data (few epochs)
  2. SVD decomposition + OSFT layer application
  3. Fine-tune with OSFT (few epochs)
  4. Evaluate and report metrics

Usage:
    python scripts/quickstart.py                          # full pipeline
    python scripts/quickstart.py --skip-pretrain           # skip pre-training
    python scripts/quickstart.py --resume                  # resume from checkpoint
"""

import sys
import os
import argparse
import time

# Add project root to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from main.model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from main.model.physics_loss import PhysicsConstraintLoss
from main.osft.config import OSFTConfig
from main.osft.decomposer import SVDWeightDecomposer
from main.osft.subspace_layers import apply_osft_to_generator
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="OSFT Quickstart Smoke Test")
    p.add_argument("--data", default="data/synthetic_train.npy")
    p.add_argument("--checkpoint-dir", default="checkpoints/quickstart")
    p.add_argument("--results-dir", default="results/quickstart")
    p.add_argument("--n-pretrain-epochs", type=int, default=30)
    p.add_argument("--n-ft-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-pretrain", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def pretrain(args, cfg, device):
    """Pre-train a TopologyGAN from scratch on synthetic data."""
    print("\n" + "=" * 60)
    print("Step 1: Pre-training TopologyGAN (from scratch)")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, height=cfg.img_height, width=cfg.img_width,
        batch_size=args.batch_size, num_workers=0,
    )

    gen = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=cfg.gf_dim,
        variant=cfg.generator_variant,
        height=cfg.img_height, width=cfg.img_width,
    ).to(device)

    disc = TopologyGANDiscriminator(
        condition_dim=cfg.condition_dim, output_c_dim=1, df_dim=cfg.df_dim,
        height=cfg.img_height, width=cfg.img_width,
    ).to(device)

    pretrain_cfg = OSFTConfig(
        n_epochs=args.n_pretrain_epochs,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(args.checkpoint_dir, "pretrain"),
        lr=1e-3,
        eval_every=5,
        save_every=10,
        device=args.device,
    )

    trainer = FullFinetuneTrainer(pretrain_cfg, generator=gen, discriminator=disc)
    trainer.train(train_loader, val_loader, n_epochs=args.n_pretrain_epochs,
                  resume=args.resume)

    # Save pre-trained weights
    pretrained_path = os.path.join(args.checkpoint_dir, "pretrained_generator.pt")
    torch.save(gen.state_dict(), pretrained_path)
    print(f"\nPre-trained generator saved → {pretrained_path}")

    return gen, pretrained_path, test_loader


def finetune_osft(args, cfg, pretrained_gen, device):
    """Apply OSFT and fine-tune."""
    print("\n" + "=" * 60)
    print("Step 2: OSFT Fine-Tuning")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data, height=cfg.img_height, width=cfg.img_width,
        batch_size=args.batch_size, num_workers=0,
    )

    ft_cfg = OSFTConfig(
        n_epochs=args.n_ft_epochs,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(args.checkpoint_dir, "osft"),
        lr=1e-4,
        eval_every=2,
        save_every=5,
        energy_threshold=0.80,
        lambda_gan=1.0,
        lambda_l1=100.0,
        lambda_orth=0.01,
        lambda_ksv=0.001,
        gf_dim=cfg.gf_dim,
        df_dim=cfg.df_dim,
        generator_variant=cfg.generator_variant,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        device=args.device,
    )

    # Use the pre-trained generator directly
    trainer = OSFTTrainer(ft_cfg, generator=pretrained_gen)

    # SVD decomposition
    decomposer = SVDWeightDecomposer(energy_threshold=ft_cfg.energy_threshold)
    decomposer.decompose_model(trainer.generator, verbose=True)
    summary = decomposer.summary()
    print(f"\nDecomposition summary: {summary}")

    apply_osft_to_generator(trainer.generator, decomposer.results)

    # Rebuild optimizer for OSFT trainable params
    import torch.optim as optim
    trainer.g_optimizer = optim.Adam(
        [p for p in trainer.generator.parameters() if p.requires_grad],
        lr=ft_cfg.lr, betas=(ft_cfg.beta1, ft_cfg.beta2),
    )

    # Fine-tune
    trainer.train(train_loader, val_loader, n_epochs=args.n_ft_epochs,
                  resume=args.resume)

    return trainer.generator, test_loader


def evaluate(args, generator, test_loader, device, stage: str):
    """Evaluate and print metrics."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation: {stage}")
    print(f"{'=' * 60}")

    metrics = evaluate_model(generator, test_loader, device)
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.6f}")

    trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total = sum(p.numel() for p in generator.parameters())
    print(f"\n  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.data}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = OSFTConfig(
        img_height=64,
        img_width=128,
        gf_dim=64,
        df_dim=16,
        batch_size=args.batch_size,
        generator_variant="unet",
        device=args.device,
    )

    t_start = time.time()

    # Step 1: Pre-train
    if args.skip_pretrain:
        pretrained_path = os.path.join(args.checkpoint_dir, "pretrained_generator.pt")
        if not os.path.exists(pretrained_path):
            print(f"[ERROR] Pre-trained model not found: {pretrained_path}")
            print("Run without --skip-pretrain first.")
            sys.exit(1)
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=cfg.gf_dim,
            variant=cfg.generator_variant,
            height=cfg.img_height, width=cfg.img_width,
        ).to(device)
        gen.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
        _, _, test_loader = create_dataloaders(
            args.data, height=cfg.img_height, width=cfg.img_width,
            batch_size=args.batch_size, num_workers=0,
        )
    else:
        gen, pretrained_path, test_loader = pretrain(args, cfg, device)

    # Step 2: Evaluate pre-trained
    pre_metrics = evaluate(args, gen, test_loader, device, "Pre-trained")

    # Step 3: OSFT fine-tune (pass pre-trained generator directly)
    gen_osft, test_loader = finetune_osft(args, cfg, gen, device)

    # Step 4: Evaluate OSFT
    osft_metrics = evaluate(args, gen_osft, test_loader, device, "OSFT Fine-tuned")

    # Summary
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"{'Metric':<10} {'Pre-trained':<16} {'OSFT':<16}")
    print("-" * 42)
    for k in pre_metrics:
        print(f"{k:<10} {pre_metrics[k]:<16.6f} {osft_metrics[k]:<16.6f}")


if __name__ == "__main__":
    main()
