#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthogonal Subspace Fine-Tuning (OSFT) — Entry Point.

Performs SVD decomposition of pre-trained TopologyGAN generator weights
and prepares the orthogonal subspace layers for efficient fine-tuning.

Usage:
    python main.py --pretrained path/to/checkpoint.pt
    python main.py --pretrained checkpoint.pt --threshold 0.85 --output results.pkl
"""

import argparse
import pickle
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from model.topologygan import TopologyGANGenerator
from osft.decomposer import SVDWeightDecomposer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSFT SVD Decomposition & Subspace Preparation"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pre-trained generator checkpoint (.pt)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Energy threshold for SVD truncation (default: 0.80)"
    )
    parser.add_argument(
        "--output", type=str, default="./svd_decomposition_results.pkl",
        help="Output path for decomposition results (default: ./svd_decomposition_results.pkl)"
    )
    parser.add_argument(
        "--variant", type=str, default="se_res_unet", choices=("unet", "se_res_unet"),
        help="Generator architecture variant (default: se_res_unet)"
    )
    parser.add_argument(
        "--img-height", type=int, default=64,
    )
    parser.add_argument(
        "--img-width", type=int, default=128,
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu"
    )
    parser.add_argument(
        "--target-layers", type=str, nargs="*",
        default=None,
        help="Target layer name patterns (default: all Conv/Linear layers)"
    )
    return parser.parse_args()


def load_pretrained(checkpoint_path: str, variant: str,
                    img_height: int, img_width: int, device: torch.device):
    """Load a pre-trained TopologyGAN generator."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    generator = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=128,
        variant=variant, height=img_height, width=img_width,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "generator_state_dict" in state:
        state = state["generator_state_dict"]
    generator.load_state_dict(state, strict=False)

    print(f"[Loaded] Pre-trained model from {checkpoint_path}")
    return generator


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cuda" if args.device == "cuda"
        else "cpu"
    )

    print("=" * 60)
    print("OSFT: Orthogonal Subspace Fine-Tuning — SVD Decomposition")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"Energy τ:    {args.threshold}")
    print(f"Variant:     {args.variant}")
    print(f"Resolution:  {args.img_height}×{args.img_width}")

    # Build or load generator
    if args.pretrained:
        generator = load_pretrained(
            args.pretrained, args.variant,
            args.img_height, args.img_width, device,
        )
    else:
        print("\n[Warning] No pre-trained checkpoint provided. "
              "Using randomly initialized generator for demo.\n")
        generator = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=128,
            variant=args.variant,
            height=args.img_height, width=args.img_width,
        ).to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator params: {total_params:,}")

    # SVD decomposition
    decomposer = SVDWeightDecomposer(energy_threshold=args.threshold)
    target = args.target_layers

    print(f"\n[SVD] Decomposing layers" +
          (f": {target}" if target else " (all Conv/Linear layers)"))

    decomposer.decompose_model(generator, target_layers=target)

    # Summary
    summary = decomposer.summary()
    print(f"\n[Summary]")
    print(f"  Layers decomposed:    {summary['n_layers']}")
    print(f"  Original params:      {summary['total_params_orig']:,}")
    print(f"  Residual dimensions:  {summary['total_residual_dim']:,}")
    print(f"  Avg energy retained:  {summary['avg_energy_retained']:.1%}")
    print(f"  Avg rank ratio:       {summary['avg_rank_ratio']:.1%}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    decomposer.save(args.output)
    print(f"\n[Saved] Results → {os.path.abspath(args.output)}")
    print("[Done]")

    return generator, decomposer


if __name__ == "__main__":
    main()
