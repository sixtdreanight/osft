#!/usr/bin/env python3
"""Generate synthetic topology optimization data for pipeline testing.

Output format: .npy file matching TopologyGAN's expected 7-channel layout.
Each sample is 7 * SN values, where SN = height * width.

Channels:
  0: VF            (volume fraction, input)
  1: VM_stress     (von Mises stress, input)
  2: strain_energy (strain energy, input)
  3: output_B      (topology density, target)
  4: bc            (boundary conditions)
  5: load_x        (X-direction load)
  6: load_y        (Y-direction load)
"""

import numpy as np
import argparse
import os


def generate_dataset(
    n_samples: int = 500,
    height: int = 64,
    width: int = 128,
    output_path: str = "data/synthetic_train.npy",
):
    SN = height * width
    total_cols = 7 * SN

    data = np.zeros((n_samples, total_cols), dtype=np.float32)

    for i in range(n_samples):
        # Channel 0-2: input conditions (VF + VM_stress + strain_energy)
        # Generate smooth random fields via low-frequency noise
        for c in range(3):
            field = _random_smooth_field(height, width)
            data[i, c * SN:(c + 1) * SN] = field.ravel()

        # Channel 3: output topology (target) — binarized smooth field
        output_field = _random_topology(height, width)
        data[i, 3 * SN:4 * SN] = output_field.ravel()

        # Channel 4: boundary conditions
        # Convention: 1=fixed X, 2=fixed Y, 3=fixed X+Y (fully clamped)
        bc_field = np.zeros((height, width), dtype=np.float32)
        bc_field[:, 0] = 3.0   # left edge: fully clamped (cantilever)
        data[i, 4 * SN:5 * SN] = bc_field.ravel()

        # Channel 5-6: loads (point load at right edge, center, downward)
        lx_field = np.zeros((height, width), dtype=np.float32)
        ly_field = np.zeros((height, width), dtype=np.float32)
        cx, cy = height // 2, width - 2  # near right edge, vertical center
        ly_field[cx, cy] = -1.0  # downward force
        data[i, 5 * SN:6 * SN] = lx_field.ravel()
        data[i, 6 * SN:7 * SN] = ly_field.ravel()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, data)
    print(f"Saved {n_samples} samples ({total_cols} cols) → {output_path}")
    print(f"  Height={height}, Width={width}, SN={SN}")
    print(f"  File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")


def _random_smooth_field(h: int, w: int) -> np.ndarray:
    """Generate a smooth random field via low-pass filtered noise."""
    # Create low-resolution random field and upsample
    lr_h, lr_w = max(h // 8, 2), max(w // 8, 2)
    lr = np.random.rand(lr_h, lr_w).astype(np.float32)
    from scipy.ndimage import zoom
    field = zoom(lr, (h / lr_h, w / lr_w), order=3)
    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field.astype(np.float32)


def _random_topology(h: int, w: int) -> np.ndarray:
    """Generate a plausible-looking binary topology field."""
    field = _random_smooth_field(h, w)
    # Threshold to create binary-like structure with smooth transitions
    threshold = np.random.uniform(0.3, 0.7)
    # Soft binarization
    steepness = 15.0
    binary = 1.0 / (1.0 + np.exp(-steepness * (field - threshold)))
    return binary.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic topology data")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples (default: 500)")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--output", type=str, default="data/synthetic_train.npy")
    args = parser.parse_args()

    generate_dataset(args.n_samples, args.height, args.width, args.output)
