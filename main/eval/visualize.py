"""Visualization utilities for topology optimization results.

Generates:
  - Density field comparison (fake vs real vs error map)
  - Stress/strain field visualizations
  - Training curves (loss, metrics)
  - Parameter efficiency plots
  - Ablation study heatmaps
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os
import torch


def set_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


def plot_topology_comparison(
    fake: np.ndarray,
    real: np.ndarray,
    save_path: str,
    title: str = "",
    vf_fake: Optional[float] = None,
    vf_real: Optional[float] = None,
):
    """Plot fake vs real topology with error map.

    Args:
        fake: [H, W] generated topology
        real: [H, W] ground truth topology
        save_path: Output file path
        title: Optional title
    """
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(fake, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[0].set_title(f"Generated{' (VF={vf_fake:.3f})' if vf_fake else ''}")
    axes[0].axis("off")

    axes[1].imshow(real, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[1].set_title(f"Ground Truth{' (VF={vf_real:.3f})' if vf_real else ''}")
    axes[1].axis("off")

    # Error map
    error = np.abs(fake - real)
    im = axes[2].imshow(error, cmap="hot", origin="lower")
    axes[2].set_title(f"|Error| (MSE={np.mean(error**2):.4f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Binary comparison (threshold at 0.5)
    fake_bin = fake > 0.5
    real_bin = real > 0.5
    diff = np.zeros_like(fake)
    diff[fake_bin & real_bin] = 0.5   # Agreement (gray)
    diff[fake_bin & ~real_bin] = 1.0   # False positive (white)
    diff[~fake_bin & real_bin] = 0.0   # False negative (black)
    axes[3].imshow(diff, cmap="gray", origin="lower")
    iou = np.sum(fake_bin & real_bin) / max(np.sum(fake_bin | real_bin), 1)
    axes[3].set_title(f"Binary (IoU={iou:.3f})")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(
    metrics_history: List[Dict],
    save_path: str,
    title: str = "Training Curves",
):
    """Plot training loss and metric curves.

    Args:
        metrics_history: List of per-epoch metric dicts
        save_path: Output file path
    """
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = [m.get("step", i) for i, m in enumerate(metrics_history)]

    # Loss
    g_loss = [m.get("train_G_loss", 0) for m in metrics_history]
    d_loss = [m.get("train_D_loss", 0) for m in metrics_history]
    axes[0, 0].plot(epochs, g_loss, label="Generator", linewidth=1)
    axes[0, 0].plot(epochs, d_loss, label="Discriminator", linewidth=1)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("GAN Losses")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # L1 + Physics
    g_l1 = [m.get("train_G_l1", 0) for m in metrics_history]
    g_comp = [m.get("train_G_comp", 0) for m in metrics_history]
    axes[0, 1].plot(epochs, g_l1, label="L1 Recon", linewidth=1)
    axes[0, 1].plot(epochs, g_comp, label="Compliance", linewidth=1)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Reconstruction & Physics Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Valuation metrics
    val_mse = [m.get("val_mse", 0) for m in metrics_history if "val_mse" in m]
    val_epochs = [m.get("step", i) for i, m in enumerate(metrics_history) if "val_mse" in m]
    axes[1, 0].plot(val_epochs, val_mse, "o-", markersize=3, linewidth=1)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Validation MSE")
    axes[1, 0].grid(True, alpha=0.3)

    # OSFT-specific
    g_orth = [m.get("train_G_orth", 0) for m in metrics_history]
    g_ksv = [m.get("train_G_ksv", 0) for m in metrics_history]
    axes[1, 1].plot(epochs, g_orth, label="L_orth", linewidth=1)
    axes[1, 1].plot(epochs, g_ksv, label="L_ksv", linewidth=1)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Subspace Regularization")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_parameter_efficiency_curve(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    x_metric: str = "trainable_pct",
    y_metric: str = "mse",
):
    """Plot parameter efficiency vs quality curve.

    Args:
        results: {method_name: {metric_name: value}}
        save_path: Output path
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = []
    xs = []
    ys = []
    for method, metrics in results.items():
        if x_metric in metrics and y_metric in metrics:
            methods.append(method)
            xs.append(metrics[x_metric])
            ys.append(metrics[y_metric])

    if xs:
        ax.scatter(xs, ys, s=80, alpha=0.8)
        for i, method in enumerate(methods):
            ax.annotate(method, (xs[i], ys[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Trainable Parameters (%)")
    ax.set_ylabel(y_metric.upper())
    ax.set_title("Parameter Efficiency vs Quality Trade-off")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_ablation_heatmap(
    ablation_data: Dict[str, Dict[str, float]],
    save_path: str,
    metrics: Optional[List[str]] = None,
):
    """Plot ablation study as a heatmap.

    Args:
        ablation_data: {variant_name: {metric_name: value}}
        save_path: Output path
        metrics: List of metrics to include
    """
    if metrics is None:
        metrics = ["mse", "ssim", "compliance_error"]

    set_style()
    variants = list(ablation_data.keys())
    data = np.array([[ablation_data[v].get(m, 0) for m in metrics]
                     for v in variants])

    fig, ax = plt.subplots(figsize=(len(metrics) * 2, len(variants) * 0.6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)

    # Annotate cells
    for i in range(len(variants)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax)
    ax.set_title("Ablation Study")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_convergence_curves(
    curves: Dict[str, List[float]],
    save_path: str,
    metric_name: str = "MSE",
):
    """Plot convergence curves for multiple methods.

    Args:
        curves: {method_name: [metric_values_per_epoch]}
        save_path: Output path
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"OSFT": "#1f77b4", "Full FT": "#ff7f0e", "LoRA": "#2ca02c", "Adapter": "#d62728"}

    for method, values in curves.items():
        color = colors.get(method, None)
        ax.plot(values, label=method, linewidth=1.5, color=color, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Convergence Analysis - Validation {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
