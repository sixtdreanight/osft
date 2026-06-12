"""Evaluation metrics for topology optimization GAN.

Image quality: MSE, MAE, SSIM, PSNR, LPIPS, IoU
Physics fidelity: VFAE, Compliance Error, Stress Field MSE, Displacement MSE
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from math import log10


def compute_mse(fake: torch.Tensor, real: torch.Tensor) -> float:
    """Mean Squared Error."""
    return F.mse_loss(fake, real).item()


def compute_mae(fake: torch.Tensor, real: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return F.l1_loss(fake, real).item()


def compute_psnr(fake: torch.Tensor, real: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(fake, real).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * log10(max_val ** 2 / mse)


def compute_ssim(
    fake: torch.Tensor,
    real: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """Structural Similarity Index (SSIM).

    Implementation using 11x11 Gaussian window.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=fake.dtype, device=fake.device)
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window @ _1d_window.T
    window = _2d_window.expand(1, 1, window_size, window_size).contiguous()

    # Channel-wise SSIM (handle multi-channel by groups=channels)
    C = fake.size(1)
    window_batch = window.expand(C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(fake, window_batch, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(real, window_batch, padding=window_size // 2, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(fake * fake, window_batch, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(real * real, window_batch, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(fake * real, window_batch, padding=window_size // 2, groups=C) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_iou(fake: torch.Tensor, real: torch.Tensor, threshold: float = 0.5) -> float:
    """IoU after binarization at threshold."""
    fake_bin = (fake > threshold).float()
    real_bin = (real > threshold).float()
    intersection = (fake_bin * real_bin).sum()
    union = ((fake_bin + real_bin) > 0).float().sum()
    if union < 1e-8:
        return 1.0
    return (intersection / union).item()


def compute_lpips(fake: torch.Tensor, real: torch.Tensor) -> float:
    """LPIPS using SqueezeNet features (lightweight substitute for AlexNet LPIPS).

    For production use, install 'lpips' package.
    This is a simplified feature-based distance.
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="squeeze", verbose=False).to(fake.device)
        # LPIPS expects 3-channel images
        if fake.size(1) == 1:
            fake_rgb = fake.repeat(1, 3, 1, 1)
            real_rgb = real.repeat(1, 3, 1, 1)
        else:
            fake_rgb = fake
            real_rgb = real
        return loss_fn(fake_rgb, real_rgb).mean().item()
    except ImportError:
        # Fallback: compute feature-based distance using simple conv features
        return _simple_feature_distance(fake, real)


def _simple_feature_distance(fake: torch.Tensor, real: torch.Tensor) -> float:
    """Simple feature-based distance using Sobel-like gradient features."""
    C = fake.size(1)
    # Sobel kernels for edge detection (per-channel)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=fake.device).view(1, 1, 3, 3)
    sobel_x = sobel_x.expand(C, 1, 3, 3).contiguous()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=fake.device).view(1, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3).contiguous()

    def grad_features(x):
        gx = F.conv2d(x, sobel_x, padding=1, groups=C)
        gy = F.conv2d(x, sobel_y, padding=1, groups=C)
        return torch.cat([gx, gy], dim=1)

    feat_fake = grad_features(fake)
    feat_real = grad_features(real)
    return F.mse_loss(feat_fake, feat_real).item()


def compute_vfae(fake: torch.Tensor, real: torch.Tensor) -> float:
    """Volume Fraction Absolute Error."""
    vf_fake = fake.sum(dim=(1, 2, 3))
    vf_real = real.sum(dim=(1, 2, 3))
    N = fake[0].numel()
    return (vf_fake - vf_real).abs().mean().item() / N


def compute_all_image_metrics(fake: torch.Tensor, real: torch.Tensor) -> Dict[str, float]:
    """Compute all image quality metrics."""
    return {
        "mse": compute_mse(fake, real),
        "mae": compute_mae(fake, real),
        "psnr": compute_psnr(fake, real),
        "ssim": compute_ssim(fake, real),
        "iou": compute_iou(fake, real),
        "lpips": compute_lpips(fake, real),
        "vfae": compute_vfae(fake, real),
    }


def evaluate_model(
    generator: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate generator on a full dataloader.

    Returns averaged metrics over all samples.
    """
    generator.eval()
    all_metrics = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0,
                   "iou": 0.0, "lpips": 0.0, "vfae": 0.0}
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if n_batches and i >= n_batches:
                break
            real_A, real_B = batch[0].to(device), batch[1].to(device)
            fake_B = generator(real_A)

            metrics = compute_all_image_metrics(fake_B, real_B)
            batch_size = real_A.size(0)
            for k in all_metrics:
                all_metrics[k] += metrics[k] * batch_size
            total_samples += batch_size

    for k in all_metrics:
        all_metrics[k] /= max(total_samples, 1)

    return all_metrics
