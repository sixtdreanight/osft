"""Tests for evaluation metrics."""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from eval.metrics import (
    compute_mse, compute_mae, compute_psnr, compute_ssim,
    compute_iou, compute_vfae, compute_all_image_metrics,
    compute_lpips,
)


class TestBasicMetrics:

    def test_mse_identical(self):
        x = torch.randn(2, 1, 16, 16)
        assert compute_mse(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_mse_positive(self):
        x = torch.randn(2, 1, 16, 16)
        y = x + 0.1
        assert compute_mse(x, y) > 0

    def test_mae_identical(self):
        x = torch.randn(2, 1, 16, 16)
        assert compute_mae(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_psnr_identical(self):
        x = torch.randn(2, 1, 16, 16)
        assert compute_psnr(x, x) == float("inf")

    def test_psnr_lower_for_noisier(self):
        x = torch.ones(1, 1, 16, 16)
        y_small = x + 0.01
        y_large = x + 0.5
        assert compute_psnr(x, y_small) > compute_psnr(x, y_large)


class TestSSIM:

    def test_identical(self):
        x = torch.ones(1, 1, 32, 32)
        assert compute_ssim(x, x) == pytest.approx(1.0, abs=0.01)

    def test_opposite(self):
        x = torch.ones(1, 1, 32, 32)
        y = torch.zeros(1, 1, 32, 32)
        s = compute_ssim(x, y)
        assert s < 0.5

    def test_range_zero_to_one(self):
        torch.manual_seed(42)
        x = torch.rand(1, 1, 32, 32)
        y = torch.rand(1, 1, 32, 32)
        s = compute_ssim(x, y)
        assert 0.0 <= s <= 1.0

    def test_multi_channel(self):
        torch.manual_seed(42)
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        s = compute_ssim(x, y)
        assert 0.0 <= s <= 1.0

    def test_ssim_improves_with_closer_input(self):
        x = torch.ones(1, 1, 32, 32)
        y_near = x + 0.05 * torch.randn(1, 1, 32, 32)
        y_far = x + 0.5 * torch.randn(1, 1, 32, 32)
        s_near = compute_ssim(x, y_near)
        s_far = compute_ssim(x, y_far)
        assert s_near > s_far


class TestIoU:

    def test_identical(self):
        x = torch.ones(1, 1, 8, 8) * 0.8
        assert compute_iou(x, x) == pytest.approx(1.0)

    def test_non_overlapping(self):
        x = torch.ones(1, 1, 8, 8)
        y = torch.zeros(1, 1, 8, 8)
        # All 1s > 0.5 = all solid for x, all void for y → intersection = 0, union = all
        assert compute_iou(x, y) == pytest.approx(0.0)

    def test_high_threshold(self):
        x = torch.ones(1, 1, 8, 8) * 0.6
        y = torch.ones(1, 1, 8, 8) * 0.4
        # At threshold 0.5: x → all 1, y → all 0
        assert compute_iou(x, y, threshold=0.5) == pytest.approx(0.0)


class TestVFAE:

    def test_same_volume(self):
        x = torch.ones(2, 1, 4, 8)
        y = torch.ones(2, 1, 4, 8)
        assert compute_vfae(x, y) == pytest.approx(0.0, abs=1e-6)

    def test_different_volume(self):
        x = torch.ones(2, 1, 4, 8)
        y = torch.zeros(2, 1, 4, 8)
        vfae = compute_vfae(x, y)
        assert vfae > 0


class TestComputeAllImageMetrics:

    def test_returns_all_keys(self):
        x = torch.rand(2, 1, 16, 16)
        y = torch.rand(2, 1, 16, 16)
        result = compute_all_image_metrics(x, y)
        expected_keys = {"mse", "mae", "psnr", "ssim", "iou", "lpips", "vfae"}
        assert set(result.keys()) == expected_keys

    def test_identical_inputs_perfect_scores(self):
        x = torch.ones(2, 1, 16, 16)
        result = compute_all_image_metrics(x, x)
        assert result["mse"] == pytest.approx(0.0, abs=1e-6)
        assert result["mae"] == pytest.approx(0.0, abs=1e-6)
        assert result["ssim"] > 0.95
        assert result["iou"] == pytest.approx(1.0)
        assert result["vfae"] == pytest.approx(0.0, abs=1e-6)

    def test_handles_batch_first_dim(self):
        for b in [1, 4, 8]:
            x = torch.rand(b, 1, 16, 16)
            y = torch.rand(b, 1, 16, 16)
            result = compute_all_image_metrics(x, y)
            assert all(isinstance(v, float) for v in result.values())


class TestLPIPS:

    def test_fallback_feature_distance(self):
        """When lpips package is not installed, should use Sobel fallback."""
        x = torch.rand(2, 1, 32, 32)
        y = torch.rand(2, 1, 32, 32)
        result = compute_lpips(x, y)
        assert isinstance(result, float)
        assert result >= 0

    def test_identical_fallback(self):
        x = torch.ones(1, 1, 16, 16)
        d = compute_lpips(x, x)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_3channel_input(self):
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        d = compute_lpips(x, y)
        assert isinstance(d, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
