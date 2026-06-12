"""Tests for data loader and dataset creation."""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from utils.data_loader import TopologyDataset, create_dataloaders


def _make_synthetic_npy(path: str, n_samples: int = 200, height: int = 8, width: int = 16):
    """Create a synthetic .npy file in TopologyGAN format."""
    SN = height * width
    cols = 7 * SN
    data = np.random.rand(n_samples, cols).astype(np.float32)
    np.save(path, data)
    return path


class TestTopologyDataset:

    @pytest.fixture
    def tmp_data_path(self, tmp_path):
        path = str(tmp_path / "data.npy")
        return _make_synthetic_npy(path, n_samples=200, height=8, width=16)

    def test_dataset_lengths(self, tmp_data_path):
        train_ds = TopologyDataset(tmp_data_path, height=8, width=16, split="train")
        val_ds = TopologyDataset(tmp_data_path, height=8, width=16, split="val")
        test_ds = TopologyDataset(tmp_data_path, height=8, width=16, split="test")
        assert len(train_ds) == 160  # 80% of 200
        assert len(val_ds) == 20     # 10% of 200
        assert len(test_ds) == 20    # 10% of 200

    def test_getitem_shape(self, tmp_data_path):
        ds = TopologyDataset(tmp_data_path, height=8, width=16, split="train")
        real_A, real_B, bc, load_x, load_y = ds[0]
        assert real_A.shape == (3, 8, 16)        # [3 channels, H, W]
        assert real_B.shape == (1, 8, 16)        # [1 channel, H, W]
        assert bc.shape == (1, 8, 16)
        assert load_x.shape == (1, 8, 16)
        assert load_y.shape == (1, 8, 16)

    def test_getitem_value_ranges(self, tmp_data_path):
        ds = TopologyDataset(tmp_data_path, height=8, width=16, split="train", normalize=True)
        real_A, real_B, bc, load_x, load_y = ds[0]
        # With normalization, values should be in [0, 1] range
        assert 0 <= real_A.min() <= real_A.max() <= 1
        assert 0 <= real_B.min() <= real_B.max() <= 1

    def test_no_normalization(self, tmp_data_path):
        ds = TopologyDataset(tmp_data_path, height=8, width=16, split="train", normalize=False)
        real_A, real_B, _, _, _ = ds[0]
        # Without normalization, random values can be outside [0, 1]
        # But they should be float32
        assert real_A.dtype == torch.float32

    def test_split_all(self, tmp_data_path):
        ds = TopologyDataset(tmp_data_path, height=8, width=16, split="all")
        assert len(ds) == 200

    def test_deterministic_split(self, tmp_data_path):
        ds1 = TopologyDataset(tmp_data_path, height=8, width=16, split="train", seed=42)
        ds2 = TopologyDataset(tmp_data_path, height=8, width=16, split="train", seed=42)
        # Same seed should produce same split
        assert torch.equal(ds1[0][0], ds2[0][0])

    def test_different_split_different_data(self, tmp_data_path):
        ds_train = TopologyDataset(tmp_data_path, height=8, width=16, split="train", seed=0)
        ds_test = TopologyDataset(tmp_data_path, height=8, width=16, split="test", seed=0)
        # Train and test should have different data
        assert not torch.equal(ds_train[0][0], ds_test[0][0])


class TestCreateDataLoaders:

    @pytest.fixture
    def tmp_data_path(self, tmp_path):
        path = str(tmp_path / "data.npy")
        return _make_synthetic_npy(path, n_samples=200, height=8, width=16)

    def test_returns_three_loaders(self, tmp_data_path):
        train, val, test = create_dataloaders(
            tmp_data_path, height=8, width=16, batch_size=16, num_workers=0,
        )
        assert len(train.dataset) > 0
        assert len(val.dataset) > 0
        assert len(test.dataset) > 0

    def test_batch_shapes(self, tmp_data_path):
        train, val, test = create_dataloaders(
            tmp_data_path, height=8, width=16, batch_size=8, num_workers=0,
        )
        batch = next(iter(train))
        real_A, real_B, bc, load_x, load_y = batch
        assert real_A.shape == (8, 3, 8, 16)
        assert real_B.shape == (8, 1, 8, 16)
        assert bc.shape == (8, 1, 8, 16)

    def test_drop_last_train_loader(self, tmp_data_path):
        """Train loader drops last incomplete batch; val/test don't."""
        # With 160 samples and batch=32, train has 5 full batches (no drop needed)
        # With batch=33, train has 4 full batches (last 28 dropped)
        train, _, _ = create_dataloaders(
            tmp_data_path, height=8, width=16, batch_size=33, num_workers=0,
        )
        n_batches = sum(1 for _ in train)
        assert n_batches == 160 // 33  # 4 full batches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
