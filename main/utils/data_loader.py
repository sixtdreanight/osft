"""Data loader for TopologyGAN datasets.

Supports:
  - Original .npy format from TopologyGAN (128x64 resolution)
  - Generated .npy datasets (configurable resolution)
  - CSV-based datasets

Data layout (original TopologyGAN format):
  Columns 0*SN ~ 1*SN: volume fraction (input)
  Columns 1*SN ~ 2*SN: VM stress (input)
  Columns 2*SN ~ 3*SN: strain energy (input)
  Columns 3*SN ~ 4*SN: output structure (ground truth)
  Columns 4*SN ~ 5*SN: boundary conditions (condition)
  Columns 5*SN ~ 6*SN: load X (condition)
  Columns 6*SN ~ 7*SN: load Y (condition)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
import os


class TopologyDataset(Dataset):
    """Topology optimization dataset.

    Data shape per sample: [total_cols] where total_cols = SN * channels.
    SN = height * width (nodes), SE = (height-1) * (width-1) (elements).

    Default layout (7 channels):
      0: VF       (volume fraction, input)       [SN]
      1: VM_stress (von Mises stress, input)      [SN]
      2: strain_energy (strain energy, input)     [SN]
      3: output_B  (topology density, target)     [SN]
      4: bc        (boundary conditions, condition) [SN]
      5: load_x    (X-direction load, condition)  [SN]
      6: load_y    (Y-direction load, condition)  [SN]
    """

    def __init__(
        self,
        data_path: str,
        height: int = 64,
        width: int = 128,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        normalize: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.SN = height * width
        self.SE = (height - 1) * (width - 1)
        self.normalize = normalize

        data = np.load(data_path).astype(np.float32)
        if data.ndim == 2:
            total = data.shape[0]
        else:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        # Split deterministically
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        if split == "train":
            indices = indices[:n_train]
        elif split == "val":
            indices = indices[n_train:n_train + n_val]
        elif split == "test":
            indices = indices[n_train + n_val:]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = data[indices]

        # Compute normalization stats on training data
        if normalize:
            self._compute_stats()

    def _compute_stats(self):
        """Compute per-channel min/max for [0,1] normalization."""
        self.channel_min = []
        self.channel_max = []
        for c in range(min(7, self.data.shape[1] // self.SN)):
            col_data = self.data[:, c * self.SN:(c + 1) * self.SN]
            self.channel_min.append(col_data.min())
            self.channel_max.append(col_data.max() + 1e-8)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.data[idx]
        SN = self.SN
        H, W = self.height, self.width

        # Extract channels
        vf = row[0 * SN:1 * SN].reshape(H, W)
        vm_stress = row[1 * SN:2 * SN].reshape(H, W)
        strain_energy = row[2 * SN:3 * SN].reshape(H, W)
        output_b = row[3 * SN:4 * SN].reshape(H, W)
        bc = row[4 * SN:5 * SN].reshape(H, W)
        load_x = row[5 * SN:6 * SN].reshape(H, W)
        load_y = row[6 * SN:7 * SN].reshape(H, W)

        # Stack input channels: [3, H, W]
        real_A = np.stack([vf, vm_stress, strain_energy], axis=0)
        real_B = output_b[np.newaxis, :, :]  # [1, H, W]
        bc = bc[np.newaxis, :, :]
        load_x = load_x[np.newaxis, :, :]
        load_y = load_y[np.newaxis, :, :]

        if self.normalize:
            for c in range(3):
                real_A[c] = (real_A[c] - self.channel_min[c]) / (self.channel_max[c] - self.channel_min[c])
            real_B = (real_B - self.channel_min[3]) / (self.channel_max[3] - self.channel_min[3])
            # Condition channels (bc, load_x, load_y) are NOT normalized:
            # they encode physical BCs (discrete flags, force magnitudes) whose
            # numerical values must be preserved for FEM validation.

        return (
            torch.from_numpy(real_A),
            torch.from_numpy(real_B),
            torch.from_numpy(bc),
            torch.from_numpy(load_x),
            torch.from_numpy(load_y),
        )


def create_dataloaders(
    data_path: str,
    height: int = 64,
    width: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    train_ds = TopologyDataset(data_path, height, width, split="train",
                               train_ratio=train_ratio, val_ratio=val_ratio)
    val_ds = TopologyDataset(data_path, height, width, split="val",
                             train_ratio=train_ratio, val_ratio=val_ratio)
    test_ds = TopologyDataset(data_path, height, width, split="test",
                              train_ratio=train_ratio, val_ratio=val_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader
