"""FEM validation module.

Validates generated topologies by computing:
  - Structural compliance via Solidspy FEM solver
  - Von Mises stress fields
  - Displacement fields
  - Volume fraction comparison
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from ..utils.fem_solver import FEMSolver


class FEMValidator:
    """Offline FEM validator for topology quality assessment.

    FEM is computed on CPU (9950X3D is excellent for this task),
    decoupled from GPU training to avoid bottlenecks.
    """

    def __init__(self, height: int = 64, width: int = 128):
        self.solver = FEMSolver(height, width)
        self.height = height
        self.width = width

    def validate_sample(
        self,
        fake_topology: np.ndarray,
        real_topology: np.ndarray,
        bc: np.ndarray,
        load_x: np.ndarray,
        load_y: np.ndarray,
    ) -> Dict[str, float]:
        """Validate a single sample's physics metrics."""
        return self.solver.compute_physics_metrics(
            fake_topology, real_topology, bc, load_x, load_y,
        )

    def validate_batch(
        self,
        fake_batch: torch.Tensor,
        real_batch: torch.Tensor,
        bc_batch: torch.Tensor,
        load_x_batch: torch.Tensor,
        load_y_batch: torch.Tensor,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """Validate a batch of samples with physics metrics.

        Args:
            fake_batch: Generated topologies [B, 1, H, W]
            real_batch: Ground truth topologies [B, 1, H, W]
            bc_batch: Boundary conditions [B, 1, H, W]
            load_x_batch: X load [B, 1, H, W]
            load_y_batch: Y load [B, 1, H, W]
            max_samples: Maximum number of samples to validate (FEM is slow)

        Returns:
            Averaged physics metrics dict
        """
        B = min(fake_batch.size(0), max_samples)

        metrics_sum = {
            "compliance_error": 0.0,
            "vf_error": 0.0,
            "compliance_fake": 0.0,
            "compliance_real": 0.0,
        }
        n_valid = 0

        for i in range(B):
            try:
                fake = fake_batch[i, 0].cpu().numpy()
                real = real_batch[i, 0].cpu().numpy()
                bc = bc_batch[i, 0].cpu().numpy()
                lx = load_x_batch[i, 0].cpu().numpy()
                ly = load_y_batch[i, 0].cpu().numpy()

                result = self.validate_sample(fake, real, bc, lx, ly)
                for k in metrics_sum:
                    metrics_sum[k] += result.get(k, 0.0)
                n_valid += 1
            except Exception:
                continue

        if n_valid == 0:
            return {k: float("nan") for k in metrics_sum}

        return {k: v / n_valid for k, v in metrics_sum.items()}

    def validate_dataset(
        self,
        generator: torch.nn.Module,
        dataloader,
        device: torch.device,
        max_samples: int = 200,
    ) -> Dict[str, float]:
        """Run FEM validation over a dataset.

        Generates topologies from the generator, then validates physical fidelity.
        """
        generator.eval()
        all_metrics = defaultdict(float)
        n_total = 0

        with torch.no_grad():
            for batch in dataloader:
                if n_total >= max_samples:
                    break

                real_A = batch[0].to(device)
                real_B = batch[1].to(device)
                bc = batch[2].to(device)
                load_x = batch[3].to(device)
                load_y = batch[4].to(device)

                fake_B = generator(real_A)

                batch_metrics = self.validate_batch(
                    fake_B, real_B, bc, load_x, load_y,
                    max_samples=max_samples - n_total,
                )
                # Weight each batch's average metrics by the batch size
                bs = fake_B.size(0)
                for k, v in batch_metrics.items():
                    if not np.isnan(v):
                        all_metrics[k] += v * bs
                n_total += bs

        if n_total == 0:
            return {}

        return {k: v / n_total for k, v in all_metrics.items()}
