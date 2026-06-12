"""Physics constraint losses for topology optimization.

Computes FEM compliance, volume fraction error, stress/strain field errors
using Solidspy (wrapped in fem_solver.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VolumeFractionLoss(nn.Module):
    """Volume Fraction Absolute Error (VFAE).

    L_vf = |sum(fake) - sum(real)| / N
    """

    def __init__(self):
        super().__init__()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        vf_fake = fake.sum(dim=(1, 2, 3))
        vf_real = real.sum(dim=(1, 2, 3))
        N = fake[0].numel()
        return (vf_fake - vf_real).abs().mean() / N


class SurrogateComplianceLoss(nn.Module):
    """Differentiable SIMP-based compliance surrogate.

    Uses a density-stiffness interpolation (SIMP with p=3):
      C ≈ ∫_Ω 1/E(ρ) dΩ  where E(ρ) = E_min + (E₀-E_min)·ρ^p

    This is a smooth, differentiable approximation suitable for backpropagation.
    Exact FEM compliance is computed offline by FEMValidator for evaluation.
    NOT a replacement for true FEM — the surrogate only captures the
    qualitative relationship between density and compliance.
    """

    def __init__(self, penalty_factor: float = 1.0):
        super().__init__()
        self.penalty_factor = penalty_factor

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        bc: torch.Tensor,
        load_x: torch.Tensor,
        load_y: torch.Tensor,
    ) -> torch.Tensor:
        # Differentiable compliance surrogate:
        # C ≈ ∫_Ω σ_ij ε_ij dΩ = ∫_Ω (density^p) * σ0_ij ε0_ij dΩ
        # For a linear elastic material with SIMP interpolation:
        # C ∝ ∫_Ω (E_min + (E_0 - E_min) * ρ^p) * ε0^2 dΩ
        eps = 1e-8
        p = 3.0  # SIMP penalty exponent
        E_min = 1e-9
        density = fake.view(fake.size(0), -1)
        stiff = E_min + (1.0 - E_min) * (density ** p + eps)
        compliance_surrogate = (1.0 / stiff).mean(dim=1)

        density_real = real.view(real.size(0), -1)
        stiff_real = E_min + (1.0 - E_min) * (density_real ** p + eps)
        compliance_real = (1.0 / stiff_real).mean(dim=1)

        rel_error = ((compliance_surrogate - compliance_real) / (compliance_real + eps)).abs()
        return self.penalty_factor * rel_error.mean()


class StressFieldLoss(nn.Module):
    """Approximate stress field discrepancy.

    Uses density-weighted stress approximation:
    σ_vm ≈ (1/ρ) * constant for a given load case.
    """

    def __init__(self):
        super().__init__()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Von Mises stress is inversely proportional to density for fixed load
        # Higher density → lower stress → stiffer structure
        fake_stress = 1.0 / (fake + 1e-8)
        real_stress = 1.0 / (real + 1e-8)
        return F.mse_loss(fake_stress, real_stress)


class PhysicsConstraintLoss(nn.Module):
    """Combined physics constraint loss.

    L_phys = λ_vf * L_vf + λ_comp * L_comp + λ_stress * L_stress
    """

    def __init__(
        self,
        lambda_vf: float = 1.0,
        lambda_comp: float = 100.0,
        lambda_stress: float = 0.0,
    ):
        super().__init__()
        self.lambda_vf = lambda_vf
        self.lambda_comp = lambda_comp
        self.lambda_stress = lambda_stress
        self.vf_loss = VolumeFractionLoss()
        self.comp_loss = SurrogateComplianceLoss()
        self.stress_loss = StressFieldLoss()

    def forward(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
        bc: Optional[torch.Tensor] = None,
        load_x: Optional[torch.Tensor] = None,
        load_y: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}
        total = torch.tensor(0.0, device=fake.device)

        if self.lambda_vf > 0:
            l_vf = self.vf_loss(fake, real)
            losses["vf_loss"] = self.lambda_vf * l_vf
            total = total + losses["vf_loss"]

        if self.lambda_comp > 0 and bc is not None and load_x is not None and load_y is not None:
            l_comp = self.comp_loss(fake, real, bc, load_x, load_y)
            losses["comp_loss"] = self.lambda_comp * l_comp
            total = total + losses["comp_loss"]

        if self.lambda_stress > 0:
            l_stress = self.stress_loss(fake, real)
            losses["stress_loss"] = self.lambda_stress * l_stress
            total = total + losses["stress_loss"]

        losses["total"] = total
        return losses
