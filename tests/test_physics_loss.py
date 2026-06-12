"""Tests for physics constraint losses."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from model.physics_loss import (
    VolumeFractionLoss, SurrogateComplianceLoss,
    StressFieldLoss, PhysicsConstraintLoss,
)


class TestVolumeFractionLoss:

    @pytest.fixture
    def loss_fn(self):
        return VolumeFractionLoss()

    def test_identical_zero(self, loss_fn):
        x = torch.ones(2, 1, 16, 32)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_different_positive(self, loss_fn):
        x = torch.ones(2, 1, 16, 32)
        y = torch.zeros(2, 1, 16, 32)
        assert loss_fn(x, y).item() > 0

    def test_batch_averages_correctly(self, loss_fn):
        x = torch.rand(4, 1, 8, 8)
        y = torch.rand(4, 1, 8, 8)
        loss = loss_fn(x, y)
        assert loss.ndim == 0  # scalar


class TestSurrogateComplianceLoss:

    @pytest.fixture
    def loss_fn(self):
        return SurrogateComplianceLoss()

    def test_identical_density_zero_error(self, loss_fn):
        x = torch.rand(2, 1, 16, 32)
        bc = torch.rand(2, 1, 16, 32)
        lx = torch.rand(2, 1, 16, 32)
        ly = torch.rand(2, 1, 16, 32)
        loss = loss_fn(x, x, bc, lx, ly)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_output_is_scalar(self, loss_fn):
        x = torch.rand(2, 1, 16, 32)
        y = torch.rand(2, 1, 16, 32)
        bc = torch.rand(2, 1, 16, 32)
        lx = torch.rand(2, 1, 16, 32)
        ly = torch.rand(2, 1, 16, 32)
        loss = loss_fn(x, y, bc, lx, ly)
        assert loss.ndim == 0

    def test_penalty_scales_loss(self):
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        bc = torch.rand(2, 1, 8, 8)
        lx = torch.rand(2, 1, 8, 8)
        ly = torch.rand(2, 1, 8, 8)

        loss1 = SurrogateComplianceLoss(penalty_factor=1.0)(x, y, bc, lx, ly)
        loss10 = SurrogateComplianceLoss(penalty_factor=10.0)(x, y, bc, lx, ly)
        assert loss10.item() == pytest.approx(10.0 * loss1.item(), rel=1e-5)

    def test_numerically_stable(self, loss_fn):
        x = torch.rand(2, 1, 8, 8)
        y = torch.full((2, 1, 8, 8), 1e-10)
        bc = torch.rand(2, 1, 8, 8)
        lx = torch.rand(2, 1, 8, 8)
        ly = torch.rand(2, 1, 8, 8)
        loss = loss_fn(x, y, bc, lx, ly)
        assert not torch.isnan(loss) and not torch.isinf(loss)


class TestStressFieldLoss:

    @pytest.fixture
    def loss_fn(self):
        return StressFieldLoss()

    def test_identical_zero(self, loss_fn):
        x = torch.ones(2, 1, 16, 32)
        assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_different_positive(self, loss_fn):
        x = torch.ones(2, 1, 16, 32)
        y = torch.ones(2, 1, 16, 32) * 0.5
        assert loss_fn(x, y).item() > 0

    def test_numerically_stable_zero_density(self, loss_fn):
        x = torch.zeros(2, 1, 4, 4)
        y = torch.zeros(2, 1, 4, 4)
        loss = loss_fn(x, y)
        assert not torch.isnan(loss) and not torch.isinf(loss)


class TestPhysicsConstraintLoss:

    @pytest.fixture
    def loss_fn(self):
        return PhysicsConstraintLoss(lambda_vf=1.0, lambda_comp=100.0)

    def test_output_dict_keys(self, loss_fn):
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        bc = torch.rand(2, 1, 8, 8)
        lx = torch.rand(2, 1, 8, 8)
        ly = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y, bc, lx, ly)
        assert "total" in result
        assert "vf_loss" in result
        assert "comp_loss" in result

    def test_total_is_sum_of_parts(self, loss_fn):
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        bc = torch.rand(2, 1, 8, 8)
        lx = torch.rand(2, 1, 8, 8)
        ly = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y, bc, lx, ly)
        expected_total = result["vf_loss"] + result["comp_loss"]
        assert result["total"].item() == pytest.approx(expected_total.item(), abs=1e-5)

    def test_no_conditions_skips_compliance(self, loss_fn):
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y, None, None, None)
        assert "comp_loss" not in result
        assert "vf_loss" in result

    def test_disabled_components(self):
        loss_fn = PhysicsConstraintLoss(lambda_vf=0.0, lambda_comp=0.0)
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y, None, None, None)
        assert result["total"].item() == pytest.approx(0.0, abs=1e-6)

    def test_stress_component(self):
        loss_fn = PhysicsConstraintLoss(lambda_vf=1.0, lambda_stress=10.0)
        x = torch.rand(2, 1, 8, 8)
        y = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y)
        assert "stress_loss" in result

    def test_outputs_require_grad(self, loss_fn):
        x = torch.rand(2, 1, 8, 8, requires_grad=True)
        y = torch.rand(2, 1, 8, 8)
        bc = torch.rand(2, 1, 8, 8)
        lx = torch.rand(2, 1, 8, 8)
        ly = torch.rand(2, 1, 8, 8)
        result = loss_fn(x, y, bc, lx, ly)
        # "total" should be differentiable w.r.t. x
        result["total"].backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
