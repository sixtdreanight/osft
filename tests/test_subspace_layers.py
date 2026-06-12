"""Tests for Orthogonal Subspace Layers."""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from osft.decomposer import SVDWeightDecomposer
from osft.subspace_layers import (
    OrthogonalSubspaceConv2d,
    OrthogonalSubspaceConvTranspose2d,
    OrthogonalSubspaceLinear,
    apply_osft_to_generator,
    _get_parent_attr,
)


def make_decomp(decomposer, layer, name="test_layer"):
    """Helper: decompose a layer and return the result dict."""
    return decomposer.decompose(name, layer)


class TestOrthogonalSubspaceConv2d:

    @pytest.fixture
    def decomposer(self):
        return SVDWeightDecomposer(energy_threshold=0.80)

    @pytest.fixture
    def orig_conv(self):
        torch.manual_seed(42)
        return nn.Conv2d(4, 8, 3, stride=1, padding=1)

    @pytest.fixture
    def osft_conv(self, decomposer, orig_conv):
        decomp = make_decomp(decomposer, orig_conv, "conv")
        return OrthogonalSubspaceConv2d(decomp, orig_conv)

    def test_forward_shape(self, osft_conv):
        x = torch.randn(2, 4, 16, 16)
        out = osft_conv(x)
        assert out.shape == (2, 8, 16, 16)

    def test_forward_close_to_original(self, osft_conv, orig_conv):
        """At init, OSFT preserves full original weight: W_init = Wr + dW = W.
        A@B=0 and C@D=0 at init, so dW_init = Unr_orig @ Vnr_orig^T = dW_orig."""
        x = torch.randn(1, 4, 16, 16)
        with torch.no_grad():
            out_orig = orig_conv(x)
            out_osft = osft_conv(x)
        assert torch.allclose(out_osft, out_orig, atol=1e-5)

    def test_wr_frozen(self, osft_conv):
        Wr = osft_conv.Wr.clone()
        # Forward + backward should not change Wr (it's a buffer)
        x = torch.randn(1, 4, 16, 16)
        out = osft_conv(x)
        loss = out.sum()
        loss.backward()
        assert torch.equal(osft_conv.Wr, Wr)

    def test_ab_learnable_gradients(self, osft_conv):
        x = torch.randn(1, 4, 16, 16, requires_grad=False)
        osft_conv.train()
        out = osft_conv(x)
        loss = out.sum()
        loss.backward()
        assert osft_conv.A.grad is not None
        assert osft_conv.B.grad is not None
        assert osft_conv.C.grad is not None
        assert osft_conv.D.grad is not None

    def test_weights_update_after_gradient_step(self, osft_conv):
        opt = torch.optim.SGD(osft_conv.parameters(), lr=0.1)
        x = torch.randn(1, 4, 16, 16)
        W_before = osft_conv._full_weight().clone()
        out = osft_conv(x)
        loss = ((out - torch.ones_like(out)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        W_after = osft_conv._full_weight()
        assert not torch.allclose(W_before, W_after)

    def test_orthogonality_loss(self, osft_conv):
        loss = osft_conv.orthogonality_loss()
        assert loss.item() >= 0
        assert loss.item() < 1000

    def test_singular_value_constraint(self, osft_conv):
        loss = osft_conv.singular_value_constraint()
        assert loss.item() >= 0

    def test_residual_weight_shape(self, osft_conv):
        dW = osft_conv._residual_weight()
        assert dW.ndim == 2

    def test_full_weight_equals_wr_plus_residual(self, osft_conv):
        W_full = osft_conv._full_weight()
        Wr = osft_conv.Wr
        dW = osft_conv._residual_weight()
        assert torch.allclose(W_full, Wr + dW, atol=1e-6)

    def test_weight_ref_none_in_eval_mode(self, osft_conv):
        """_weight_ref should be None after eval-mode forward (not set)."""
        osft_conv.eval()
        x = torch.randn(1, 4, 16, 16)
        out = osft_conv(x)
        loss = out.sum()
        loss.backward()
        assert osft_conv._weight_ref is None


class TestOrthogonalSubspaceConvTranspose2d:

    @pytest.fixture
    def decomposer(self):
        return SVDWeightDecomposer(energy_threshold=0.80)

    @pytest.fixture
    def orig_deconv(self):
        torch.manual_seed(42)
        return nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, output_padding=1)

    @pytest.fixture
    def osft_deconv(self, decomposer, orig_deconv):
        decomp = make_decomp(decomposer, orig_deconv, "deconv")
        return OrthogonalSubspaceConvTranspose2d(decomp, orig_deconv)

    def test_forward_shape(self, osft_deconv):
        x = torch.randn(2, 8, 4, 4)
        out = osft_deconv(x)
        # ConvTranspose2d with k=4, s=2, p=1, op=1: H_out = (4-1)*2 - 2 + 4 + 1 = 9
        assert out.shape[0] == 2 and out.shape[1] == 4

    def test_bias_shape(self, osft_deconv):
        # Bias should match out_channels (4), not in_channels (8)
        assert osft_deconv.bias.shape[0] == 4

    def test_forward_close_to_original(self, osft_deconv, orig_deconv):
        """At init, OSFT preserves full original weight."""
        x = torch.randn(1, 8, 4, 4)
        with torch.no_grad():
            out_orig = orig_deconv(x)
            out_osft = osft_deconv(x)
        assert torch.allclose(out_osft, out_orig, atol=1e-5)

    def test_ab_gradients_flow(self, osft_deconv):
        x = torch.randn(1, 8, 4, 4)
        osft_deconv.train()
        out = osft_deconv(x)
        loss = out.sum()
        loss.backward()
        assert osft_deconv.A.grad is not None
        assert osft_deconv.C.grad is not None


class TestOrthogonalSubspaceLinear:

    @pytest.fixture
    def decomposer(self):
        return SVDWeightDecomposer(energy_threshold=0.80)

    @pytest.fixture
    def orig_linear(self):
        torch.manual_seed(42)
        return nn.Linear(128, 10)

    @pytest.fixture
    def osft_linear(self, decomposer, orig_linear):
        decomp = make_decomp(decomposer, orig_linear, "fc")
        return OrthogonalSubspaceLinear(decomp, orig_linear)

    def test_forward_shape(self, osft_linear):
        x = torch.randn(2, 128)
        out = osft_linear(x)
        assert out.shape == (2, 10)

    def test_forward_close_to_original(self, osft_linear, orig_linear):
        """At init, OSFT preserves full original weight."""
        x = torch.randn(1, 128)
        with torch.no_grad():
            out_orig = orig_linear(x)
            out_osft = osft_linear(x)
        assert torch.allclose(out_osft, out_orig, atol=1e-5)

    def test_bias_shape(self, osft_linear):
        assert osft_linear.bias.shape[0] == 10


class TestApplyOSFTToGenerator:

    @pytest.fixture
    def toy_generator(self):
        class ToyGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Conv2d(3, 8, 3, padding=1)
                self.dec = nn.ConvTranspose2d(8, 1, 3, padding=1)
                self.fc = nn.Linear(64, 32)
                self.act = nn.ReLU()

            def forward(self, x):
                h = self.enc(x)
                h = self.act(h)
                h = self.dec(h)
                h = self.fc(h.view(h.size(0), -1))
                return h

        return ToyGen()

    @pytest.fixture
    def decomp_results(self, toy_generator):
        decomposer = SVDWeightDecomposer(energy_threshold=0.80)
        decomposer.decompose_model(toy_generator, verbose=False)
        return decomposer.results

    def test_replaces_conv_and_linear(self, toy_generator, decomp_results):
        apply_osft_to_generator(toy_generator, decomp_results)
        assert isinstance(toy_generator.enc, OrthogonalSubspaceConv2d), \
            f"Got {type(toy_generator.enc)}"
        assert isinstance(toy_generator.dec, OrthogonalSubspaceConvTranspose2d), \
            f"Got {type(toy_generator.dec)}"
        assert isinstance(toy_generator.fc, OrthogonalSubspaceLinear), \
            f"Got {type(toy_generator.fc)}"

    def test_relu_unchanged(self, toy_generator, decomp_results):
        apply_osft_to_generator(toy_generator, decomp_results)
        assert isinstance(toy_generator.act, nn.ReLU)

    def test_non_osft_params_frozen(self, toy_generator, decomp_results):
        apply_osft_to_generator(toy_generator, decomp_results)
        assert toy_generator.enc.A.requires_grad
        assert not toy_generator.enc.Wr.requires_grad  # buffer

    def test_forward_still_works(self, toy_generator, decomp_results):
        apply_osft_to_generator(toy_generator, decomp_results)
        x = torch.randn(2, 3, 8, 8)
        out = toy_generator(x)
        assert out.shape[1] == 32  # fc output dim

    def test_trainable_params_count_reduced(self, toy_generator, decomp_results):
        total_before = sum(p.numel() for p in toy_generator.parameters())
        apply_osft_to_generator(toy_generator, decomp_results)
        trainable_after = sum(p.numel() for p in toy_generator.parameters() if p.requires_grad)
        assert trainable_after < total_before


class TestGetParentAttr:

    def test_simple(self):
        assert _get_parent_attr("encoder.0.conv") == ("encoder.0", "conv")

    def test_two_levels(self):
        assert _get_parent_attr("res_blocks.0.conv1") == ("res_blocks.0", "conv1")

    def test_no_parent(self):
        assert _get_parent_attr("root") == ("", "root")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
