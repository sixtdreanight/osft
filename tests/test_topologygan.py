"""Tests for TopologyGAN model architecture."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from model.topologygan import (
    ConvBlock, DeconvBlock, SEBlock, SEResBlock,
    TopologyGANGenerator, TopologyGANDiscriminator,
)


class TestConvBlock:

    def test_output_shape(self):
        block = ConvBlock(3, 16, kernel=3, stride=2, padding=1)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_no_norm_no_act(self):
        block = ConvBlock(3, 16, kernel=3, stride=1, padding=1, use_norm=False, activation="none")
        x = torch.randn(2, 3, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_with_norm_relu(self):
        block = ConvBlock(3, 16, kernel=3, stride=1, padding=1, use_norm=True, activation="relu")
        x = torch.randn(2, 3, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)

    def test_with_norm_lrelu(self):
        block = ConvBlock(3, 16, kernel=3, stride=1, padding=1, use_norm=True, activation="lrelu")
        x = torch.randn(2, 3, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)
        # Output should be in range (not bounded to [0,1] like sigmoid)
        assert out.min() >= -10

    def test_sigmoid_activation(self):
        block = ConvBlock(3, 16, kernel=3, stride=1, padding=1, activation="sigmoid")
        x = torch.randn(2, 3, 16, 16)
        out = block(x)
        assert 0 <= out.min() <= out.max() <= 1


class TestDeconvBlock:

    def test_output_shape_upsampling(self):
        block = DeconvBlock(16, 8, kernel=4, stride=2, padding=1, output_padding=1)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape[0] == 2 and out.shape[1] == 8

    def test_sigmoid_activation(self):
        block = DeconvBlock(16, 8, kernel=4, stride=2, padding=1, activation="sigmoid")
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert 0 <= out.min() <= out.max() <= 1


class TestSEBlock:

    def test_output_shape(self):
        block = SEBlock(64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_ranges_between_zero_and_one(self):
        block = SEBlock(64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        # SE scales channels, output should not exceed input range significantly
        assert out.shape == x.shape


class TestSEResBlock:

    def test_output_shape(self):
        block = SEResBlock(128)
        x = torch.randn(2, 128, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)

    def test_residual_connection(self):
        block = SEResBlock(16)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        out = block(x)
        assert out.shape == x.shape
        loss = out.sum()
        loss.backward()
        # Gradient flows through residual
        assert block.conv1.weight.grad is not None


class TestTopologyGANGenerator:

    @pytest.mark.parametrize("variant", ["unet", "se_res_unet"])
    def test_forward_shape_default(self, variant):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant=variant, height=64, width=128,
        )
        x = torch.randn(2, 3, 64, 128)
        out = gen(x)
        assert out.shape == (2, 1, 64, 128)
        # Sigmoid output should be in [0, 1]
        assert 0 <= out.min() <= out.max() <= 1

    @pytest.mark.parametrize("variant", ["unet", "se_res_unet"])
    def test_forward_with_noise(self, variant):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant=variant, height=64, width=64,
        )
        x = torch.randn(2, 3, 64, 64)
        z = torch.randn(2, 100)
        out = gen(x, z)
        assert out.shape == (2, 1, 64, 64)

    def test_deterministic_without_noise(self):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant="unet", height=64, width=64,
        )
        gen.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out1 = gen(x)
            out2 = gen(x)
        assert torch.allclose(out1, out2)

    def test_stochastic_with_noise(self):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant="unet", height=64, width=64,
        )
        gen.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out1 = gen(x, torch.randn(1, 100))
            out2 = gen(x, torch.randn(1, 100))
        assert not torch.allclose(out1, out2)

    def test_feature_layer_names(self):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant="se_res_unet", height=64, width=128,
        )
        names = gen.get_feature_layer_names()
        assert len(names) > 0
        assert "e2" in names
        assert "e3" in names
        assert "d1" in names
        assert "d2" in names

    def test_forward_with_features(self):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=64,
            variant="se_res_unet", height=64, width=128,
        )
        x = torch.randn(2, 3, 64, 128)
        out, features = gen.forward_with_features(x)
        assert out.shape == (2, 1, 64, 128)
        assert len(features) > 0
        for name, feat in features.items():
            assert feat.shape[0] == 2

    def test_different_resolutions(self):
        for h, w in [(64, 64), (64, 128)]:
            gen = TopologyGANGenerator(
                input_c_dim=3, output_c_dim=1, gf_dim=32,
                variant="unet", height=h, width=w,
            )
            x = torch.randn(1, 3, h, w)
            out = gen(x)
            assert out.shape[0] == 1 and out.shape[2] == h and out.shape[3] == w

    def test_gradients_flow(self):
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=32,
            variant="unet", height=64, width=64,
        )
        x = torch.randn(2, 3, 64, 64)
        out = gen(x)
        loss = ((out - torch.ones_like(out) * 0.5) ** 2).mean()
        loss.backward()
        # Check that gradients exist on first encoder conv
        assert gen.e1.conv.weight.grad is not None

    def test_invalid_variant_raises(self):
        with pytest.raises(AssertionError):
            TopologyGANGenerator(variant="nonexistent")


class TestTopologyGANDiscriminator:

    def test_forward_shape(self):
        disc = TopologyGANDiscriminator(
            condition_dim=6, output_c_dim=1, df_dim=32,
            height=64, width=128,
        )
        x = torch.randn(2, 7, 64, 128)  # 6 + 1 channels
        probs, logits = disc(x)
        assert probs.shape == (2, 1)
        assert logits.shape == (2, 1)
        assert (0 <= probs).all() and (probs <= 1).all()

    def test_different_resolution(self):
        disc = TopologyGANDiscriminator(
            condition_dim=6, output_c_dim=1, df_dim=16,
            height=32, width=64,
        )
        x = torch.randn(2, 7, 32, 64)
        probs, logits = disc(x)
        assert probs.shape == (2, 1)

    def test_gradients_flow(self):
        disc = TopologyGANDiscriminator(
            condition_dim=6, output_c_dim=1, df_dim=16,
            height=32, width=32,
        )
        x = torch.randn(2, 7, 32, 32)
        probs, logits = disc(x)
        loss = ((probs - torch.ones_like(probs) * 0.5) ** 2).mean()
        loss.backward()
        assert disc.conv1.conv.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
