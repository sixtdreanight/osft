"""Tests for SVD Weight Decomposer."""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main"))

from osft.decomposer import SVDWeightDecomposer


class TestSVDWeightDecomposer:

    @pytest.fixture
    def decomposer(self):
        return SVDWeightDecomposer(energy_threshold=0.80)

    @pytest.fixture
    def simple_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Linear(16 * 8 * 8, 10),
        )
        return model

    # ---- extract_weight ----

    def test_extract_conv2d_weight_shape(self, decomposer):
        layer = nn.Conv2d(3, 8, 3, padding=1)
        W = decomposer.extract_weight(layer)
        assert W.shape == (8, 3 * 3 * 3)  # [out_ch, in_ch * k * k]

    def test_extract_conv_transpose_weight_shape(self, decomposer):
        layer = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        W = decomposer.extract_weight(layer)
        assert W.shape == (16, 8 * 4 * 4)

    def test_extract_linear_weight_shape(self, decomposer):
        layer = nn.Linear(256, 10)
        W = decomposer.extract_weight(layer)
        assert W.shape == (10, 256)

    def test_extract_weight_returns_none_for_relu(self, decomposer):
        layer = nn.ReLU()
        assert decomposer.extract_weight(layer) is None

    # ---- SVD ----

    def test_svd_reconstruction(self, decomposer):
        W = torch.randn(20, 30)
        U, Sigma, V = decomposer.svd(W)
        W_recon = U @ Sigma @ V.T
        assert torch.allclose(W, W_recon, atol=1e-4)

    def test_svd_singular_values_nonnegative(self, decomposer):
        W = torch.randn(20, 30)
        _, Sigma, _ = decomposer.svd(W)
        sv = torch.diag(Sigma)
        assert (sv >= 0).all()

    # ---- energy_ratio ----

    def test_energy_ratio_monotonic(self, decomposer):
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
        ratio = decomposer.energy_ratio(S)
        assert ratio[0] < ratio[-1]
        assert torch.allclose(ratio[-1], torch.tensor(1.0), atol=1e-6)
        # Monotonic: each step >= previous
        diffs = torch.diff(ratio)
        assert (diffs > -1e-10).all()

    def test_energy_ratio_zero_total(self, decomposer):
        S = torch.zeros(10)
        ratio = decomposer.energy_ratio(S)
        assert torch.allclose(ratio, torch.ones_like(S))

    # ---- determine_rank ----

    def test_determine_rank_hits_threshold(self, decomposer):
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
        r = decomposer.determine_rank(S)
        # energy retained should be >= 0.80
        ratio = decomposer.energy_ratio(S)
        assert ratio[r - 1] >= decomposer.energy_threshold - 1e-9

    def test_determine_rank_returns_full_for_high_threshold(self):
        decomposer = SVDWeightDecomposer(energy_threshold=0.999)
        S = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
        r = decomposer.determine_rank(S)
        assert r == len(S)

    def test_determine_rank_at_least_one(self):
        decomposer = SVDWeightDecomposer(energy_threshold=0.01)
        S = torch.tensor([100.0, 0.1, 0.05])
        r = decomposer.determine_rank(S)
        assert r >= 1

    # ---- decompose ----

    def test_decompose_conv2d_layer(self, decomposer):
        layer = nn.Conv2d(3, 8, 3, padding=1)
        result = decomposer.decompose("layers.0", layer)
        assert result is not None
        assert result["layer_name"] == "layers.0"
        assert result["layer_type"] == "Conv2d"
        assert "Ur" in result and "Vr" in result
        assert "Wr" in result and "dW" in result
        assert result["rank"] >= 1
        assert result["rank"] <= result["total_dim"]
        assert result["residual_dim"] == result["total_dim"] - result["rank"]
        assert result["energy_retained"] >= decomposer.energy_threshold - 1e-9
        # Reconstruction should be near-perfect
        assert result["reconstruction_error"] < 1e-4
        assert "conv_config" in result

    def test_decompose_linear_layer(self, decomposer):
        layer = nn.Linear(128, 10)
        result = decomposer.decompose("fc", layer)
        assert result is not None
        assert result["layer_type"] == "Linear"
        assert "linear_config" in result

    def test_decompose_skip_relu(self, decomposer):
        layer = nn.ReLU()
        assert decomposer.decompose("act", layer) is None

    # ---- decompose_model ----

    def test_decompose_model_finds_all_conv_linear(self, decomposer, simple_model):
        decomposer.decompose_model(simple_model, verbose=True)
        assert len(decomposer.results) == 3  # 2 Conv + 1 Linear

    def test_decompose_model_with_target_layers(self, decomposer, simple_model):
        decomposer.decompose_model(simple_model, target_layers=["0"], verbose=False)
        assert len(decomposer.results) == 1

    def test_decompose_model_with_target_list(self, decomposer, simple_model):
        decomposer.decompose_model(simple_model, target_layers=["0", "2"], verbose=False)
        assert len(decomposer.results) == 2

    # ---- _matches_target ----

    def test_matches_exact_segment(self):
        assert SVDWeightDecomposer._matches_target("d1.deconv", "deconv") is True
        assert SVDWeightDecomposer._matches_target("encoder.0.conv", "conv") is True
        assert SVDWeightDecomposer._matches_target("encoder.1.conv", "conv") is True

    def test_matches_prefix_segments(self):
        assert SVDWeightDecomposer._matches_target("res_blocks.0.conv1", "res_blocks.0") is True

    def test_no_match_wrong_segment(self):
        assert SVDWeightDecomposer._matches_target("res_blocks.10.conv1", "res_blocks.1") is False
        assert SVDWeightDecomposer._matches_target("encoder.0", "decoder") is False

    # ---- summary ----

    def test_summary_statistics(self, decomposer, simple_model):
        decomposer.decompose_model(simple_model, verbose=False)
        s = decomposer.summary()
        assert s["n_layers"] == 3
        assert s["total_params_orig"] > 0
        assert s["avg_energy_retained"] >= decomposer.energy_threshold - 1e-9
        assert 0 < s["avg_rank_ratio"] <= 1.0

    # ---- save / load ----

    def test_save_load_roundtrip(self, decomposer, simple_model, tmp_path):
        decomposer.decompose_model(simple_model, verbose=False)
        path = str(tmp_path / "decomp.pkl")
        decomposer.save(path)

        loaded = SVDWeightDecomposer()
        loaded.load(path)
        assert loaded.energy_threshold == decomposer.energy_threshold
        assert len(loaded.results) == len(decomposer.results)
        for name in decomposer.results:
            assert name in loaded.results
            assert torch.allclose(loaded.results[name]["Wr"], decomposer.results[name]["Wr"])

    # ---- edge cases ----

    def test_deterministic_decomposition(self, decomposer):
        torch.manual_seed(42)
        layer = nn.Conv2d(3, 8, 3)
        r1 = decomposer.decompose("layer", layer)
        layer2 = nn.Conv2d(3, 8, 3)
        layer2.weight.data = layer.weight.data.clone()
        r2 = decomposer.decompose("layer", layer2)
        assert r1["rank"] == r2["rank"]

    def test_tiny_weight_matrix(self, decomposer):
        layer = nn.Conv2d(1, 2, 1, stride=1, padding=0)
        result = decomposer.decompose("tiny", layer)
        assert result is not None
        assert result["total_dim"] >= 1

    def test_energy_threshold_zero(self):
        decomposer = SVDWeightDecomposer(energy_threshold=0.01)
        layer = nn.Conv2d(4, 8, 3)
        result = decomposer.decompose("layer", layer)
        # With very low threshold, should keep minimal rank
        assert result["rank"] >= 1

    def test_energy_threshold_near_one(self):
        decomposer = SVDWeightDecomposer(energy_threshold=0.999)
        layer = nn.Conv2d(4, 8, 3)
        result = decomposer.decompose("layer", layer)
        # Should keep nearly all singular values
        assert result["rank"] >= result["total_dim"] - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
