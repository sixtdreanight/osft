"""SVD Weight Decomposer for Orthogonal Subspace Fine-Tuning.

Decomposes pre-trained model weights via SVD into:
  - Principal subspace (Wr, frozen): captures dominant knowledge
  - Residual subspace (dW, learnable): fine-tuned for domain adaptation

Reference: main.py Step 1 (SVD decomposition and orthogonal subspace preparation)
"""

import torch
import torch.nn as nn
import pickle
import os
from typing import Dict, List, Tuple, Optional


class SVDWeightDecomposer:
    """SVD-based weight decomposition for parameter-efficient fine-tuning.

    W = U @ Σ @ V^T  →  W = W_r + ΔW
      where W_r = U_r @ Σ_r @ V_r^T  (principal, frozen)
            ΔW = U_nr @ Σ_nr @ V_nr^T  (residual, learnable)
    """

    def __init__(self, energy_threshold: float = 0.80):
        self.energy_threshold = energy_threshold
        self.results: Dict[str, dict] = {}

    @staticmethod
    def extract_weight(layer: nn.Module) -> Optional[torch.Tensor]:
        """Extract weight matrix and flatten to 2D."""
        if isinstance(layer, nn.Linear):
            return layer.weight.data.clone()
        elif isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            w = layer.weight.data.clone()
            return w.view(w.size(0), -1)
        return None

    @staticmethod
    def svd(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SVD: W = U @ Σ @ V^T."""
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U, torch.diag(S), Vh.T

    def energy_ratio(self, S: torch.Tensor) -> torch.Tensor:
        """Cumulative energy ratio from singular values."""
        total = (S ** 2).sum()
        if total < 1e-10:
            return torch.ones_like(S)
        return (S ** 2).cumsum(dim=0) / total

    def determine_rank(self, S: torch.Tensor) -> int:
        """Find minimum rank r such that retained energy >= threshold."""
        ratio = self.energy_ratio(S)
        r = torch.searchsorted(ratio, self.energy_threshold).item() + 1
        return min(r, len(S))

    def decompose(self, layer_name: str, layer: nn.Module) -> Optional[dict]:
        """Decompose a single layer via SVD."""
        W = self.extract_weight(layer)
        if W is None:
            return None

        U, Sigma, V = self.svd(W)
        sv = torch.diag(Sigma)
        r = self.determine_rank(sv)
        n = len(sv)

        Ur, Sr, Vr = U[:, :r], Sigma[:r, :r], V[:, :r]
        Unr, Snr, Vnr = U[:, r:], Sigma[r:, r:], V[:, r:]

        Wr = Ur @ Sr @ Vr.T
        dW = Unr @ Snr @ Vnr.T

        recon_err = torch.norm(W - Wr - dW, p="fro").item()

        result = {
            "layer_name": layer_name,
            "original_shape": W.shape,
            "layer_type": type(layer).__name__,
            "U": U,
            "Sigma": Sigma,
            "V": V,
            "singular_values": sv,
            "rank": r,
            "total_dim": n,
            "residual_dim": n - r,
            "energy_retained": self.energy_ratio(sv)[r - 1].item(),
            # Principal subspace (frozen)
            "Ur": Ur,
            "Sr": Sr,
            "Vr": Vr,
            "Wr": Wr,
            # Residual subspace (learnable)
            # Absorb Snr into Unr so OSFT layer reconstructs dW = Unr @ Vnr^T
            "Unr": Unr @ Snr,  # pre-multiplied with singular values
            "Snr": Snr,        # stored for reference
            "Vnr": Vnr,
            "dW": dW,
            "reconstruction_error": recon_err,
        }

        # Store layer config for Conv layers
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            config = {
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding,
                "has_bias": layer.bias is not None,
            }
            if isinstance(layer, nn.ConvTranspose2d):
                config["output_padding"] = getattr(layer, "output_padding", 0)
            result["conv_config"] = config
        elif isinstance(layer, nn.Linear):
            result["linear_config"] = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "has_bias": layer.bias is not None,
            }

        self.results[layer_name] = result
        return result

    @staticmethod
    def _matches_target(name: str, target: str) -> bool:
        """Check if target matches module name respecting dot boundaries.

        A target matches if either:
          - it equals a dot-separated segment of name (e.g. "deconv" in "d1.deconv")
          - it equals the first N dot-separated segments (e.g. "res_blocks.0" in "res_blocks.0.conv1")

        This avoids substring false matches like "res_blocks.0" matching "res_blocks.10".
        """
        segs = name.split(".")
        t_segs = target.split(".")
        # Match as prefix of segments: "res_blocks.0" matches "res_blocks.0.conv1"
        if len(t_segs) <= len(segs) and segs[:len(t_segs)] == t_segs:
            return True
        # Match as a single segment: "deconv" matches "d1.deconv"
        if len(t_segs) == 1 and target in segs:
            return True
        return False

    def decompose_model(
        self, model: nn.Module, target_layers: Optional[List[str]] = None, verbose: bool = True
    ):
        """Decompose all targeted layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                if target_layers is None or any(self._matches_target(name, t) for t in target_layers):
                    result = self.decompose(name, module)
                    if result and verbose:
                        pct = result["energy_retained"] * 100
                        print(
                            f"[SVD] {name}: rank={result['rank']}/{result['total_dim']} "
                            f"({pct:.1f}% energy), residual_dim={result['residual_dim']}"
                        )

    def save(self, path: str):
        """Save decomposition results to disk."""
        save_data = {
            "energy_threshold": self.energy_threshold,
            "results": {
                name: {
                    k: v for k, v in r.items()
                    if k not in ("U", "Sigma", "V")
                }
                for name, r in self.results.items()
            },
        }
        # Save tensors separately for the full SVD matrices
        for name, r in self.results.items():
            save_data["results"][name]["Ur"] = r["Ur"]
            save_data["results"][name]["Vr"] = r["Vr"]
            save_data["results"][name]["Unr"] = r["Unr"]
            save_data["results"][name]["Snr"] = r["Snr"]
            save_data["results"][name]["Vnr"] = r["Vnr"]
            save_data["results"][name]["Wr"] = r["Wr"]

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        """Load decomposition results from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.energy_threshold = data["energy_threshold"]
        self.results = data["results"]

    def summary(self) -> dict:
        """Return decomposition summary statistics."""
        total_orig = 0
        total_residual = 0
        for name, r in self.results.items():
            orig = r["original_shape"][0] * r["original_shape"][1]
            resid = r["residual_dim"]
            total_orig += orig
            total_residual += resid

        return {
            "n_layers": len(self.results),
            "total_params_orig": total_orig,
            "total_residual_dim": total_residual,
            "avg_energy_retained": sum(r["energy_retained"] for r in self.results.values()) / len(self.results),
            "avg_rank_ratio": sum(r["rank"] / r["total_dim"] for r in self.results.values()) / len(self.results),
        }
