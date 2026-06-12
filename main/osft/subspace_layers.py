"""Orthogonal Subspace Layers for parameter-efficient fine-tuning.

Replaces standard Conv2d/ConvTranspose2d/Linear layers with OSFT variants
that freeze the principal subspace (Wr) and only optimize the residual subspace (ΔW).

Key concepts:
  - W = Wr + ΔW, where ΔW = (Unr_orig + A@B) @ (Vnr_orig + C@D)^T
  - Orthogonality constraint: ||Ur^T @ Unr||_F^2 + ||Vr^T @ Vnr||_F^2
  - Singular value constraint: ||ΔW||_F^2 (prevent over-adaptation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class _OSFTBase(nn.Module):
    """Shared base for all orthogonal subspace fine-tuning layers.

    W = Wr (frozen, principal) + dW (learnable, residual).
    dW = (Unr_orig + A@B) @ (Vnr_orig + C@D)^T, where A,B,C,D are
    low-rank adapters (rank ≤ 8 by default).
    """

    def __init__(self, decomposition: dict, residual_rank: int = None):
        super().__init__()
        self.register_buffer("Wr", decomposition["Wr"])
        r = decomposition["rank"]
        n = decomposition["total_dim"]
        self.residual_rank = residual_rank or min(n - r, 8)
        m, k = decomposition["Unr"].shape  # m=rows, k=residual dimension

        self.A = nn.Parameter(torch.randn(m, self.residual_rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.residual_rank, k))
        self.C = nn.Parameter(torch.randn(decomposition["Vnr"].shape[0], self.residual_rank) * 0.01)
        self.D = nn.Parameter(torch.zeros(self.residual_rank, k))

        self.register_buffer("Ur", decomposition["Ur"])
        self.register_buffer("Vr", decomposition["Vr"])
        self.register_buffer("Unr_orig", decomposition["Unr"])
        self.register_buffer("Vnr_orig", decomposition["Vnr"])

    def _residual_weight(self) -> torch.Tensor:
        Unr = self.Unr_orig + self.A @ self.B
        Vnr = self.Vnr_orig + self.C @ self.D
        return Unr @ Vnr.T

    def _full_weight(self) -> torch.Tensor:
        return self.Wr + self._residual_weight()

    def orthogonality_loss(self) -> torch.Tensor:
        Unr = self.Unr_orig + self.A @ self.B
        Vnr = self.Vnr_orig + self.C @ self.D
        return (torch.norm(self.Ur.T @ Unr, p="fro") ** 2 +
                torch.norm(self.Vr.T @ Vnr, p="fro") ** 2)

    def singular_value_constraint(self) -> torch.Tensor:
        return torch.norm(self._residual_weight(), p="fro") ** 2


class OrthogonalSubspaceConv2d(_OSFTBase):
    """Conv2d with orthogonal subspace fine-tuning."""

    def __init__(self, decomposition: dict, original_layer: nn.Conv2d,
                 residual_rank: Optional[int] = None):
        super().__init__(decomposition, residual_rank)
        cfg = decomposition["conv_config"]
        self.stride = cfg["stride"]
        self.padding = cfg["padding"]
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.kernel_size = cfg["kernel_size"]
        self.weight_shape = original_layer.weight.shape
        self.bias = nn.Parameter(original_layer.bias.data.clone()) \
            if cfg["has_bias"] and original_layer.bias is not None \
            else nn.Parameter(torch.zeros(cfg["out_channels"]))
        self._weight_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._full_weight().view(self.weight_shape)
        if self.training and W.requires_grad:
            W.retain_grad()
            self._weight_ref = W
        return F.conv2d(x, W, self.bias, self.stride, self.padding)


class OrthogonalSubspaceConvTranspose2d(_OSFTBase):
    """ConvTranspose2d with orthogonal subspace fine-tuning."""

    def __init__(self, decomposition: dict, original_layer: nn.ConvTranspose2d,
                 residual_rank: Optional[int] = None):
        super().__init__(decomposition, residual_rank)
        cfg = decomposition["conv_config"]
        self.stride = cfg["stride"]
        self.padding = cfg["padding"]
        self.output_padding = cfg.get("output_padding", 1)
        self.in_channels = cfg["in_channels"]
        self.out_channels = cfg["out_channels"]
        self.kernel_size = cfg["kernel_size"]
        self.weight_shape = original_layer.weight.shape
        self.bias = nn.Parameter(original_layer.bias.data.clone()) \
            if cfg["has_bias"] and original_layer.bias is not None \
            else nn.Parameter(torch.zeros(cfg["out_channels"]))
        self._weight_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._full_weight().view(self.weight_shape)
        if self.training and W.requires_grad:
            W.retain_grad()
            self._weight_ref = W
        return F.conv_transpose2d(x, W, self.bias, self.stride, self.padding,
                                   output_padding=self.output_padding)


class OrthogonalSubspaceLinear(_OSFTBase):
    """Linear layer with orthogonal subspace fine-tuning."""

    def __init__(self, decomposition: dict, original_layer: nn.Linear,
                 residual_rank: Optional[int] = None):
        super().__init__(decomposition, residual_rank)
        lcfg = decomposition.get("linear_config", {})
        has_bias = lcfg.get("has_bias", False) and original_layer.bias is not None
        self.bias = nn.Parameter(original_layer.bias.data.clone()) \
            if has_bias else nn.Parameter(torch.zeros(decomposition["Unr"].shape[0]))
        self._weight_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._full_weight()
        if self.training and W.requires_grad:
            W.retain_grad()
            self._weight_ref = W
        return F.linear(x, W, self.bias)


def apply_osft_to_generator(
    generator: nn.Module,
    decomposition_results: Dict[str, dict],
) -> nn.Module:
    """Replace targeted layers in the generator with OSFT variants.

    Args:
        generator: Pre-trained TopologyGANGenerator instance.
        decomposition_results: Output from SVDWeightDecomposer.decompose_model().

    Returns:
        Generator with OSFT layers (in-place modification).
    """
    osft_layers = {}
    for name, module in generator.named_modules():
        if name in decomposition_results:
            decomp = decomposition_results[name]
            layer_type = decomp["layer_type"]
            if layer_type == "Conv2d":
                new_layer = OrthogonalSubspaceConv2d(decomp, module)
            elif layer_type == "ConvTranspose2d":
                new_layer = OrthogonalSubspaceConvTranspose2d(decomp, module)
            elif layer_type == "Linear":
                new_layer = OrthogonalSubspaceLinear(decomp, module)
            else:
                continue
            osft_layers[name] = new_layer

    # Replace layers in-place and move to generator's device
    device = next(generator.parameters()).device
    for full_name, new_layer in osft_layers.items():
        new_layer.to(device)
        parent_name, attr_name = _get_parent_attr(full_name)
        if parent_name == "":
            setattr(generator, attr_name, new_layer)
        else:
            parent = generator.get_submodule(parent_name)
            setattr(parent, attr_name, new_layer)

    # Freeze all non-OSFT parameters (BatchNorm, SE-ResNet convs, etc.)
    _OSFT_TYPES = (OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d,
                   OrthogonalSubspaceLinear)
    osft_module_names = {mod_name for mod_name, mod in generator.named_modules()
                         if isinstance(mod, _OSFT_TYPES)}
    for param_name, param in generator.named_parameters():
        parent = param_name.rsplit(".", 1)[0] if "." in param_name else ""
        if parent not in osft_module_names:
            param.requires_grad = False

    return generator


def _get_parent_attr(full_name: str) -> Tuple[str, str]:
    """Split 'encoder.0.conv' into ('encoder.0', 'conv')."""
    parts = full_name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", full_name
