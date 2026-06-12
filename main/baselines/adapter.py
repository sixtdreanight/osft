"""Adapter baseline for fine-tuning.

Inserts 1x1 convolutional adapter layers after each Conv/Deconv block.
Adapters are small bottleneck networks: Linear → GELU → Linear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ConvAdapter(nn.Module):
    """Adapter module inserted after a convolutional layer.

    Structure: Conv2d(1x1, in_ch → hidden_dim) → GELU → Conv2d(1x1, hidden_dim → in_ch).
    Includes residual connection to preserve original behavior initially.
    """

    def __init__(self, channels: int, hidden_dim: int = 32):
        super().__init__()
        self.down = nn.Conv2d(channels, hidden_dim, 1)
        self.up = nn.Conv2d(hidden_dim, channels, 1)
        # Initialize near-zero for identity-like behavior at start
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.down(residual))
        h = self.up(h)
        return x + h


class AdaptedConv2d(nn.Module):
    """Conv2d with post-convolution adapter."""

    def __init__(self, original: nn.Conv2d, hidden_dim: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(
            original.in_channels, original.out_channels,
            original.kernel_size, original.stride, original.padding,
            bias=original.bias is not None,
        )
        self.conv.weight.data = original.weight.data.clone()
        if original.bias is not None:
            self.conv.bias.data = original.bias.data.clone()
        # Freeze original conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.adapter = ConvAdapter(original.out_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.adapter(h, h)


class AdaptedConvTranspose2d(nn.Module):
    """ConvTranspose2d with post-convolution adapter."""

    def __init__(self, original: nn.ConvTranspose2d, hidden_dim: int = 32):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            original.in_channels, original.out_channels,
            original.kernel_size, original.stride, original.padding,
            output_padding=original.output_padding, bias=original.bias is not None,
        )
        self.deconv.weight.data = original.weight.data.clone()
        if original.bias is not None:
            self.deconv.bias.data = original.bias.data.clone()
        for p in self.deconv.parameters():
            p.requires_grad = False
        self.adapter = ConvAdapter(original.out_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.deconv(x)
        return self.adapter(h, h)


def _match_target(name: str, target: str) -> bool:
    """Match a target pattern to a module name respecting dot boundaries."""
    segs = name.split(".")
    t_segs = target.split(".")
    if len(t_segs) <= len(segs) and segs[:len(t_segs)] == t_segs:
        return True
    if len(t_segs) == 1 and target in segs:
        return True
    return False


def apply_adapter_to_generator(
    generator: nn.Module,
    hidden_dim: int = 32,
    target_layers: Optional[List[str]] = None,
) -> nn.Module:
    """Replace conv layers with adapter-equipped variants."""
    replacements = {}

    for name, module in generator.named_modules():
        if target_layers and not any(_match_target(name, t) for t in target_layers):
            continue
        if isinstance(module, nn.Conv2d):
            replacements[name] = AdaptedConv2d(module, hidden_dim)
        elif isinstance(module, nn.ConvTranspose2d):
            replacements[name] = AdaptedConvTranspose2d(module, hidden_dim)

    for full_name, new_module in replacements.items():
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = generator.get_submodule(parts[0])
            setattr(parent, parts[1], new_module)

    return generator


def count_adapter_params(model: nn.Module) -> dict:
    """Count adapter-specific trainable parameters."""
    adapter_params = 0
    total_trainable = 0
    total_all = sum(p.numel() for p in model.parameters())
    total_all += sum(b.numel() for b in model.buffers())

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if "adapter" in name:
                adapter_params += param.numel()

    return {
        "adapter_params": adapter_params,
        "total_trainable": total_trainable,
        "total_all": total_all,
        "trainable_pct": 100 * total_trainable / total_all,
    }
