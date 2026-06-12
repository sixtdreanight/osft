"""LoRA (Low-Rank Adaptation) baseline for Conv2d/ConvTranspose2d layers.

Standard LoRA adapters for convolutional GAN fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LoRAConv2d(nn.Module):
    """LoRA-adapted Conv2d: W' = W_frozen + B @ A (low-rank update)."""

    def __init__(
        self,
        original: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.stride = original.stride
        self.padding = original.padding

        # Frozen original weights
        self.register_buffer("weight", original.weight.data.clone())
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.clone())
        else:
            self.register_buffer("bias", torch.zeros(original.out_channels))

        out_ch, in_ch, kh, kw = original.weight.shape
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kh = kh
        self.kw = kw

        # LoRA: W_update = B @ A, where A is (rank, in_ch*kh*kw), B is (out_ch, rank)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_ch * kh * kw) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_update = (self.lora_B @ self.lora_A).view(self.out_ch, self.in_ch, self.kh, self.kw)
        W = self.weight + self.scale * W_update
        return F.conv2d(x, W, self.bias, self.stride, self.padding)


class LoRAConvTranspose2d(nn.Module):
    """LoRA-adapted ConvTranspose2d."""

    def __init__(
        self,
        original: nn.ConvTranspose2d,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.stride = original.stride
        self.padding = original.padding
        self.output_padding = original.output_padding

        self.register_buffer("weight", original.weight.data.clone())
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.clone())
        else:
            self.register_buffer("bias", torch.zeros(original.out_channels))

        in_ch, out_ch, kh, kw = original.weight.shape
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.kh = kh
        self.kw = kw

        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_ch * kh * kw) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_update = (self.lora_B @ self.lora_A).view(self.in_ch, self.out_ch, self.kh, self.kw)
        W = self.weight + self.scale * W_update
        return F.conv_transpose2d(x, W, self.bias, self.stride, self.padding,
                                   output_padding=self.output_padding)


def _match_target(name: str, target: str) -> bool:
    """Match a target pattern to a module name respecting dot boundaries."""
    segs = name.split(".")
    t_segs = target.split(".")
    if len(t_segs) <= len(segs) and segs[:len(t_segs)] == t_segs:
        return True
    if len(t_segs) == 1 and target in segs:
        return True
    return False


def apply_lora_to_generator(
    generator: nn.Module,
    rank: int = 4,
    target_layers: Optional[List[str]] = None,
) -> nn.Module:
    """Replace Conv2d/ConvTranspose2d layers in generator with LoRA variants."""
    replacements = {}

    for name, module in generator.named_modules():
        if target_layers and not any(_match_target(name, t) for t in target_layers):
            continue
        if isinstance(module, nn.Conv2d):
            replacements[name] = LoRAConv2d(module, rank=rank)
        elif isinstance(module, nn.ConvTranspose2d):
            replacements[name] = LoRAConvTranspose2d(module, rank=rank)

    for full_name, new_module in replacements.items():
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = generator.get_submodule(parts[0])
            setattr(parent, parts[1], new_module)

    return generator


def count_lora_params(model: nn.Module) -> dict:
    """Count LoRA-specific trainable parameters."""
    lora_params = 0
    total_trainable = 0
    total_all = sum(p.numel() for p in model.parameters())
    total_all += sum(b.numel() for b in model.buffers())

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if "lora" in name:
                lora_params += param.numel()

    return {
        "lora_params": lora_params,
        "total_trainable": total_trainable,
        "total_all": total_all,
        "trainable_pct": 100 * total_trainable / total_all,
    }
