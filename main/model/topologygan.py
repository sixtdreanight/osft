"""TopologyGAN PyTorch Implementation

Exact reimplementation matching the original TensorFlow version from:
"TopologyGAN: Topology Optimization Using Generative Adversarial Networks"
(2020_TopologyGAN-master/code/model.py)

Supports two generator variants:
  - model_gan_unet: 5-layer U-Net with skip connections
  - model_gan_se_res_unet: 3-layer encoder + 32 SE-ResNet blocks + 4-layer decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class ConvBlock(nn.Module):
    """Conv2d + optional BatchNorm + Activation, matching TF conv2d behavior."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 5,
        stride: int = 2,
        padding: int = 2,
        use_norm: bool = False,
        activation: str = "none",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_norm)
        self.norm = nn.BatchNorm2d(out_ch) if use_norm else nn.Identity()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DeconvBlock(nn.Module):
    """ConvTranspose2d + optional BatchNorm + Activation, matching TF deconv2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 5,
        stride: int = 2,
        padding: int = 2,
        output_padding: int = 1,
        use_norm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel, stride, padding, output_padding, bias=not use_norm
        )
        self.norm = nn.BatchNorm2d(out_ch) if use_norm else nn.Identity()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.deconv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block matching the TF SE-Block in ops.py."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    """SE-ResNet block matching 'residual_block' in ops.py.

    Structure: Conv(k5,s1) + BN + LReLU + Conv(k5,s1) + BN + SEBlock + residual
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 5, 1, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return out + residual


class TopologyGANGenerator(nn.Module):
    """TopologyGAN Generator.

    Two architecture variants:
    - 'unet': 5-layer encoder-decoder U-Net with skip connections
    - 'se_res_unet': 3-layer encoder + 32 SE-ResNet bottleneck + 4-layer decoder

    Input: input conditions [B, input_c_dim, H, W]
           (VF + VM_stress + strain_energy)
    Output: topology density field [B, 1, H, W]
    """

    VARIANTS = ("unet", "se_res_unet")

    def __init__(
        self,
        input_c_dim: int = 3,
        output_c_dim: int = 1,
        gf_dim: int = 128,
        variant: str = "se_res_unet",
        height: int = 64,
        width: int = 128,
        nz: int = 100,
    ):
        super().__init__()
        assert variant in self.VARIANTS
        self.variant = variant
        self.height = height
        self.width = width
        self.nz = nz
        g = gf_dim

        # Noise projection (for stochastic generation / Jacobian analysis)
        # Dimension depends on variant's bottleneck spatial size
        if variant == "unet":
            noise_dim = g * 16 * (height // 32) * (width // 32)
        else:
            noise_dim = g * 4 * (height // 8) * (width // 8)
        self.fc_noise = nn.Linear(nz, noise_dim)

        if variant == "unet":
            self._build_unet(input_c_dim, output_c_dim, g)
        else:
            self._build_se_res_unet(input_c_dim, output_c_dim, g)

    def _build_unet(self, in_c: int, out_c: int, g: int):
        # Encoder: 5 levels (H->H/2->H/4->H/8->H/16->H/32)
        self.e1 = ConvBlock(in_c, g * 1, 5, 2, 2)                      # H/2 x W/2
        self.e2 = ConvBlock(g * 1, g * 2, 5, 2, 2, True, "lrelu")     # H/4 x W/4
        self.e3 = ConvBlock(g * 2, g * 4, 5, 2, 2, True, "lrelu")     # H/8 x W/8
        self.e4 = ConvBlock(g * 4, g * 8, 5, 2, 2, True, "lrelu")     # H/16 x W/16
        self.e5 = ConvBlock(g * 8, g * 16, 5, 2, 2, True, "lrelu")    # H/32 x W/32

        # Decoder: 6 levels with skip connections
        self.d1 = DeconvBlock(g * 16, g * 16, 5, 1, 2, 0, True, "relu")   # same size (H/32)
        self.d2 = DeconvBlock(g * 16 * 2, g * 8, 5, 2, 2, 1, True, "relu")  # H/16
        self.d3 = DeconvBlock(g * 8 * 2, g * 4, 5, 2, 2, 1, True, "relu")   # H/8
        self.d4 = DeconvBlock(g * 4 * 2, g * 2, 5, 2, 2, 1, True, "relu")   # H/4
        self.d5 = DeconvBlock(g * 2 * 2, g * 1, 5, 2, 2, 1, True, "relu")   # H/2
        self.d6 = DeconvBlock(g * 1 * 2, out_c, 5, 2, 2, 1, False, "sigmoid")  # H

    def _build_se_res_unet(self, in_c: int, out_c: int, g: int):
        # Encoder: 3 levels
        self.e1 = ConvBlock(in_c, g * 1, 5, 2, 2)                          # H/2 x W/2
        self.e2 = ConvBlock(g * 1, g * 2, 5, 2, 2, True, "lrelu")         # H/4 x W/4
        self.e3 = ConvBlock(g * 2, g * 4, 5, 2, 2, True, "lrelu")         # H/8 x W/8

        # Bottleneck: 32 SE-ResNet blocks
        self.res_blocks = nn.ModuleList([SEResBlock(g * 4) for _ in range(32)])

        # Decoder: 4 levels
        self.d1 = DeconvBlock(g * 4, g * 4, 5, 1, 2, 0, True, "relu")       # same (H/8)
        self.d2 = DeconvBlock(g * 4 * 2, g * 2, 5, 2, 2, 1, True, "relu")   # H/4
        self.d3 = DeconvBlock(g * 2 * 2, g * 1, 5, 2, 2, 1, True, "relu")   # H/2
        self.d4 = DeconvBlock(g * 1 * 2, out_c, 5, 2, 2, 1, False, "sigmoid")  # H

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input conditions [B, input_c_dim, H, W]
            z: Optional noise [B, nz]. If None, uses zero noise (deterministic mode).
               Required for diversity generation and Jacobian analysis.

        Returns:
            Topology density field [B, 1, H, W]
        """
        if self.variant == "unet":
            return self._forward_unet(x, z)
        return self._forward_se_res_unet(x, z)

    def _forward_unet(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # Noise injection at bottleneck
        if z is not None:
            B, C, H, W = e5.shape
            z_feat = self.fc_noise(z).view(B, C, H, W)
            e5 = e5 + z_feat

        d1 = self.d1(F.relu(e5))
        d1 = torch.cat([d1, e5], dim=1)
        d2 = torch.cat([self.d2(F.relu(d1)), e4], dim=1)
        d3 = torch.cat([self.d3(F.relu(d2)), e3], dim=1)
        d4 = torch.cat([self.d4(F.relu(d3)), e2], dim=1)
        d5 = torch.cat([self.d5(F.relu(d4)), e1], dim=1)
        return self.d6(F.relu(d5))

    def _forward_se_res_unet(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        # Noise injection at bottleneck (before SE-ResNet blocks)
        if z is not None:
            B, C, H, W = e3.shape
            z_feat = self.fc_noise(z).view(B, C, H, W)
            e3 = e3 + z_feat

        h = e3
        for res_block in self.res_blocks:
            h = res_block(h)

        d1 = self.d1(F.relu(h))
        d1 = torch.cat([d1, e3], dim=1)
        d2 = torch.cat([self.d2(F.relu(d1)), e2], dim=1)
        d3 = torch.cat([self.d3(F.relu(d2)), e1], dim=1)
        return self.d4(F.relu(d3))

    def get_feature_layer_names(self) -> List[str]:
        """Return layer names for CKA and feature extraction analysis.

        Covers encoder outputs, bottleneck, and decoder outputs
        at each resolution level.
        """
        if self.variant == "unet":
            return ["e2", "e3", "e4", "e5", "d2", "d3", "d4", "d5"]
        else:
            names = ["e2", "e3"]
            # Add first few and last few SE-ResNet blocks as representatives
            for i in [0, 15, 31]:
                names.append(f"res_blocks.{i}")
            names.extend(["d1", "d2", "d3"])
            return names

    def forward_with_features(
        self, x: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning both output and intermediate features.

        Used for CKA analysis and layer-wise knowledge localization.

        Returns:
            (density_field, {layer_name: feature_tensor})
        """
        features = {}

        if self.variant == "unet":
            e1 = self.e1(x)
            e2 = self.e2(e1); features["e2"] = e2
            e3 = self.e3(e2); features["e3"] = e3
            e4 = self.e4(e3); features["e4"] = e4
            e5 = self.e5(e4); features["e5"] = e5
            if z is not None:
                B, C, H, W = e5.shape
                e5 = e5 + self.fc_noise(z).view(B, C, H, W)
            d1 = self.d1(F.relu(e5))
            d1 = torch.cat([d1, e5], dim=1)
            d2 = torch.cat([self.d2(F.relu(d1)), e4], dim=1); features["d2"] = d2
            d3 = torch.cat([self.d3(F.relu(d2)), e3], dim=1); features["d3"] = d3
            d4 = torch.cat([self.d4(F.relu(d3)), e2], dim=1); features["d4"] = d4
            d5 = torch.cat([self.d5(F.relu(d4)), e1], dim=1); features["d5"] = d5
            out = self.d6(F.relu(d5))
        else:
            e1 = self.e1(x)
            e2 = self.e2(e1); features["e2"] = e2
            e3 = self.e3(e2); features["e3"] = e3
            if z is not None:
                B, C, H, W = e3.shape
                e3 = e3 + self.fc_noise(z).view(B, C, H, W)
            h = e3
            for idx, res_block in enumerate(self.res_blocks):
                h = res_block(h)
                if idx in [0, 15, 31]:
                    features[f"res_blocks.{idx}"] = h
            d1 = self.d1(F.relu(h))
            d1 = torch.cat([d1, e3], dim=1); features["d1"] = d1
            d2 = torch.cat([self.d2(F.relu(d1)), e2], dim=1); features["d2"] = d2
            d3 = torch.cat([self.d3(F.relu(d2)), e1], dim=1); features["d3"] = d3
            out = self.d4(F.relu(d3))

        return out, features


class TopologyGANDiscriminator(nn.Module):
    """TopologyGAN PatchGAN-style Discriminator.

    Input: [real_A + real_B] or [real_A + fake_B] concatenated along channel dim,
           shape [B, condition_dim + output_c_dim, H, W]
    Output: scalar (PatchGAN decision per sample).
    """

    def __init__(
        self,
        condition_dim: int = 6,
        output_c_dim: int = 1,
        df_dim: int = 32,
        height: int = 64,
        width: int = 128,
    ):
        super().__init__()
        in_ch = condition_dim + output_c_dim  # 7 channels
        d = df_dim

        # Downsampling by factor 2 each layer (H/2,W/2) -> (H/4,W/4) -> (H/8,W/8) -> (H/16,W/16)
        self.conv1 = ConvBlock(in_ch, d * 1, 5, 2, 2, False, "lrelu")
        self.conv2 = ConvBlock(d * 1, d * 2, 5, 2, 2, True, "lrelu")
        self.conv3 = ConvBlock(d * 2, d * 4, 5, 2, 2, True, "lrelu")
        self.conv4 = ConvBlock(d * 4, d * 8, 5, 2, 2, True, "lrelu")

        # Compute flattened feature size
        h_out = height // 16
        w_out = width // 16
        self.fc = nn.Linear(d * 8 * h_out * w_out, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h_flat = h.view(h.size(0), -1)
        logits = self.fc(h_flat)
        return torch.sigmoid(logits), logits
