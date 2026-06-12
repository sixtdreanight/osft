"""OSFT configuration dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OSFTConfig:
    # SVD decomposition
    energy_threshold: float = 0.80
    target_layers: List[str] = field(
        default_factory=lambda: ["deconv", "d1", "d2", "d3", "d4", "e1", "e2", "e3"]
    )

    # Training
    n_epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Loss weights
    lambda_gan: float = 1.0
    lambda_l1: float = 100.0         # L1 reconstruction (matching original L1_lambda)
    lambda_vf: float = 1.0            # Volume fraction constraint
    lambda_comp: float = 100.0        # Compliance constraint
    lambda_orth: float = 0.01         # Orthogonality constraint
    lambda_ksv: float = 0.001         # Singular value constraint

    # Architecture
    generator_variant: str = "se_res_unet"
    img_height: int = 64
    img_width: int = 128
    input_c_dim: int = 3
    output_c_dim: int = 1
    condition_dim: int = 6
    gf_dim: int = 64
    df_dim: int = 16

    # Hardware
    use_amp: bool = True              # Automatic Mixed Precision
    gradient_accumulation_steps: int = 2
    num_workers: int = 4
    device: str = "cuda"

    # Logging
    log_every: int = 10
    eval_every: int = 5
    save_every: int = 20
    checkpoint_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
