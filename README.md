# OSFT: Orthogonal Subspace Fine-Tuning for Physics-Constrained TopologyGAN

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

**OSFT** preserves pretrained structural knowledge during physics-constrained GAN fine-tuning by freezing the principal singular subspace and restricting updates to the residual subspace — achieving **99.3% parameter reduction** (523K trainable / 79M total) while matching or exceeding full fine-tuning on topology preservation.

### Key Results at a Glance

| Metric | Pre-trained | Full FT | LoRA-r8 | Adapter | **OSFT (ours)** |
|--------|------------|---------|---------|---------|-----------------|
| MSE ↓ | 0.2481 | 0.2027 | 0.2498 | 0.1981 | **0.1822** |
| SSIM ↑ | 0.0543 | 0.0573 | 0.0525 | 0.0646 | **0.0723** |
| IOU ↑ | 0.3565 | 0.3008 | 0.3423 | 0.3289 | **0.3561** |
| CKA vs Pretrained | 1.000 | 0.532 | 0.797 | — | **0.895** |
| Trainable Params | — | 79.3M | 2.1M | 1.1M | **523K (0.7%)** |

*Synthetic Cantilever, 3 seeds × 50 epochs. CKA measures representation similarity with pretrained model.*

### Core Discovery: The 40× Gradient Concentration Anomaly

When fine-tuning a pretrained TopologyGAN with physical losses (compliance + volume fraction), we measured where the gradient flows relative to the SVD principal subspace:

```
Spectral energy prediction:  η ≈ 0.20  (20% of gradient in residual subspace)
Actual measurement:         η = 0.005 (0.5% of gradient in residual subspace)
Concentration factor:       40×
```

**99.5% of the physical gradient attacks the principal subspace** — the subspace that encodes structural priors (topology, connectivity). Full fine-tuning lets this gradient overwrite pretrained knowledge, causing IOU to drop by 15.6%. OSFT blocks this by freezing the principal subspace.

### Mechanism

```
Pretrained Weight W (79M params)
    ↓ SVD (τ = 0.80, retain 80% spectral energy)
┌─────────────────────────┬──────────────────────┐
│   Wᵣ (frozen)            │   ΔW (trainable)      │
│   Structural knowledge   │   Physical adaptation │
│   Protects IOU, CKA      │   Reduces MSE, VF     │
│   78.5M params (99.3%)   │   523K params (0.7%)  │
└─────────────────────────┴──────────────────────┘
```

### Why OSFT ≠ EWC (ρ = 0.117)

We compared SVD principal directions with Fisher Information Matrix diagonal directions. The Pearson correlation across all 11 decomposed layers is **ρ = 0.117** — near zero. SVD and Fisher capture fundamentally different knowledge dimensions. OSFT is an independent mechanism from EWC, not an approximation of it.

### Installation

```bash
# Clone
git clone https://github.com/DreamNight/osft.git
cd osft

# Install
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With visualization support
pip install -e ".[viz]"
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU (12GB+ VRAM recommended).

### Quick Start

```bash
# 1. Download pretrained weights and dataset (see links below)
# 2. Run the main experiment suite
python -m main.experiments.run_all --experiment E1 --data synthetic --seeds 3

# 3. Run OSFT fine-tuning
python -m main.osft.trainer \
    --pretrained checkpoints/quickstart/pretrained_generator.pt \
    --data data/synthetic_train.npy \
    --tau 0.80 \
    --epochs 50
```

### Experiment Index

| ID | Name | Key Finding |
|----|------|-------------|
| E1 | Main benchmark | OSFT: MSE↓26.6%, IOU preserved, 523K params |
| E2 | τ (energy threshold) scan | τ=0.80 is the robust optimum |
| E4 | Gradient projection flow | η≈0.005 — 99.5% gradient in principal subspace |
| E5 | CKA representational similarity | d2 decoder: OSFT 0.940 vs FT 0.090 (+943%) |
| E6 | SVD spectral dynamics | OSFT slows spectral collapse by 4pp |
| E10 | Parameter efficiency | 523K trainable = 0.7% of full model |
| E13 | Pretraining scaling law | U-shaped OSFT advantage curve on real data |
| E14 | Per-layer freeze ablation | d3 is the critical adaptation layer on synthetic |
| E16 | Fisher-SVD correlation | ρ=0.117 — independent mechanisms |
| E_Jac | Jacobian manifold diversity | OSFT: 199.9 effective rank vs FT: 159.3 |

All results are in `results/`. Detailed analysis in `docs/notes/`.

### Project Structure

```
osft/
├── main/
│   ├── osft/              # Core OSFT implementation
│   │   ├── config.py          # OSFTConfig dataclass
│   │   ├── decomposer.py      # SVD decomposition engine
│   │   ├── trainer.py         # Training loop
│   │   ├── subspace_layers.py # SubspaceLinear layers
│   │   └── checkpoint.py      # Checkpoint I/O
│   ├── model/             # GAN models
│   │   ├── topologygan.py     # TopologyGAN generator + discriminator
│   │   └── physics_loss.py    # Compliance + VF + topology losses
│   ├── baselines/         # Comparison methods
│   │   ├── full_finetune.py
│   │   ├── lora.py
│   │   └── adapter.py
│   ├── eval/              # Evaluation suite
│   │   ├── metrics.py         # MSE, SSIM, IOU, LPIPS, Betti
│   │   ├── spectral.py        # CKA, SVD dynamics, effective rank
│   │   ├── latent_geometry.py # UMAP, Jacobian analysis
│   │   ├── fem_validator.py   # FEM physical validation
│   │   └── visualize.py       # Plotting utilities
│   ├── utils/             # Data & FEM utilities
│   │   ├── data_loader.py
│   │   ├── fem_solver.py
│   │   └── logger.py
│   └── experiments/       # Experiment orchestration
│       └── run_all.py
├── tests/                 # pytest test suite
├── scripts/               # Data generation & conversion
├── results/               # Experiment outputs (JSON + figures)
├── data/                  # Dataset previews (full .npy files hosted separately)
└── docs/
    └── notes/             # Research notes (Chinese)
```

### Data & Model Weights

- **Datasets**: [Baidu Netdisk / HuggingFace — link TBD]
- **Pretrained checkpoints**: [Baidu Netdisk / HuggingFace — link TBD]
- **Data format**: 64×128 Cantilever beam topology optimization, 7-channel (density + 6 BC fields)

### Limitations (Honest Assessment)

1. **Cross-domain weakness**: Synthetic→real transfer yields IOU=0.22 (vs Full FT IOU=0.40). OSFT requires domain consistency between pretraining and fine-tuning.
2. **Pretraining quality dependency**: OSFT effect ≈ f(pretraining quality). With weak pretraining (50ep real), Full FT wins. At 300ep, OSFT matches or exceeds.
3. **Diffusion models**: For pure generation quality, diffusion-based TO methods outperform GANs. OSFT targets interactive engineering workflows where GAN's single-pass inference is essential.

### Citation

```bibtex
@software{peng2026osft,
  author = {Zhiwei Peng},
  title = {{OSFT}: Orthogonal Subspace Fine-Tuning for Physics-Constrained TopologyGAN},
  year = {2026},
  url = {https://github.com/DreamNight/osft},
}
```

### License

MIT — see [LICENSE](LICENSE).

---

<a name="中文"></a>
## 中文

**OSFT（正交子空间微调）** 通过对预训练 GAN 权重做 SVD 分解，冻结主子空间（保留结构知识），仅微调残差子空间（适配物理约束），在仅 **0.7% 可训参数**（523K / 79M）下实现与全量微调相当或更优的拓扑保真度。

### 核心结果一览

| 指标 | 预训练 | 全量微调 | LoRA-r8 | Adapter | **OSFT（本文）** |
|------|--------|---------|---------|---------|-----------------|
| MSE ↓ | 0.2481 | 0.2027 | 0.2498 | 0.1981 | **0.1822** |
| SSIM ↑ | 0.0543 | 0.0573 | 0.0525 | 0.0646 | **0.0723** |
| IOU ↑ | 0.3565 | 0.3008 | 0.3423 | 0.3289 | **0.3561** |
| CKA vs 预训练 | 1.000 | 0.532 | 0.797 | — | **0.895** |
| 可训参数 | — | 79.3M | 2.1M | 1.1M | **523K (0.7%)** |

*合成悬臂梁数据，3 个随机种子 × 50 epoch。CKA 衡量与预训练模型的表征相似度。*

### 核心发现：40 倍梯度集中异常

在用物理损失（柔度 + 体积分数）微调预训练 TopologyGAN 时，我们测量了梯度相对于 SVD 主子空间的流向：

```
谱能量均匀分布预测:   η ≈ 0.20  (20% 梯度流向残差子空间)
实际测量:             η = 0.005 (0.5% 梯度流向残差子空间)
集中因子:              40×
```

**物理梯度的 99.5% 冲向了主子空间**——这正是编码结构先验（拓扑、连通性）的空间。全量微调放任这种冲击，导致 IOU 下降 15.6%。OSFT 通过冻结主子空间来阻断这种知识破坏。

### 机制

```
预训练权重 W (79M 参数)
    ↓ SVD 分解 (τ = 0.80, 保留 80% 谱能量)
┌─────────────────────────┬──────────────────────┐
│   Wᵣ (冻结)              │   ΔW (可训)           │
│   结构/拓扑知识           │   物理约束适配         │
│   保护 IOU, CKA          │   降低 MSE, VF        │
│   78.5M 参数 (99.3%)     │   523K 参数 (0.7%)    │
└─────────────────────────┴──────────────────────┘
```

### OSFT ≠ EWC (ρ = 0.117)

我们比较了 SVD 主方向与 Fisher 信息矩阵对角线方向。11 层分解层的 Pearson 相关系数平均为 **ρ = 0.117**——接近零相关。SVD 和 Fisher 捕获的是两个几乎正交的知识维度。OSFT 是独立于 EWC 的新机制。

### 安装

```bash
git clone https://github.com/DreamNight/osft.git
cd osft
pip install -e .            # 基础安装
pip install -e ".[dev]"     # 含测试工具
pip install -e ".[viz]"     # 含可视化
```

**环境要求**: Python ≥ 3.10, PyTorch ≥ 2.0, CUDA GPU (推荐 12GB+ 显存)。

### 快速开始

```bash
# 1. 下载预训练权重和数据集（链接见下方）
# 2. 运行主实验
python -m main.experiments.run_all --experiment E1 --data synthetic --seeds 3

# 3. 单独运行 OSFT 微调
python -m main.osft.trainer \
    --pretrained checkpoints/quickstart/pretrained_generator.pt \
    --data data/synthetic_train.npy \
    --tau 0.80 \
    --epochs 50
```

### 实验索引

| 编号 | 名称 | 关键发现 |
|------|------|---------|
| E1 | 主性能对比 | OSFT: MSE↓26.6%, IOU 持平, 523K 参数 |
| E2 | τ 阈值扫描 | τ=0.80 是稳健最优解（MSE 最优 + 三个相位转变） |
| E4 | 梯度投影流 | η≈0.005——99.5% 梯度冲向主子空间 |
| E5 | CKA 表征相似度 | d2 解码器层：OSFT 0.940 vs FT 0.090（+943%） |
| E6 | SVD 谱动力学 | OSFT 减缓光谱崩塌 4 个百分点 |
| E10 | 参数效率 | 523K 可训 = 完整模型的 0.7% |
| E13 | 预训练 Scaling Law | 真实数据呈 U 型（ep5/10 OSFT 赢 → ep30-200 FT 赢 → ep300 OSFT 赢） |
| E14 | 逐层冻结消融 | d3 是合成数据上的关键适配层（+35% IOU） |
| E16 | Fisher-SVD 相关性 | ρ=0.117——两种机制独立 |
| E_Jac | 雅可比流形多样性 | OSFT 有效秩 199.9 > FT 159.3 |

所有结果数据在 `results/` 目录，详细分析见 `docs/notes/`。

### 项目结构

见上方英文部分 [Project Structure](#project-structure)。

### 数据与模型权重

- **数据集**: [百度网盘 / HuggingFace — 待上传]
- **预训练权重**: [百度网盘 / HuggingFace — 待上传]
- **数据格式**: 64×128 悬臂梁拓扑优化，7 通道（密度 + 6 个边界条件场）

### 局限性（诚实评估）

1. **跨域限制**: 合成→真实迁移时 OSFT IOU 仅 0.22（Full FT 可达 0.40）。OSFT 依赖预训练域与微调域的分布一致性。
2. **预训练质量依赖**: OSFT 效果 ≈ f(预训练质量)。弱预训练（50ep 真实数据）下 Full FT 更优；300ep 预训练下 OSFT 持平或反超。
3. **扩散模型对比**: 纯生成质量上扩散模型优于 GAN。OSFT 面向的是需要 GAN 单次前向推理速度的交互式工程优化场景。

### 引用

```bibtex
@software{peng2026osft,
  author = {Zhiwei Peng},
  title = {{OSFT}: Orthogonal Subspace Fine-Tuning for Physics-Constrained TopologyGAN},
  year = {2026},
  url = {https://github.com/DreamNight/osft},
}
```

### 许可

MIT — 详见 [LICENSE](LICENSE)。
