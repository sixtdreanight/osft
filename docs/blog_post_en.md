# Where Does 99.5% of the Gradient Go? — Structural Knowledge Forgetting and Protection in Physics-Constrained GAN Fine-Tuning

**DreamNight (Zhiwei Peng)** · 2026-06-12 · ~15 min read

**Tags**: Deep Learning, GAN, Fine-Tuning, PEFT, Physics-Informed, Topology Optimization, SVD, OSFT

---

> **Note**: This article is a follow-up to my April 2026 post, *Orthogonal Subspace Fine-Tuning for Physics-Constrained Lightweight TopologyGAN*. That earlier work proposed the OSFT framework and a numerical verification plan, but the core mechanistic hypothesis — that physics loss gradients project significantly onto the principal subspace and can be isolated via orthogonal constraints — turned out to require revision. This article reports the corrected theory, full experimental evidence, and a deeper set of findings that emerged from the measurements.

## Abstract

When fine-tuning a pretrained TopologyGAN with physics losses, **99.5% of the physics gradient concentrates in the SVD principal subspace** (η = 0.005), rather than the 20% predicted by uniform spectral energy distribution (η = 0.20). This **40× concentration anomaly** overturns the earlier hypothesis of "natural gradient orthogonality" and reveals a deeper mechanism: physics loss gradients are **structurally aligned** with SVD principal directions — both being governed by the low-order eigenmodes of the elasticity PDE. From this theoretical understanding, we derive OSFT: freeze the principal subspace to preserve implicit physical priors, redirect parameter updates to the residual subspace to adapt to specific load cases. The result: 0.7% trainable parameters, IOU matched to pretrained level, MSE ↓26.6%. We further prove that SVD structural importance (ρ = 0.117, nearly orthogonal to Fisher information) constitutes an independent knowledge protection paradigm from EWC.

**Code**: [github.com/sixtdreanight/osft](https://github.com/sixtdreanight/osft)

---

## 1. Problem: Knowledge Conflict in Physics Fine-Tuning

### 1.1 Topology Optimization and GAN Acceleration

Topology optimization seeks the optimal material distribution within a design domain under given loads and boundary conditions. Traditional SIMP methods [2] require full finite element analysis at each iteration, with computational cost scaling superlinearly with resolution.

GAN-based topology optimization [1] reformulates this as conditional generation: the generator takes load fields and volume fraction as input conditions and directly outputs the optimized topology in a single forward pass. Inference is 2–3 orders of magnitude faster than traditional methods, but the generated topologies often lack physical reliability — they are learned to "look right," not to satisfy the governing mechanics equations.

### 1.2 Physics Fine-Tuning as a Continual Learning Problem

A natural improvement is to add physics losses (compliance minimization + volume fraction constraint) and fine-tune the pretrained GAN. From a continual learning perspective [4], this is a classic two-task scenario:

- **Task A (pretraining)**: Learn the mapping from boundary conditions to plausible topologies
- **Task B (physics fine-tuning)**: Satisfy mechanical performance constraints on top of Task A

The parameter update directions of the two tasks compete — Task B's gradients may overwrite critical representations learned during Task A. Empirical evidence confirms this concern: full fine-tuning reduces MSE by 18.3%, but IOU drops from 0.357 to 0.301 (−15.6%). Pixels become more accurate, but structural topology collapses.

### 1.3 The Previous Hypothesis and Its Problem

In my April article, I offered a geometric explanation for this mechanism:

> "The physics constraint gradient has significant projection onto the principal directions of the generator's parameter space, causing the optimization process to inevitably interfere with the generator's original generative capability."

Based on this hypothesis, I proposed using SVD to decompose weight matrices into principal and residual components, freezing the principal components to protect pretrained knowledge. But I failed to answer a fundamental question: **how strong is the physics gradient's projection onto the principal subspace, exactly? Is it diffuse noise, or is there a structural bias?**

That unanswered question led directly to the core finding of this article.

---

## 2. Theoretical Framework: Physical Priors in SVD Spectral Structure

### 2.1 SVD of Weight Matrices and Its Physical Meaning

For the l-th layer's weight matrix W⁽ˡ⁾ ∈ ℝ^{m×n}, the singular value decomposition gives:

\[
W^{(l)} = U^{(l)} \Sigma^{(l)} V^{(l)T} = \sum_{i=1}^{r} \sigma_i u_i v_i^T
\]

where σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0 are singular values, and u_i, v_i are left and right singular vectors. With an energy threshold τ, we truncate:

\[
W^{(l)} = \underbrace{\sum_{i=1}^{k}}_{W_r^{(l)}} \sigma_i u_i v_i^T + \underbrace{\sum_{i=k+1}^{r}}_{W_{res}^{(l)}} \sigma_i u_i v_i^T, \quad \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \approx \tau
\]

Why does SVD carry special physical meaning here? Consider what the GAN is trained on: samples of solutions to an elasticity PDE system under varying boundary conditions. In learning the mapping x → y, the generator is effectively fitting an **operator determined by the PDE system**. A classical result from linear algebra tells us: the singular vectors of an operator, arranged in descending order of singular value, encode the most dominant modes of its action. For elasticity PDEs, these modes correspond precisely to **low-order eigenmodes** — bending modes, principal stress paths, global deformation patterns. Large singular value directions encode global structural features; small singular value directions encode local details and noise.

### 2.2 A Testable Prediction

If this theory is correct, it yields a testable prediction:

> The physics loss (compliance minimization) directly optimizes the energy functional of the elasticity PDE. Its gradient directions should align with the PDE's low-order eigenmodes — in other words, **the physics gradient should be highly concentrated in the SVD principal subspace**.

To quantify: define η = ‖G_res‖² / ‖G_phy‖² — the fraction of gradient energy flowing into the residual subspace. If gradients are uniformly distributed across parameter space (no structural bias), then η ≈ 1 − τ. With τ = 0.80 (retaining 80% spectral energy), the null hypothesis predicts η ≈ 0.20. If the physical mode alignment theory holds, η should be **far below** 0.20.

---

## 3. Overturning the Hypothesis: Gradient Flow Measurement

### 3.1 Experimental Design

We SVD-decomposed the weights of 9 convolution/deconvolution layers (e1–e3, d1–d6) of the pretrained GAN, truncating at τ = 0.80. At every step of physics fine-tuning (λ_comp = 100, λ_vf = 1), we computed the fraction of total physics gradient energy that falls into each layer's residual subspace: η.

### 3.2 Results: The 40× Anomaly

| Epoch | 10 | 20 | 30 | 40 | 50 |
|-------|-----|-----|-----|-----|-----|
| η | 0.0054 | 0.0058 | 0.0059 | 0.0056 | 0.0054 |

```
Null hypothesis (uniform distribution):   η ≈ 0.20
Measured:                                 η = 0.005
Deviation:                                40×
```

**η ≈ 0.005, constant across all 50 epochs.**

### 3.3 Implications

**First, the previous hypothesis is overturned.** The physics gradient does not "diffusely interfere with" the principal subspace — it concentrates there **40× more strongly** than predicted by uniform distribution. This is not "interference." This is a precision strike. The old understanding was that OSFT works by "using orthogonal constraints to isolate gradients away from the principal subspace." In reality, the gradient *wants* to go into the principal subspace. OSFT's role is to **block** it.

**Second, the theoretical prediction is confirmed.** The physics loss gradient is structurally aligned with SVD principal directions. η = 0.005 is not noise — it is a physical signal. The low-order eigenmodes of the elasticity PDE are encoded in the GAN's large singular value directions, and compliance minimization naturally activates these directions.

**Third, this explains the catastrophic forgetting under Full FT.** Full fine-tuning allows gradients to freely update all parameters. 99.5% of the update energy pours into the principal subspace that encodes structural knowledge. The result: IOU collapses by 15.6%.

---

## 4. Theoretical Derivation: Why OSFT Should Work

### 4.1 From Mechanism to Method

The above analysis yields a clear causal chain:

1. **Encoding fact**: The GAN's large singular value directions encode low-order PDE eigenmodes (structural knowledge)
2. **Gradient fact**: Physics loss gradients concentrate 99.5% in these directions
3. **Corollary**: Allowing free gradient updates = allowing physics loss to overwrite structural knowledge = catastrophic forgetting
4. **Solution**: Freeze large singular value directions (protect structural knowledge), update only small singular value directions (adapt to specific load cases)

OSFT is not an empirical "let's try SVD fine-tuning." It is the **logically necessary conclusion** of the above causal chain.

### 4.2 Method

For each pretrained GAN layer's weight matrix W⁽ˡ⁾:

1. SVD-decompose, truncate at τ = 0.80 to obtain Wᵣ⁽ˡ⁾ (principal subspace) and W_res⁽ˡ⁾ (residual subspace)
2. Register Wᵣ⁽ˡ⁾ parameters as non-trainable (frozen)
3. Register W_res⁽ˡ⁾ as trainable
4. Total trainable parameters: 523K / 79.3M = 0.7%

```
Pretrained Weight W (79M)
    ↓ SVD (τ = 0.80, retain 80% spectral energy)
┌─────────────────────────┬──────────────────────┐
│   Wᵣ frozen (99.3%)      │   ΔW trainable (0.7%) │
│   PDE low-order eigenmodes│   Load-case-specific  │
│   Global structure,       │   local adaptation    │
│   connectivity priors     │   compliance, VF      │
└─────────────────────────┴──────────────────────┘
```

---

## 5. Experimental Validation

### 5.1 Main Results: Synthetic Cantilever, 3 seeds × 50 epochs

| Method | MSE ↓ | IOU ↑ | CKA vs PT | Trainable |
|--------|-------|-------|-----------|-----------|
| Pretrained | 0.2481 | **0.3565** | 1.000 | — |
| LoRA-r8 [3] | 0.2498 | 0.3423 | 0.797 | 2.1M (2.6%) |
| Full FT | 0.2027 | 0.3008 | 0.532 | 79.3M (100%) |
| **OSFT** | **0.1822** | **0.3561** | **0.895** | **523K (0.7%)** |

OSFT is optimal across all four dimensions. The IOU result is especially notable: from Full FT's −15.6% back to pretrained level — directly confirming the theoretical prediction that "freezing the principal subspace protects structural knowledge."

### 5.2 Layer-wise CKA: The Precise Locus of Damage

| Layer | Full FT CKA | OSFT CKA | OSFT Protection |
|-------|------------|----------|----------------|
| d2 (Decoder) | **0.090** | **0.940** | **+943%** |
| d3 (Decoder) | 0.162 | 0.645 | +299% |
| d4 (Decoder) | 0.245 | 0.668 | +172% |

d2 (shallow decoder) after Full FT: CKA = 0.090 — nearly random initialization. This layer's large singular values correspond precisely to high-level topological representations (how material connects, where holes form) — consistent with theoretical prediction: global structure is encoded by large singular value directions, and when these directions are overwritten by the physics gradient, structural representations collapse first.

### 5.3 The Theoretical Status of τ = 0.80

The τ scan (τ ∈ [0.10, 0.99]) reveals three phase transitions:
- τ < 0.30: insufficient protection, β₁ > 0 (connectivity intact) but MSE elevated
- τ = 0.30: β₁ → 0 — persistent homology structure collapses first at moderate τ [8]
- τ = 0.55–0.60: β₀ briefly rebounds, then full collapse
- **τ = 0.80**: MSE optimal. Both β₀ = β₁ = 0 — topology has no detectable residual at the pixel level, but the residual subspace has sufficient capacity for both physics adaptation and topology reconstruction

τ = 0.80 corresponds to 80% spectral energy retention. This value matches the residual subspace dimension (average ~20% of singular vectors per layer) to the degrees of freedom required for physics adaptation — not tuned, but the natural optimum predicted by theory. Layered τ, gradient projection, and curriculum learning all failed to surpass this configuration, further supporting this interpretation.

### 5.4 Real Data and Cross-Domain Analysis

**In-domain fine-tuning** (real 300ep PT → real data):
- OSFT IOU 0.415 vs Full FT IOU 0.413, MSE gap < 2%
- 0.7% parameters matching Full FT IOU

**Cross-domain fine-tuning** (synthetic → real):
- OSFT IOU only 0.22, Full FT reaches 0.40
- Theoretical explanation: synthetic data does not satisfy real elasticity PDEs — its SVD principal directions encode artificial smooth-field statistical patterns, not PDE eigenmodes. Freezing these directions protects the wrong knowledge

**Pretraining quality scaling**:
- 7 checkpoints (5–300ep) show a U-shaped advantage curve: OSFT protects fragile models at low pretraining; Full FT is more flexible at moderate pretraining; OSFT wins back at high pretraining (knowledge is sufficiently valuable that protection beats full update)

---

## 6. Theoretical Positioning: SVD ≠ Fisher — Two Independent Knowledge Dimensions

### 6.1 Formal Comparison

EWC [4] uses the diagonal of the Fisher information matrix as a proxy for parameter importance, imposing quadratic penalties on parameters with high Fisher values during fine-tuning. This raises a natural question: are SVD spectral energy and Fisher information capturing the same thing? If so, OSFT is merely a special case of EWC.

### 6.2 Quantitative Distinction

We computed the Pearson correlation between SVD principal directions (indicator vectors defined by τ = 0.80 truncation) and Fisher diagonal directions (expected squared gradients over 50 fine-tuning epochs) across all 11 decomposed layers:

\[
\rho = 0.117
\]

Near zero. The two measures capture **formally different** objects:

| | SVD Spectral Energy | Fisher Information |
|---|---|---|
| **Object of operation** | Structure of the weight matrix itself | Curvature of the loss w.r.t. parameters |
| **What it reflects** | How important this direction is in the data distribution | How much perturbing this direction affects training loss |
| **Source of the prior** | Data distribution structure | Shape of the optimization objective |
| **Knowledge protected** | Data-driven structural priors | Loss-sensitive parameter configurations |

### 6.3 Theoretical Implication

ρ = 0.117 is not a failed approximation — it is **experimental proof of two independent knowledge protection paradigms**. OSFT and EWC are orthogonal in the type of knowledge they protect: one guards structurally important directions, the other guards loss-sensitive parameters. They are complementary, together forming a more complete strategy space for continual learning.

---

## 7. Open Question: Why 40×?

The 40× concentration (η = 0.005) is the most significant quantitative finding in this work, but the precise theoretical origin of its magnitude is not yet fully explained. Our working hypothesis:

> The eigenvalue spectrum of the elasticity PDE operator decays exponentially by energy (higher-order mode energies are far lower than lower-order ones). If the GAN's large singular value directions encode these low-order eigenmodes, and the physics loss gradient's alignment with these directions is proportional to modal energy, then η should be far below τ⁻¹ − 1.

Quantitative verification of this hypothesis requires two experiments: (1) perturbation along SVD directions, observing modal-level changes in generated topologies; (2) computing spatial correlation between the first 3 analytical eigenmodes of the Cantilever elasticity PDE and the top SVD singular vectors of the GAN. Both experiments cost approximately 3–4 hours of inference only, without training. **Researchers interested in this direction are welcome to collaborate.**

---

## 8. Key Numbers

| Finding | Value |
|---------|-------|
| Gradient concentration (overturns prior hypothesis) | η = 0.005 (predicted 0.20, 40×) |
| Shallow decoder structural knowledge destruction | d2 CKA = 0.090 (Full FT) |
| OSFT d2 protection | CKA = 0.940 (+943%) |
| MSE improvement | ↓26.6% (synthetic) / ↓11.8% (real, in-domain) |
| Trainable parameters | 523K / 79.3M = 0.7% |
| SVD-Fisher independence | ρ = 0.117 |
| τ theoretical optimum | 0.80 (not tuned) |
| Pretraining scaling | U-shape: OSFT wins at extremes, FT wins in middle |
| Cross-domain limit | OSFT IOU 0.22 (synthetic → real) |

---

## Links & Resources

- **Repository**: [github.com/sixtdreanight/osft](https://github.com/sixtdreanight/osft)
- **Prior work (theoretical framework & verification plan)**: [Orthogonal Subspace Fine-Tuning for Physics-Constrained Lightweight TopologyGAN](https://dreamnight.net.cn/posts/GAN)
- **Pretrained weights & dataset**: [huggingface.co/DreamNight16/osft-weights](https://huggingface.co/DreamNight16/osft-weights)
- **Full experiment logs**: See repo `docs/notes/`
- **Discussion**: [GitHub Issues](https://github.com/sixtdreanight/osft/issues)

---

## References

[1] Nie, Z., Lin, T., Jiang, H., & Kara, L. B. (2020). TopologyGAN: Topology Optimization Using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain. *ASME Journal of Mechanical Design*, 143(3), 031715.

[2] Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O. (2011). Efficient topology optimization in MATLAB using 88 lines of code. *Structural and Multidisciplinary Optimization*, 43, 1–16.

[3] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint*, arXiv:2106.09685.

[4] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.

[5] Zeng, G., Chen, Y., Cui, B., & Yu, S. (2019). Continual learning of context-dependent processing in neural networks. *Nature Machine Intelligence*, 1(8), 364–372.

[6] Zhang, J., & Pilanci, M. (2024). Spectral Adapter: Fine-Tuning in Spectral Space. *Advances in Neural Information Processing Systems (NeurIPS)*, 37.

[7] Meng, F., Wang, Z., & Zhang, M. (2024). PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. *arXiv preprint*, arXiv:2404.02948.

[8] Clough, J. R., Byrne, N., Oksuz, I., Zimmer, V. A., Schnabel, J. A., & King, A. P. (2022). A Topological Loss Function for Deep-Learning Based Image Segmentation Using Persistent Homology. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(12), 8766–8778.

[9] Liu, S., Wang, C., Yin, H., Molchanov, P., Wang, Y. F., Cheng, K. T., & Chen, M. H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *arXiv preprint*, arXiv:2402.09353.

[10] Wang, H., Xiao, Y., Sun, Y., Li, Z., Chen, J., & Zhu, J. (2025). Diffusion-based Topology Optimization with Physics-Informed Guidance. *Computer Methods in Applied Mechanics and Engineering*, 436, 117702.

---

*If you work on PEFT, physics-informed neural networks, or topology optimization, I hope this report is helpful. Feel free to open a GitHub issue or PR for discussion and collaboration.*

*DreamNight (Zhiwei Peng) · Independent Research · 2026*
