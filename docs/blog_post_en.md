# Where Does 99.5% of the Gradient Go? — Structural Knowledge Forgetting and Protection in Physics-Constrained GAN Fine-Tuning

**DreamNight (Zhiwei Peng)** · 2026-06-12 · ~12 min read

**Tags**: Deep Learning, GAN, Fine-Tuning, PEFT, Physics-Informed, Topology Optimization, SVD

---

## Abstract

When fine-tuning a pretrained topology optimization GAN with physics losses (compliance + volume fraction), we discovered an unexpected phenomenon: **99.5% of the physics gradient does not distribute uniformly across parameter directions — it precision-strikes the principal subspace that encodes structural knowledge.** The concentration is **40× stronger** than what spectral energy distribution predicts. This gradient-concentration mechanism explains why full fine-tuning causes catastrophic forgetting (IOU ↓15.6%), and directly motivates our solution: freeze the principal subspace (99.3% of parameters), fine-tune only the residual subspace (0.7%), preserving topology while adapting to physics constraints.

**Code**: [github.com/sixtdreanight/osft](https://github.com/sixtdreanight/osft)

---

## 1. Background: GANs in Topology Optimization

Topology optimization is a classic engineering problem: find the optimal material distribution given loads and boundary conditions. GANs can accelerate this by orders of magnitude — the generator maps boundary conditions directly to optimized topologies, bypassing iterative PDE solves.

But standard GANs generate topologies that "look right" without satisfying real physics constraints. The natural fix: add physics losses and fine-tune. When we tried full fine-tuning:

> **MSE ↓ 18.3%, but IOU ↓ 15.6%.**

Pixel-level fidelity improved, but structural topology — how members connect, where holes form — was wrecked. Classic catastrophic forgetting. But the question is: **how exactly does the forgetting happen?**

---

## 2. Core Discovery: Gradients Precision-Strike, They Don't Overflow

We designed a diagnostic: SVD-decompose each layer's weight matrix, split into principal subspace (τ=0.80, retaining 80% spectral energy) and residual subspace, then measure the gradient projection ratio at every training step:

\[
\eta = \frac{\|\mathbf{G}_{\text{res}}\|^2}{\|\mathbf{G}_{\text{phy}}\|^2}
\]

**If gradients were uniformly distributed**, η ≈ 1 − τ = 0.20 (since the residual subspace holds 20% of spectral energy).

**Measured**:

| Epoch | 10 | 20 | 30 | 40 | 50 |
|-------|-----|-----|-----|-----|-----|
| η | 0.0054 | 0.0058 | 0.0059 | 0.0056 | 0.0054 |

> **η ≈ 0.005, constant across all 50 epochs. This is not noise. This is a structural physical phenomenon.**

```
Spectral energy prediction:   η ≈ 0.20
Actual measurement:           η = 0.005
Gap:                          40×
```

**99.5% of the physics gradient attacks the principal subspace that encodes structural knowledge.** It's like trying to repaint a room, but all the force goes into the load-bearing walls.

---

## 3. The Precise Location of Damage: Shallow Decoder Layers

Layer-wise CKA (Centered Kernel Alignment) analysis reveals the damage is concentrated in a few critical layers:

| Layer | Full FT CKA | OSFT CKA | Protection |
|-------|------------|----------|------------|
| d2 (Decoder) | **0.090** | **0.940** | **+943%** |
| d3 (Decoder) | 0.162 | 0.645 | +299% |
| d4 (Decoder) | 0.245 | 0.668 | +172% |
| Average | 0.532 | 0.895 | +68% |

> **d2 after Full FT: CKA = 0.090 — nearly random initialization.** This layer controls material connectivity and hole distribution. Full fine-tuning let the physics loss completely wash out its structural knowledge.

---

## 4. The Fix: OSFT

Once the damage mechanism is understood, the fix is straightforward:

1. SVD-decompose pretrained weight matrices
2. Split into principal subspace Wᵣ (large singular values, structural knowledge) and residual subspace ΔW at τ=0.80
3. **Freeze Wᵣ, train only ΔW**

```
Pretrained Weight W (79M)
    ↓ SVD (τ=0.80)
┌─────────────────────────┬──────────────────────┐
│   Wᵣ — Frozen            │   ΔW — Trainable       │
│   Topology, connectivity │   Physics adaptation   │
│   78.5M params (99.3%)   │   523K params (0.7%)   │
└─────────────────────────┴──────────────────────┘
```

**Synthetic cantilever, 3 seeds × 50 epochs**:

| Method | MSE ↓ | IOU ↑ | CKA | Trainable |
|--------|-------|-------|-----|-----------|
| Pretrained | 0.2481 | **0.3565** | 1.000 | — |
| LoRA-r8 | 0.2498 | 0.3423 | 0.797 | 2.1M |
| Full FT | 0.2027 | 0.3008 | 0.532 | 79.3M |
| **OSFT** | **0.1822** | **0.3561** | **0.895** | **523K** |

OSFT achieves 10% lower MSE than Full FT while preserving IOU at the pretrained level. **With 0.7% trainable parameters, it delivers the strongest physics adaptation and the most complete structure preservation.**

---

## 5. τ=0.80: Not Tuned, Inherently Optimal

A τ scan (τ ∈ [0.10, 0.99]) revealed **three phase transitions**:

- **τ < 0.30**: Connectivity intact (β1 > 0), but MSE is high — insufficient protection
- **τ = 0.30**: Holes die (β1 → 0) — persistent homology structure collapses first
- **τ = 0.55–0.60**: Connectivity briefly rebounds, then fully collapses
- **τ = 0.80**: MSE optimal — β0 and β1 both zero but recoverable through residual fine-tuning

We also tried layered τ, gradient projection, and curriculum learning. **None beat the simple uniform τ=0.80 configuration.** The original design is the optimum.

---

## 6. OSFT ≠ EWC: Two Independent Protection Mechanisms

A natural question: isn't SVD-based OSFT just an approximation of EWC (Elastic Weight Consolidation)? Both assess parameter "importance."

We directly measured the Pearson correlation between SVD principal directions and Fisher diagonal directions across all 11 decomposed layers:

\[
\rho = 0.117
\]

> **Near zero.** SVD captures spectral structural importance (how much energy a direction carries in the data distribution); Fisher captures loss sensitivity (how much perturbing a direction affects training loss). The two are nearly orthogonal.

OSFT and EWC protect **different types** of knowledge. They are complementary, not competing.

---

## 7. When It Works — and When It Doesn't

### ✅ In-Domain: OSFT Excels

Synthetic→synthetic: MSE ↓26.6%, IOU preserved. Real (300ep)→real: OSFT IOU 0.415 vs FT IOU 0.413. **Matches Full FT IOU with 0.7% parameters.**

### ❌ Cross-Domain: Not Applicable

Synthetic→real: OSFT IOU 0.22 vs FT IOU 0.40. OSFT faithfully protects the *wrong* knowledge when domains differ. **Use Full FT for cross-domain.**

### ⚠️ Pretraining Scaling: A U-Shaped Advantage

Evaluated across 7 pretraining checkpoints (5–300 epochs):

- **Very weak (5–10ep)**: OSFT wins — model is too fragile, subspace constraint is genuine protection
- **Moderate (30–200ep)**: Full FT wins — sufficient foundation, full updates are more flexible
- **Strong (300ep)**: OSFT wins back — knowledge is too valuable, protection beats full update

> **OSFT is a conservative strategy, not an unconditional winner.** Understanding a method's boundaries is more valuable than claiming universal superiority.

---

## 8. Open Question: Why 40×?

Why does the physics gradient concentrate 40× in the principal subspace? We have a testable hypothesis:

> GAN training data are solutions to elasticity PDEs. During training, the large singular value directions naturally encode the PDE's low-order eigenmodes (bending modes, principal stress paths). The physics loss (compliance minimization) naturally activates these low-order modes, so gradients flow along large singular value directions.

If true, this reveals more than a PEFT trick — it suggests **neural networks implicitly encode physical domain priors in their spectral structure**, connecting deep learning theory, PDE numerical analysis, and engineering optimization.

We have designed low-cost verification experiments (SVD direction perturbation + eigenfunction correlation) but haven't run them yet. **Interested collaborators are welcome.**

---

## 9. Key Numbers

| Finding | Value |
|---------|-------|
| Gradient concentration anomaly | η=0.005 (predicted 0.20, measured 40×) |
| Decoder shallow-layer forgetting | d2 CKA=0.090 (Full FT vs pretrained) |
| OSFT protection | d2 CKA=0.940 (+943%) |
| MSE improvement | ↓26.6% (synthetic) / ↓11.8% (real, in-domain) |
| Trainable parameters | 523K / 79M = 0.7% |
| Fisher-SVD correlation | ρ=0.117 |
| Optimal τ | 0.80 (design optimum, not tuned) |
| Pretraining scaling | U-shape: OSFT wins at extremes, FT wins in middle |
| Cross-domain limit | OSFT IOU 0.22 vs FT 0.40 |

---

## Links & Resources

- **Repository**: [github.com/sixtdreanight/osft](https://github.com/sixtdreanight/osft)
- **Pretrained Weights & Dataset**: Coming soon (will be hosted on HuggingFace)
- **Full Experiment Logs**: See repo `docs/notes/`
- **Discussion**: [GitHub Issues](https://github.com/sixtdreanight/osft/issues)

---

*If you work on PEFT, physics-informed neural networks, or topology optimization, I hope this report is helpful. Feel free to open a GitHub issue or PR for discussion and collaboration.*

*DreamNight (Zhiwei Peng) · Independent Research · 2026*
