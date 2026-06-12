# When Physics Gradients Ambush Your GAN: The 40× Anomaly

> By: DreamNight (Zhiwei Peng)  
> Date: 2026-06-12  
> Code: [github.com/DreamNight/osft-topologygan](https://github.com/DreamNight/osft-topologygan)

---

**TL;DR**: When fine-tuning a topology optimization GAN with physics losses, we found that 99.5% of the gradient attacks the principal subspace that encodes structural knowledge — a concentration 40× stronger than spectral energy distribution predicts. Based on this discovery, we propose OSFT: freeze the principal subspace, fine-tune only the residual subspace, preserving topology with just 0.7% trainable parameters. The directions protected by SVD are nearly orthogonal to those protected by Fisher information (EWC): ρ = 0.117.

---

## 1. The Problem: Catastrophic Forgetting in GAN Fine-Tuning

Topology optimization is a classic engineering problem: find the optimal material distribution given loads and boundary conditions. GANs have recently been used to accelerate this — the generator learns to directly map boundary conditions to optimized topologies, bypassing iterative PDE solves.

But there's a catch: **standard GANs generate topologies that "look right" but don't satisfy real physics constraints** (compliance, volume fraction). The natural fix: add physics losses and fine-tune.

We tried. Full fine-tuning did reduce physics error (MSE ↓18.3%). But at what cost? **IOU dropped from 0.357 to 0.301 — a 15.6% decline.** The generated structures got closer to the target in pixel space, but their topology (how members connect, where holes form) was wrecked.

Classic catastrophic forgetting. The question is: **how exactly does the forgetting happen?**

---

## 2. Discovery: Gradients Don't Overflow — They Precision-Strike

We designed a simple diagnostic (Experiment E4):

1. SVD-decompose each layer's weight matrix of the pretrained GAN
2. Split into principal subspace (τ=0.80, retaining 80% spectral energy) and residual subspace
3. At every fine-tuning step, measure the gradient projection ratio: η = ‖G_res‖² / ‖G_phy‖²

**If gradients were uniformly distributed**, η should be ≈ 1-τ = 0.20 — after all, the residual subspace holds 20% of the spectral energy.

**Actual measurement**:

| Epoch | η |
|-------|-----|
| 10 | 0.0054 |
| 20 | 0.0058 |
| 30 | 0.0059 |
| 40 | 0.0056 |
| 50 | 0.0054 |

```
Spectral energy prediction:   η ≈ 0.20
Actual measurement:           η = 0.005
Gap:                          40×
```

**η ≈ 0.005, constant across all 50 epochs.** This is not noise. This is a structural physical phenomenon.

In plain terms: **the physics gradient doesn't uniformly update all parameter directions — almost all of it (99.5%) attacks the principal subspace that encodes structural knowledge.** It's like trying to repaint a room, but all the force is going into the load-bearing walls.

---

## 3. The Precise Location of Damage: Shallow Decoder Layers

To find where the damage actually happens, we ran CKA (Centered Kernel Alignment) analysis (E5), comparing layer-wise representational similarity between fine-tuned and pretrained models:

| Layer | Type | Full FT CKA | OSFT CKA | OSFT Protection |
|-------|------|------------|----------|-----------------|
| e5 | Encoder | 0.423 | 0.985 | +133% |
| **d2** | **Decoder** | **0.090** | **0.940** | **+943%** |
| d3 | Decoder | 0.162 | 0.645 | +299% |
| d4 | Decoder | 0.245 | 0.668 | +172% |
| **Average** | — | **0.532** | **0.895** | **+68%** |

**d2 after Full FT: CKA = 0.090 — nearly random initialization.** This layer controls topology generation (material connectivity, hole distribution). Full fine-tuning let the physics loss completely wash out this layer's structural knowledge.

OSFT's d2: CKA = 0.940. That's why IOU went from -15.6% to ±0%.

---

## 4. The Fix: OSFT — Protect Load-Bearing Walls, Renovate the Interior

Once you know the damage mechanism, the fix is straightforward:

1. SVD-decompose the pretrained GAN's weight matrices
2. At τ=0.80 energy threshold, split into principal subspace Wᵣ (large singular values, structural knowledge) and residual subspace ΔW (small singular values)
3. **Freeze Wᵣ, train only ΔW**

```
Pretrained Weight W (79M)
    ↓ SVD (τ=0.80)
┌─────────────────────────┬──────────────────────┐
│   Wᵣ — Frozen            │   ΔW — Trainable      │
│   Topology, connectivity │   Physics adaptation  │
│   78.5M params (99.3%)   │   523K params (0.7%)  │
└─────────────────────────┴──────────────────────┘
```

Results (synthetic cantilever, 3 seeds × 50 epochs):

| Method | MSE ↓ | IOU ↑ | CKA | Trainable Params |
|--------|-------|-------|-----|-----------------|
| Pretrained | 0.2481 | 0.3565 | 1.000 | — |
| LoRA-r8 | 0.2498 | 0.3423 | 0.797 | 2.1M |
| Full FT | 0.2027 | **0.3008** | **0.532** | 79.3M |
| Adapter | 0.1981 | 0.3289 | — | 1.1M |
| **OSFT** | **0.1822** | **0.3561** | **0.895** | **523K** |

**OSFT wins across all metrics.** The IOU story is particularly striking: Full FT causes a 15.6% IOU collapse, while OSFT maintains the pretrained IOU level and achieves 10% lower MSE than Full FT.

---

## 5. Why τ=0.80?

τ controls how much of the principal subspace to freeze. Higher τ = more protection, less adaptation room.

We scanned τ ∈ [0.10, 0.99] (E2) and found **three phase transitions**:

- **τ < 0.30**: Connectivity intact (β1 > 0), but MSE is high. Not enough protection.
- **τ = 0.30**: Holes die (β1 → 0). Persistent homology structure collapses first at moderate τ.
- **τ = 0.55–0.60**: Connectivity briefly rebounds (β0 peak), then fully collapses.
- **τ = 0.80**: MSE optimal. Both β0 and β1 are zero — topology is gone at pixel level, but recoverable through residual fine-tuning.

τ=0.80 is the "just right" balance point — enough structural knowledge is preserved, with enough room for physics adaptation. We tried layered τ (FIX1), gradient projection (FIX2), and curriculum learning (E_EXP1). **None beat the simple uniform τ=0.80.** The original design is the optimum.

---

## 6. OSFT Is Not an EWC Approximation (ρ = 0.117)

A natural question: isn't SVD-based OSFT just an approximation of EWC (Elastic Weight Consolidation)? EWC uses the Fisher information diagonal to assess parameter importance; OSFT uses SVD spectral energy. They sound similar.

We answered this empirically (E16): compute the Pearson correlation between SVD principal directions and Fisher diagonal directions across all 11 decomposed layers.

**ρ = 0.117 — near zero.**

SVD captures **spectral structural importance** (how much energy this direction has in the data distribution), while Fisher captures **loss sensitivity** (how much perturbing this parameter affects the training loss). The two are nearly orthogonal.

This means OSFT and EWC protect **different types** of knowledge:
- EWC protects "loss-sensitive" parameters — important for preventing catastrophic forgetting
- OSFT protects "spectrally important" directions — important for preserving generation quality

They are complementary, not competing.

---

## 7. When OSFT Works (and When It Doesn't)

We stress-tested OSFT's boundaries honestly:

### ✅ In-Domain Fine-Tuning: OSFT Excels

Synthetic→synthetic: MSE↓26.6%, IOU maintained. Real (300ep)→real: MSE↓2.0%, IOU matched (OSFT 0.415 vs FT 0.413). **In-domain, OSFT matches Full FT IOU with 0.7% parameters.**

### ❌ Cross-Domain: OSFT Fails

Synthetic→real: OSFT IOU = 0.22, Full FT IOU = 0.40. The "topological knowledge" from synthetic pretraining is nearly useless on real data. OSFT faithfully protects the wrong knowledge. Cross-domain scenarios should use Full FT.

### ⚠️ Pretraining Quality Determines the Ceiling

We ran a scaling law experiment (E13) on real data with 7 pretraining checkpoints (5–300 epochs). The OSFT vs Full FT advantage follows a **U-shape**:

- **Very weak pretraining (5–10ep)**: OSFT wins. The model is too fragile for full updates — OSFT's subspace constraint acts as genuine protection.
- **Moderate pretraining (30–200ep)**: Full FT wins. With enough foundation, full updates adapt more flexibly.
- **Strong pretraining (300ep)**: OSFT wins back. The knowledge is too valuable — protecting it beats updating everything.

**OSFT is a conservative strategy, not an unconditional winner.** Understanding a method's boundaries is more valuable than claiming universal superiority.

---

## 8. An Unverified but Exciting Hypothesis

Why does the physics gradient concentrate 40× in the principal subspace? We don't have a complete answer, but we have a testable hypothesis:

> GAN training data are solutions to elasticity PDEs. During training, the large singular value directions naturally encode the PDE's low-order eigenmodes (bending modes, principal stress paths). The physics loss (compliance minimization) naturally activates these low-order modes, so gradients flow along large singular value directions.

If true, OSFT is not just another PEFT method — it reveals **how data-driven neural networks implicitly encode physical domain priors in their spectral structure.** This connects three usually-separate fields: deep learning theory, PDE numerical analysis, and engineering optimization.

We designed two low-cost verification experiments (SVD direction perturbation + eigenfunction correlation) that would take 3–4 hours, but haven't run them yet.

---

## 9. Why This Isn't a Paper

You might think these results deserve a paper. We wrote a draft. Then we decided not to submit:

1. **GANs have been superseded by diffusion models in topology optimization.** Reviewers will ask "if Diffusion-TO works better, why study GAN fine-tuning?" We have an answer (GAN's single-pass inference is faster for interactive engineering workflows), but it's a defensive position, not an offensive innovation.
2. **The core hypothesis is unverified.** "SVD directions encode PDE eigenmodes" is the most scientifically valuable claim in this work, but the verification experiment hasn't been done. Without it, the paper stops at "we found a phenomenon" — not enough for a top venue.
3. **We'd rather not publish for publishing's sake.** The results are real, the discovery is interesting, the code is reproducible. GitHub + blog post still establish priority and may reach a wider audience.

**If anyone reading this wants to run the verification experiment or try OSFT on other physics domains — get in touch. Happy to collaborate.**

---

## 10. Key Numbers at a Glance

| Finding | Value |
|---------|-------|
| Gradient concentration anomaly | η=0.005 (predicted 0.20, 40× measured) |
| Decoder shallow layer destruction | d2 CKA=0.090 (Full FT vs pretrained) |
| OSFT protection effect | d2 CKA=0.940 (+943%) |
| MSE improvement (synthetic) | ↓26.6% (OSFT vs pretrained) |
| IOU preservation (synthetic) | 0.356 (OSFT) = 0.357 (pretrained) |
| Trainable parameters | 523K / 79M = 0.7% |
| Fisher-SVD correlation | ρ=0.117 (≈ independent) |
| Optimal τ | 0.80 (MSE-optimal, three phase transitions) |
| Cross-domain IOU | OSFT 0.22 vs FT 0.40 |
| Pretraining scaling | U-shape: OSFT wins at extremes, FT wins in middle |

---

## Links

- **Code**: [github.com/DreamNight/osft-topologygan](https://github.com/DreamNight/osft-topologygan)
- **Pretrained Weights & Dataset**: [TBD — will be uploaded to HuggingFace]
- **Full Experiment Logs**: See repo `docs/notes/`
- **Contact**: [GitHub Issues](https://github.com/DreamNight/osft-topologygan/issues)

---

*Thanks for reading. If you work on PEFT, physics-informed neural networks, or topology optimization, I hope this post is useful. Feel free to open a GitHub issue for discussion.*
