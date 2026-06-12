# Promo Posts

## Twitter / X (short)

99.5% of physics gradients attack the principal SVD subspace of a pretrained GAN — 40× stronger than predicted.

We found why fine-tuning destroys topology, and built OSFT: freeze 99.3% of weights, match full fine-tuning with just 0.7% parameters.

Code: github.com/sixtdreanight/osft 🧵

---

## Twitter / X (thread — 5 tweets)

1/ When you fine-tune a GAN with physics losses, 99.5% of the gradient doesn't evenly update weights — it precision-strikes the principal SVD subspace that encodes structural knowledge. 40× more concentrated than spectral energy predicts. η=0.005, not 0.20.

2/ Where does the damage land? Shallow decoder layers. d2 CKA drops to 0.090 under full fine-tuning — nearly random. This layer controls topology generation. That's why IOU collapses by 15.6%.

3/ The fix: OSFT. SVD-decompose weights, freeze the principal subspace (99.3% params), fine-tune only the residual (0.7%). Result: IOU preserved at pretrained level, MSE ↓26.6%. d2 CKA stays at 0.940 (+943% vs full FT).

4/ Is this just EWC with SVD? No. We measured the correlation between SVD principal directions and Fisher information diagonals: ρ = 0.117. Near zero. SVD and Fisher protect fundamentally different knowledge. Complementary mechanisms.

5/ Honest limits: OSFT is a conservative strategy — U-shaped advantage curve over pretraining quality. Cross-domain (synthetic→real) fails — OSFT protects the wrong knowledge. τ=0.80 is the sweet spot. Full writeup in the blog. Code: github.com/sixtdreanight/osft

---

## LinkedIn (professional)

I'm sharing a research project I've been working on: OSFT (Orthogonal Subspace Fine-Tuning).

**The Discovery**

When fine-tuning a pretrained topology optimization GAN with physics losses, I measured where the gradient actually flows. The result was surprising: 99.5% of the gradient concentrates in the principal SVD subspace — the part that encodes structural priors like topology and connectivity. Spectral energy distribution predicts ~20% should go there. The measured concentration is 40× stronger.

This explains why standard fine-tuning destroys topology: the physics gradient is attacking the "load-bearing walls" of the model.

**The Solution**

OSFT freezes the principal SVD subspace (99.3% of parameters) and fine-tunes only the residual subspace (0.7%). Key results:

- MSE ↓26.6% (vs pretrained), IOU preserved at pretrained level
- Decoder shallow layer CKA: 0.940 (OSFT) vs 0.090 (Full FT) — +943%
- Fisher-SVD correlation: ρ = 0.117 — proving OSFT and EWC are independent mechanisms

**When It Works**

OSFT is a conservative strategy. It excels at the extremes of pretraining quality (very weak or very strong), while full fine-tuning wins in the middle range. Cross-domain transfer is a known limitation — OSFT requires domain consistency between pretraining and fine-tuning data.

**Why Not a Paper**

GANs in topology optimization are being replaced by diffusion models. The most scientifically interesting hypothesis — that SVD directions encode PDE eigenmodes — still needs verification. I'd rather share the results openly than force a premature publication.

All code, experiments, and a detailed blog post are available at: github.com/sixtdreanight/osft

Happy to discuss or collaborate on the open questions.

---

## 知乎/微博 (中文)

发布了一个研究项目：**OSFT（正交子空间微调）**。

核心发现很简单：用物理损失微调 GAN 时，**99.5% 的梯度冲向了编码结构知识的主子空间**，集中度是理论预测的 40 倍。这解释了为什么全量微调会导致拓扑结构崩溃（IOU ↓15.6%）。

解法：SVD 分解 → 冻结主子空间（99.3% 参数）→ 只微调残差（0.7% 参数）。

关键数字：
- MSE ↓26.6%，IOU 持平预训练水平
- d2 解码器层：OSFT CKA 0.940 vs Full FT 0.090（+943%）
- SVD 保护方向与 Fisher 信息几乎正交（ρ=0.117）——两种独立机制
- 跨域不适用，预训练质量呈 U 型优势曲线

没投论文。GAN 在拓扑优化领域正被扩散模型取代，核心假说（SVD 方向编码 PDE 本征模态）还没验证，不想为发而发。

代码开源，博客有完整分析：github.com/sixtdreanight/osft

欢迎讨论、提 issue、或者一起把那个没做完的验证实验跑了。

---

## Reddit (r/MachineLearning)

**Title**: [R] OSFT: Freeze 99.3% of GAN weights, match full fine-tuning — discovered 99.5% of physics gradients attack the principal SVD subspace

**Body**:

Weird finding: when fine-tuning a pretrained TopologyGAN with physics losses (compliance + volume fraction), 99.5% of the gradient concentrates in the principal singular subspace. Spectral energy predicts ~20%. The measured ratio is η=0.005, not 0.20 — a 40× gap, constant across all 50 epochs.

This gradient-concentration mechanism explains catastrophic forgetting in physics fine-tuning. The damage is concentrated in shallow decoder layers (d2 CKA drops to 0.090 — nearly random).

**OSFT**: SVD-decompose → freeze principal subspace (99.3% params) → fine-tune residual (0.7%).

**Key results** (synthetic cantilever, 3 seeds):
- MSE: OSFT 0.182 vs Full FT 0.203 vs Pretrained 0.248
- IOU: OSFT 0.356 = Pretrained 0.357 vs Full FT 0.301
- CKA vs pretrained: OSFT 0.895 vs Full FT 0.532
- SVD-Fisher correlation: ρ=0.117 (independent mechanisms)

**Limits**: Conservative strategy (U-shaped advantage over pretraining quality). Cross-domain fails. τ=0.80 is the robust optimum.

**Code & full writeup**: github.com/sixtdreanight/osft

Not submitting as a paper — GANs are being overtaken by diffusion models in TO, and the core hypothesis (SVD directions encode PDE eigenmodes) still needs verification. Happy to collaborate on the open questions.
