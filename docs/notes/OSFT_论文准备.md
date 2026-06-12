# OSFT 论文准备清单

**状态**: 实验完成，开始写稿  
**目标期刊**: CMAME / Structural Optimization / CAE 一区

---

## 一、论文叙事（核心调整）

**旧叙事**：我们提出了一种 SVD 微调方法  
**新叙事**：我们发现拓扑生成模型中的知识遗忘机制，SVD 恰好是解决方案

**论文主线**：发现 → 解释 → 解决

```
物理微调中，梯度 99.5% 冲向主子空间 (η=0.005)
  ↓
导致解码器浅层知识被覆盖 (d2 CKA=0.09)
  ↓  
冻结主子空间 → 保护结构知识 → 只在残差子空间做物理适配
  ↓
SVD 保护的方向 ≠ Fisher 保护的方向 (ρ=0.117)
  ↓
OSFT 是独立于 EWC/LoRA 的新知识保护范式
```

**Title 建议**：
> Orthogonal Subspace Fine-Tuning: Protecting Pretrained Structural Knowledge in Physics-Constrained GAN Adaptation

关键改动：核心从 "SVD-based PEFT" 变成 "Knowledge Protection"

———

## 二、论文结构 v2

```
1. Introduction
   1.1 拓扑优化 + GAN 的背景
   1.2 微调中的知识遗忘问题
   1.3 核心发现（三层贡献）：
       第一层——发现遗忘机制：
         物理梯度 99.5% 冲向主子空间 (η=0.005)
         灾难性遗忘首先发生在解码器浅层 (d2 CKA=0.09)
       第二层——提出解决方案：
         OSFT 冻结主子空间，将更新重定向至残差子空间
         523K 参数 (0.7%)，IOU 持平，MSE↓26.6%
       第三层——揭示独立机制：
         SVD≠Fisher (ρ=0.117)，OSFT 是与 EWC/LoRA 正交的知识保护范式

2. Method
   2.1 知识遗忘现象：物理微调中的梯度威胁
   2.2 主子空间作为结构知识载体
   2.3 OSFT：正交子空间保护
   2.4 与 EWC/LoRA 的机制对比

3. Experiments
   3.1 E1: 合成数据主性能（Table 1）
   3.2 E13 + E1-real: 预训练质量与OSFT效果（Figure 1, 双曲线）
   3.3 E15: 多工况泛化矩阵（Figure 2, 跨域热力图）
   3.4 E5: CKA 表征保护（Figure 3, d2 +943%）
   3.5 E14: 逐层知识定位（teaser figure, d3关键层）
   3.6 E2: τ扫描与相位转变
   3.7 E10: 参数效率对比

4. Analysis
   4.1 E4: η≈0.005——梯度威胁机制
   4.2 E16: ρ=0.117——OSFT≠EWC的独立机制
   4.3 E6: SVD动力学——减缓光谱崩塌
   4.4 E_Jac: 多样性保护
   4.5 负结果：FIX1/FIX2/E_EXP1验证τ=0.80+9层是稳健鞍点

5. Discussion
   5.1 OSFT=保守性策略——预训练两端优势，中间FT更高上限
   5.2 跨域限制（合成→真实 IOU仅0.22）
   5.3 扩散模型vs GAN——推理速度的工程价值
   5.4 未来方向：持久同调损失修复β1

6. Conclusion
```

---

## 二、论文 Figure 规划

| Figure | 内容 | 状态 | 备注 |
|--------|------|------|------|
| Fig 1 (teaser) | E14 逐层冻结热力图 | ✅ 数据有 | d3 是关键适配层 |
| Fig 2 | E13-R U型曲线 + E1真实对比 | ✅ 数据有 | 预训练质量决定OSFT效果 |
| Fig 3 | E15 泛化矩阵热力图 | 🔄 E15跑中 | 跨域IOU矩阵 |
| Fig 4 | E5 CKA 柱状图 (d2 +943%) | ✅ 已生成 | 表征保护证据 |
| Fig 5 | E2 τ扫描曲线 | ✅ 已生成 | 三个相位转变 |
| Fig 6 | E4 η曲线 + E16 ρ分布 | ✅ 数据有 | 机制分析 |
| Fig 7 | E_Jac 多样性+UMPA | ✅ 数据有 | 多样性保护 |

---

## 三、论文 Table 规划

| Table | 内容 | 状态 |
|-------|------|------|
| Table 1 | E1 合成数据主性能 (5方法×7指标) | ✅ |
| Table 2 | E1 真实数据对比 (50ep vs 300ep) | ✅ |
| Table 3 | E15 泛化矩阵 (4×4, OSFT vs FT) | 🔄 |
| Table 4 | E10 参数效率 | ✅ |
| Table 5 | E5 CKA 逐层 | ✅ |

---

## 四、需要写的内容（按优先级）

### 马上可写（数据已有）

- [ ] Section 3.1: E1 合成数据主性能
- [ ] Section 3.4: E5 CKA 表征保护
- [ ] Section 3.6: E2 τ扫描
- [ ] Section 3.7: E10 参数效率
- [ ] Section 4.1: E4 η≈0.005 机制
- [ ] Section 4.2: E16 ρ=0.117 独立机制
- [ ] Section 4.3: E6 SVD动力学
- [ ] Section 4.4: E_Jac 多样性

### 等E15完成后可写

- [ ] Section 3.2: E13 Scaling + E1真实
- [ ] Section 3.3: E15 泛化矩阵
- [ ] Section 3.5: E14 逐层定位

### 最后写

- [ ] Section 5: Discussion（需要全部数据定稿）
- [ ] Section 1: Introduction（最后写，确保贡献列表准确）
- [ ] Abstract

---

## 五、参考文献整理

### 核心引用（必须引用）

1. TopologyGAN (Nie et al., 2020) — 基础方法
2. 88-line SIMP (Andreassen et al., 2011) — 数据生成方法
3. LoRA (Hu et al., 2021) — PEFT基线
4. EWC (Kirkpatrick et al., 2017) — 持续学习理论基础
5. OWM (Zeng et al., 2019) — 梯度投影理论前身

### 近期相关（建议引用）

6. Spectral Adapter (Zhang & Pilanci, NeurIPS 2024) — SVD微调
7. PSOFT (arXiv 2505.11235, 2025) — 主子空间正交旋转
8. SORSA (arXiv 2409.00055, 2024) — SVD+正交正则
9. SC-LoRA (arXiv 2505.23724, 2025) — 数据驱动正交方向
10. Physics-Informed Diffusion (ICLR 2025) — PDE约束扩散模型
11. Diffusion-TO (ScienceDirect 2026) — 扩散模型做TO
12. TopologyLayer (Clough et al., 2022) — 可微持久同调

### 已有参考文献目录

`机器学习及人工智能优化算法-参考文献/` 中包含：
- 11篇 PDF 参考文献
- pre.txt, pre2.txt（论文草稿）
- ds.txt, copilot.txt（AI辅助研究笔记）

---

## 六、已有论文文本素材

| 文件 | 内容概要 |
|------|---------|
| `彭志炜正交子空间微调.pdf` | 论文初稿（含Introduction, Method, 部分Experiment） |
| `pre.txt` | 论文前置研究笔记（159行） |
| `pre2.txt` | 补充研究笔记（85行） |
| `copilot.txt` | AI辅助分析记录（266行） |
| `ds.txt` | 深度分析笔记（331行） |
| `gemini.txt`, `gpt.txt`, `grok.txt`, `qwen.txt` | 多模型对比分析 |

---

## 七、写稿前的数字核对清单

须论文里出现的关键数字：

- [ ] E1 合成 OSFT MSE=0.1822±0.008, IOU=0.3561, ΔMSE=+26.6%
- [ ] E1 合成 Full FT MSE=0.2027±0.007, IOU=0.3008 (↓15.6%)
- [ ] E1 真实300ep OSFT MSE=0.3255, IOU=0.4148, ΔMSE=+2.0%
- [ ] E1 真实50ep OSFT ΔMSE=-2.2% → 300ep恢复到+2.0%
- [ ] E5 CKA OSFT=0.895, Full FT=0.532, d2 +943%
- [ ] E4 η=0.005 (恒定50 epoch)
- [ ] E16 ρ=0.117
- [ ] E2 τ=0.80 MSE最优, β1在τ>0.30死亡
- [ ] E10 OSFT 523K (0.7%), Full FT 79.3M (100%)
- [ ] E_Jac OSFT Eff.Rank=199.9, Full FT=159.3
- [ ] E14 d3-free IOU=0.483 (合成), Extreme OSFT IOU=0.270
