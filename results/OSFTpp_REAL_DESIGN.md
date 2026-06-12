# OSFT++ 真正的创新点 —— 从实验数据反推

**原则**: 不引入任何实验数据无法支撑的概念。每个创新点必须有直接的实验证据链。

---

## 一、先问：我们的实验到底告诉了我们什么（别人不知道的）？

### 事实 1: 梯度天然不在残差子空间（E4）

η = ||G_res||² / ||G_phy||² ≈ 0.005

**含义**: 优化器 99.5% 的更新意愿指向主子空间（被 OSFT 冻结），只有 0.5% 指向残差子空间（OSFT 允许训练）。

**这是整个 OSFT 故事里最反直觉的发现。** 原始假设是"梯度自然对齐残差子空间，OSFT 只是顺水推舟"。实验证明恰恰相反：**OSFT 是在逆梯度而行，像一个强制约束，逼着优化器在它不想走的方向上走。**

### 事实 2: 不同层的表征损伤模式完全不同（E5 CKA 逐层）

| 层 | Full FT CKA | OSFT CKA | 差异 |
|----|-----------|---------|------|
| e5 (深层编码器) | 0.423 | 0.985 | +56% |
| d2 (浅层解码器) | 0.090 | 0.940 | **+85%** |
| d3 (浅层解码器) | 0.162 | 0.645 | +48% |
| d4 (中层解码器) | 0.246 | 0.668 | +42% |

**含义**: Full FT 对浅层解码器（d2/d3）的破坏最严重（CKA 低至 0.09），OSFT 对这些层的保护效果也最显著（+85%）。

**推论**: 不是所有层都需要 OSFT。有些层（如 e5）Full FT 也能保留不少表征（CKA 0.423），有些层（如 d2）Full FT 几乎完全摧毁（CKA 0.09）。

### 事实 3: 层冻结实验验证了 d2/d3 是关键（E14）

| 配置 | MSE | 可训参数% |
|------|-----|----------|
| G6 只解冻 d2+d3+d4 | 0.237 | 1.3% |
| G7 Full FT | 0.200 | 100% |
| G5 只解冻 d2+d3 | 0.231 | 1.5% |

**含义**: 仅 1.3% 参数（d2+d3+d4）就能接近 Full FT 性能。这些层是"结构敏感层"。

### 事实 4: OSFT 的跨域退化是可预测的（E15 + E1-SIMP）

| 场景 | 域差距 | OSFT MSE | Full FT MSE | OSFT/FT 比 |
|------|--------|---------|-----------|-----------|
| 合成→合成 | 小 | 0.182 | 0.203 | 0.90 (OSFT 赢) |
| 合成→MBB | 中 | 0.042 | 0.015 | 2.90 |
| 合成→SIMP 悬臂梁 | **大** | 0.304 | 0.046 | **6.61 (OSFT 惨败)** |

**含义**: OSFT/FT 的性能比随域差距单调递增。存在一个"交叉点"——超过它，OSFT 的保护就变成枷锁。

### 事实 5: SVD 不编码物理结构（Eigenmode-SVD）

|r| < 0.3 —— GAN 权重的 SVD 奇异向量与物理本征模态无空间相关性。

**含义**: SVD 分解出来的"主子空间"不包含物理上可解释的结构知识。它保护的是 GAN 内部的某种分布式表征，不是我们以为的"物理拓扑知识"。

---

## 二、从五个事实推导真正的创新点

### 核心矛盾

OSFT 做了一个**全局统一的假设**: 所有层的所有参数方向，只要 SVD 认为是"主的"，就一律冻结。但实验告诉我们：

1. **不同层对冻结的容忍度不同**（E5, E14）—— d2/d3 确实需要保护，e5 不一定
2. **梯度对不同层的"反抗程度"不同**（E4 只测了全局 η，未测逐层 η）
3. **域差距决定了冻结是否合理**（E15, E1-SIMP）—— 域差距大时，冻结的东西可能是错的
4. **SVD 冻结的不是"物理知识"**（Eigenmode-SVD）—— 它冻结的是某种分布式表征

### 真正的创新点

**不是换一个更好的分解方法（Fisher/MP/whatever），而是让系统自己"发现"哪些该冻、哪些该放。**

具体来说：

---

## 创新 1: 逐层梯度对齐诊断（Layer-wise Gradient Alignment Spectroscopy）

**我们要测的东西**: 每一层，梯度有多少比例落在该层的残差子空间里。

$$\eta_l = \frac{\|\Pi_{\text{res}}^{(l)}(\nabla_{W_l}\mathcal{L})\|^2}{\|\nabla_{W_l}\mathcal{L}\|^2}$$

**实验基础**: E4 只测了全局 η ≈ 0.005，但我们有理由相信逐层 η 差异很大：
- E5 显示逐层 CKA 差异极大（0.09~0.99）
- E14 显示逐层重要性差异极大（d2/d3 贡献大部分性能）
- Eigenmode-SVD 显示不同层的 SVD 与物理模态的相关性也不同（e5: r=-0.27, d2: r=0.25, d3: r=0.19）

**假设**: 高 η_l 的层 = 梯度强烈想更新主子空间 = OSFT 的约束在这里最"痛苦" = 这些层最需要放松约束。

**这不是生搬硬套。** 这是 E4 实验的自然延伸——把全局 η 拆成逐层 η。

**可验证性**: 如果这个假设成立，我们应该看到：
- d2/d3 的 η_l **低**（梯度自然在残差子空间，OSFT 约束不痛苦 → 解释了为什么 OSFT 保护 d2/d3 效果好）
- e5 的 η_l **高**（梯度想更新主子空间，但被 OSFT 阻止 → 解释了为什么 e5 的 OSFT CKA 只有 0.985 vs pre-trained 1.000，即仍有 1.5% 损失）

---

## 创新 2: 约束屈服准则（Constraint Yield Criterion）

**从连续介质力学借来的概念**（这不是生搬硬套，是严格的类比）：

在塑性力学中，材料在应力超过屈服强度时会发生不可逆变形的屈服。类比到 OSFT：

- "应力" = 梯度在主子空间的分量（η_l 高 = 应力大）
- "屈服" = 放松约束，允许该层的主子空间参数也被更新
- "屈服准则" = η_l > η_critical → 该层切换到 Full FT

**算法**:
```
for each layer l:
    compute η_l = ||Π_res(∇L)||² / ||∇L||²
    if η_l < η_low:       # 梯度自然在残差空间
        use OSFT (freeze main, train residual)
    elif η_l < η_high:    # 中等约束力
        use leaky OSFT (mostly frozen, small learning rate on main)
    else:                  # 约束力太大，屈服
        use Full FT on this layer
```

**实验基础**: 
- E14 证明不同层可以独立选择训练策略（G2~G6 分别解冻不同层）
- E15 证明域差距大时需要放松约束
- E4 的 η 提供了"何时放松"的定量判据

---

## 创新 3: 结构敏感性加权（Structural Sensitivity Weighting）

**Eigenmode-SVD 实验告诉我们**: SVD 的主方向不编码物理结构。

**但什么编码了物理结构？** 我们可以直接测量。

对于一层权重 W_l，我们想知道：扰动这个权重的不同方向，对输出拓扑的**结构**影响有多大？

定义**结构敏感性**:

$$S_l(v) = \mathbb{E}_{z,c}\left[\text{struct\_dist}\left(G(z,c; W_l + \epsilon v),\; G(z,c; W_l)\right)\right]$$

其中 struct_dist 不是简单的 MSE，而是拓扑结构距离（可以用 Betti 数变化、连通分量数变化、或者 IOU 变化）。

对于权重矩阵 W_l 的每个右奇异向量 v_i（来自 SVD），计算 S_l(v_i)。高 S_l 的奇异向量 → 对拓扑结构敏感 → 需要保护。低 S_l 的奇异向量 → 对拓扑结构不敏感 → 可以安全更新。

**这跟 Fisher Information 的区别**:
- Fisher 度量的是"参数对 LOSS 的敏感性"——这是任务相关的
- 结构敏感性度量的是"参数对输出结构的敏感性"——这是表征相关的

**实验基础**: Eigenmode-SVD 证明 SVD 不编码物理结构。但结构敏感性直接测量"扰动这个方向会改变输出拓扑吗？"，不依赖任何先验假设。

---

## 四、OSFT++ 的完整算法（基于三个创新）

```
Algorithm: OSFT++ (Constraint-Aware Adaptive Fine-Tuning)

Phase 0: 诊断
  1. 逐层 SVD 分解: W_l = U_l Σ_l V_l^T
  2. 计算逐层梯度对齐: η_l (一个前向+反传)
  3. 计算结构敏感性: S_l(v_i) for top-k singular vectors (k 次前向)
  4. 根据 η_l 和 S_l 为每层分配策略

Phase 1: 自适应训练
  for each epoch:
      for each layer l:
          strategy[l] = select_strategy(η_l, S_l)
          # strategy ∈ {OSFT, LeakyOSFT, FullFT}
          
          if strategy == OSFT:
              W_l ← W_l - η · Π_res(∇L)     # 只在残差子空间更新
          elif strategy == LeakyOSFT:
              W_l ← W_l - η · (Π_res(∇L) + γ · Π_main(∇L))  # 允许泄漏
              γ = exp(-η_l / T)              # 温度控制泄漏量
          else:  # FullFT
              W_l ← W_l - η · ∇L             # 无约束更新
      
      每 K 步重新计算 η_l（监测约束力变化）
      如果 η_l 发生突变 → 触发策略重分配
```

### 为什么这不是生搬硬套

| 概念 | 来源 | 实验证据 |
|------|------|---------|
| 逐层 η_l | E4 全局 η 的自然延伸 | E5 逐层 CKA 差异支撑逐层 η 差异的假设 |
| 约束屈服 | 塑性力学类比 | E15 跨域退化 → 约束需要"屈服"的定量证据 |
| 结构敏感性 | 直接测量，非先验 | Eigenmode-SVD 证明 SVD 不行 → 需要直接测量 |
| 三策略混合 | 自然结论 | E14 层冻结实验 → 不同层可以不同策略 |

---

## 五、这个方案要验证的核心假设

### H1: 逐层 η_l 可以预测 OSFT 的有效性

**预测**: η_l 低的层（如 d2/d3）→ OSFT 效果好；η_l 高的层 → OSFT 效果差。

**验证**: 在合成数据上测量逐层 η_l，与 E5 的逐层 CKA 对比。如果 η_l 与 CKA_OSFT - CKA_FullFT 正相关，假设成立。

### H2: 域差距增大 → η_l 升高

**预测**: 合成→SIMP 的 η_l 普遍高于 合成→合成。

**验证**: 在两个数据集上分别测量 η_l，对比分布。

### H3: 结构敏感性 S_l 可以识别"拓扑关键"方向

**预测**: d2/d3 的高 S_l 方向比 e5 多。

**验证**: 逐层计算 S_l 分布，与 E14 层冻结实验结果对比。

---

## 六、与学界的关系（诚实版）

OSFT++ 不是在"应用"Fisher Information 或 Grassmann 流形——那些是工具，不是创新。

OSFT++ 的真正贡献是**发现并解决了 OSFT 的"约束力失配"问题**：

1. OSFT 对所有层施加相同的硬约束 → 有些层不需要（e5），有些层极需（d2/d3）
2. OSFT 对所有训练阶段施加相同的约束 → 域差距大时需要放松
3. OSFT 用 SVD 定义约束方向 → 但这些方向不编码物理结构

这三个问题都是**我们从实验中发现的**，不是从文献中读到的。解决方案（逐层 η 诊断 + 屈服准则 + 结构敏感性）是这些问题**自然推导出的答案**，不是从其他领域搬运来的。

如果这个方案有效，论文的核心贡献陈述是:

> We show that OSFT's effectiveness is determined by a previously unmeasured quantity: the per-layer gradient alignment η_l between the optimization direction and the constrained subspace. By monitoring η_l during training, we can predict when and where subspace constraints become counterproductive, and adaptively relax them. This transforms OSFT from a static regularization method into a constraint-aware adaptive system.
