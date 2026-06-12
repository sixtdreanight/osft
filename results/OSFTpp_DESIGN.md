# OSFT++ : Fisher-Guided Dynamic Subspace Fine-Tuning

**跨学科下一代方案设计**
2026-06-11

---

## 一、OSFT 的核心瓶颈（从实验结果出发）

| 瓶颈 | 实验证据 | 根本原因 |
|------|---------|---------|
| 静态残差子空间 | E1-SIMP: OSFT MSE=0.304 vs Full FT 0.046 | SVD 分解对下游任务无知 |
| τ 固定不变 | E2: β1 在 τ=0.30 死亡，无法恢复 | 阈值不考虑 domain gap |
| 跨域退化 | E15: 2/16 组合负增益 | 主子空间冻结了错误的"知识" |
| 梯度-子空间不对齐 | E4: η≈0.005 | 约束方向与优化方向正交 |

**一句话诊断**: OSFT = 用一把固定的手术刀做所有手术，但每个病人的解剖结构不同。

---

## 二、跨学科灵感矩阵

### 2.1 数学 → 新工具

| 来源 | 概念 | OSFT++ 中的应用 |
|------|------|---------------|
| **随机矩阵理论** | Marchenko-Pastur 定律：谱的 bulk edge λ⁺ 区分信号/噪声 | 自动确定 τ：τ = 1 − #{σᵢ > λ⁺}/n |
| **信息几何** | Fisher 信息矩阵定义参数空间的 Riemannian 度量 | FIM 替代 SVD 做子空间分解 |
| **Grassmann 流形** | k 维子空间构成的流形 Gr(k,n) | 子空间在流形上演化，而非硬冻结 |
| **最优传输** | Sinkhorn 算法，熵正则化 Wasserstein 距离 | 度量 domain gap，控制子空间旋转速度 |
| **谱图理论** | Laplace-Beltrami 特征函数 | 权重矩阵的谱聚类，识别功能模块 |

### 2.2 物理 → 新机制

| 来源 | 概念 | OSFT++ 中的应用 |
|------|------|---------------|
| **量子绝热定理** | 系统参数缓慢变化→保持在瞬时本征态 | τ(t) 的绝热演化：变化速率受谱间隙约束 |
| **重整化群 (RG)** | 尺度依赖的有效理论，relevant/irrelevant 算子分类 | 动态区分 relevant（需保留）和 irrelevant（可丢弃）的参数方向 |
| **相变理论 (Landau)** | 序参量、对称性破缺、临界慢化 | 检测训练中的"表征相变"，触发子空间重组 |
| **自由能最小化** | F = E − T·S，温度控制探索-利用平衡 | 训练温度 T(t): 初始高温（允许探索），退火到低温（冻结最优） |
| **非平衡统计力学** | 涨落-耗散定理 | η(t) 作为"响应函数"，度量子空间约束的有效性 |

### 2.3 计算机科学 → 新算法

| 来源 | 概念 | OSFT++ 中的应用 |
|------|------|---------------|
| **流式算法** | Frequent Directions sketch | 增量 SVD，避免每次重新分解 |
| **Bandit 算法** | UCB, Thompson Sampling | 每层独立决策：冻结/微调/OSFT |
| **退火算法** | 模拟退火，热浴 | 温度调度控制子空间刚性 |
| **元学习** | MAML, Reptile | 学习 τ 的初始化，使其快速适应新任务 |

### 2.4 2025-2026 学界最新进展

| 方法 | 来源 | 与 OSFT++ 的关系 |
|------|------|-----------------|
| **FI-LoRA** | IEEE SPL 2026 | Fisher 信息指导动态秩分配 → OSFT++ 的核心机制 |
| **Sculpting Subspaces** | arXiv 2025.04 | 自适应 SVD 持续学习 → OSFT++ 的子空间演化策略 |
| **Dynamic-Rank Training** | arXiv 2025.08 | 秩崩塌的检测与修复 → OSFT++ 的秩保持机制 |
| **Grassmannian Deep Networks** | AAAI 2026 | 多子空间融合与 Fréchet 均值 → OSFT++ 的几何基础 |
| **DisLoRA** | EMNLP 2025 | 任务特定 SVD 方向 → OSFT++ 的 Fisher-SVD 混合 |
| **FFT Subspace Selection** | arXiv 2025.05 | DCT 加速子空间投影 → OSFT++ 的高效实现 |

---

## 三、OSFT++ 核心架构

### 3.1 统一数学框架

OSFT++ 将微调形式化为 **约束流形优化**：

$$\min_{\theta \in \mathcal{M}(t)} \mathcal{L}_{\text{task}}(\theta)$$

其中 $\mathcal{M}(t)$ 是一个 **时变约束流形**，由三个组件定义：

$$\mathcal{M}(t) = \{\theta : \Pi_{\mathcal{P}(t)} \nabla \mathcal{L} = 0,\;\; \|\theta - \theta_{\text{pre}}\|_{F^{-1}(t)} \leq \epsilon(t)\}$$

- $\mathcal{P}(t)$: 动态保护子空间（由 Fisher 信息确定）
- $F^{-1}(t)$: Fisher 信息度量的逆（定义"距离"的几何意义）
- $\epsilon(t)$: 时变约束半径（温度控制的探索边界）

### 3.2 三个核心创新

#### 创新 1: Fisher 信息指导的子空间分解 (替代 SVD)

**问题**: SVD(W) 只反映权重的结构，不反映任务的重要性。

**方案**: 使用 Fisher 信息矩阵进行分解。

$$
F_{\text{pre}} = \mathbb{E}_{x\sim D_{\text{pre}}}\left[\nabla_\theta \log p(y|x;\theta) \cdot \nabla_\theta \log p(y|x;\theta)^\top\right]
$$

$$
F_{\text{target}} = \mathbb{E}_{x\sim D_{\text{target}}}\left[\nabla_\theta \log p(y|x;\theta) \cdot \nabla_\theta \log p(y|x;\theta)^\top\right]
$$

**三空间分解**（每层独立计算）:

| 子空间 | 定义 | 策略 | 占比 |
|--------|------|------|------|
| **保护子空间** $\mathcal{P}$ | $F_{\text{pre}}$ 高 _且_ $F_{\text{target}}$ 低 | 冻结 | ~60% |
| **可塑子空间** $\mathcal{A}$ | $F_{\text{target}}$ 高 | 全量训练 | ~20% |
| **中性子空间** $\mathcal{N}$ | 两者都低 | 低秩约束微调 | ~20% |

**高效近似**: 使用对角 Fisher（Kronecker-factored 近似），计算量 $O(n_{\text{params}})$ 而非 $O(n_{\text{params}}^2)$。

#### 创新 2: 绝热子空间演化 (替代固定 τ)

**问题**: τ 固定 → 无法适应 domain gap。

**方案**: τ 沿训练动态演化，受谱间隙约束。

**Marchenko-Pastur 自动 τ**:

对于每层权重矩阵 $W \in \mathbb{R}^{m\times n}$，其奇异值谱的 MP 上界为：

$$\lambda^+ = \sigma^2(1 + \sqrt{m/n})^2$$

其中 $\sigma^2$ 从谱的 bulk 估计。信号奇异值 = $\{\sigma_i : \sigma_i > \sqrt{\lambda^+}\}$。

$$\tau_{\text{auto}}(t) = 1 - \frac{\#\{\sigma_i > \sqrt{\lambda^+}\}}{\min(m,n)}$$

**绝热演化方程**:

$$\frac{d\tau}{dt} \leq \frac{\Delta(t)^2}{\|\nabla_\tau \mathcal{L}\|}$$

其中 $\Delta(t)$ 是当前谱间隙（最小信号奇异值与最大噪声奇异值之差）。当谱间隙变小时（接近相变），τ 必须缓慢变化。

**物理直觉**: 量子绝热定理 — 哈密顿量变化速率必须远小于能隙平方。

#### 创新 3: 熵正则化泄漏机制 (替代硬约束)

**问题**: OSFT 硬约束 → 太保守，跨域失败。

**方案**: 引入受控"泄漏"，允许有限梯度流入保护子空间。

**自由能形式**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot \underbrace{D_{\text{KL}}(q(\theta_{\mathcal{P}}) \| p(\theta_{\mathcal{P}}))}_{\text{熵惩罚}}
$$

其中 $q$ 是微调后的分布，$p$ 是预训练分布。$\lambda$ 是逆温度参数。

**温度调度**（模拟退火）:

$$T(t) = T_0 \cdot \left(1 - \frac{t}{T_{\text{max}}}\right)^\alpha + T_{\text{min}}$$

- $T$ 高 → 允许探索，子空间约束松弛
- $T$ 低 → 收敛，子空间约束收紧

**泄漏控制**:

$$\theta^{t+1}_{\mathcal{P}} = \theta^t_{\mathcal{P}} - \eta \cdot \gamma(T) \cdot \Pi_{\mathcal{P}}(\nabla \mathcal{L})$$

其中 $\gamma(T) = \exp(-1/T)$ 是温度控制的泄漏系数。

### 3.3 算法伪代码

```
Algorithm: OSFT++ (Fisher-Guided Dynamic Subspace Fine-Tuning)

Input:  Pretrained weights θ_pre, target data D_target,
        pretraining data D_pre (small subset), initial temp T_0,
        MP significance level α
  
Output: Fine-tuned weights θ*

1.  // Phase 0: Fisher-guided subspace decomposition
2.  for each layer l:
3.      Compute diagonal Fisher F_pre[l] on D_pre_subset
4.      Compute diagonal Fisher F_target[l] on D_target_batch
5.      Classify each parameter direction:
6.          Protected P[l]  ← {i: F_pre[i] > δ_high AND F_target[i] < δ_low}
7.          Plastic A[l]    ← {i: F_target[i] > δ_high}
8.          Neutral N[l]    ← {i: otherwise}
9.      τ_auto[l] ← MP_bulk_edge(W[l])
10.     Initialize layer temperature T[l] = T_0
11.
12. // Phase 1: Adiabatic training loop
13. for t = 1 to T_max:
14.     // Step A: Compute gradient
15.     g ← ∇_θ L_task(θ, D_target_batch)
16.     
17.     // Step B: Project gradient per subspace
18.     g_P ← Π_P(g)   // gradient in protected subspace
19.     g_A ← Π_A(g)   // gradient in plastic subspace  
20.     g_N ← Π_N(g)   // gradient in neutral subspace
21.     
22.     // Step C: Controlled leakage
23.     γ ← exp(-1/T[l])  // temperature-controlled leak
24.     θ_P ← θ_P - η · γ · g_P          // leak into protected
25.     θ_A ← θ_A - η · g_A              // full update plastic
26.     θ_N ← θ_N - η · τ_auto · g_N     // constrained update neutral
27.     
28.     // Step D: Spectral monitoring
29.     Δ[l] ← spectral_gap(θ[l])        // track phase transitions
30.     if Δ[l] < threshold:
31.         trigger_subspace_reorganization(l)  // re-classify P/A/N
32.     
33.     // Step E: Temperature annealing
34.     T[l] ← max(T_min, T[l] * (1 - t/T_max)^α)
35.     
36.     // Step F: Update τ via Marchenko-Pastur
37.     if t % τ_update_freq == 0:
38.         τ_auto[l] ← compute_mp_tau(θ[l])
39.
40. return θ*
```

### 3.4 关键超参数

| 参数 | 含义 | 默认值 | 来源 |
|------|------|--------|------|
| δ_high | Fisher 高分位阈值 | 0.8 quantile | FI-LoRA |
| δ_low | Fisher 低分位阈值 | 0.2 quantile | FI-LoRA |
| T_0 | 初始温度 | 10.0 | 模拟退火 |
| T_min | 最小温度 | 0.1 | 模拟退火 |
| α | 退火指数 | 0.5 | 经验 |
| τ_update_freq | MP 更新频率 | 10 epochs | 效率考量 |
| Δ_threshold | 相变检测阈值 | 0.1 × σ_max | 随机矩阵理论 |

---

## 四、相对于 OSFT 的理论优势

| 维度 | OSFT | OSFT++ |
|------|------|--------|
| **子空间定义** | SVD(W) — 静态、任务无关 | Fisher 信息 — 动态、任务相关 |
| **τ 选择** | 手动固定 | Marchenko-Pastur 自动 |
| **约束刚性** | 硬约束（完全冻结） | 软约束（温度控制泄漏） |
| **跨域处理** | 无机制 | 三空间分解 + 自适应重组 |
| **理论基础** | 经验性 SVD | Fisher + Riemannian + 绝热定理 |
| **相变处理** | 无 | 谱监测 + 触发重组 |

---

## 五、预期结果预测

基于 OSFT 的实验数据，OSFT++ 预期：

| 场景 | OSFT 当前 | OSFT++ 预期 | 理由 |
|------|---------|-----------|------|
| 域内微调（合成→合成）| MSE 0.182 | MSE ~0.15 | Fisher 细化子空间选择 |
| 跨域微调（合成→SIMP）| MSE 0.304 | MSE ~0.10 | 三空间+温度适应 |
| 跨工况泛化 | 2/16 负增益 | 0/16 负增益 | 动态子空间重组 |
| 真实数据 | MSE 0.342 | MSE ~0.30 | Fisher 对齐目标分布 |

---

## 六、实现路线图

### Phase 1: Fisher 分解（1-2 周）
- [ ] 实现对角 Fisher 计算（K-FAC 近似）
- [ ] 在 TopologyGAN 上验证 Fisher 分解的合理性
- [ ] 对比 SVD 分解 vs Fisher 分解的选择差异

### Phase 2: 动态 τ（1 周）
- [ ] 实现 Marchenko-Pastur bulk edge 检测
- [ ] 在训练过程中追踪 τ 的自动变化
- [ ] 验证 τ 与 domain gap 的相关性

### Phase 3: 温度退火（1 周）
- [ ] 实现温度控制的泄漏机制
- [ ] 在合成→SIMP 跨域任务上验证
- [ ] 搜索最优退火调度

### Phase 4: 完整系统（1 周）
- [ ] 集成三个组件
- [ ] 在全部 benchmark 上重新评估
- [ ] 对比 OSFT baseline

### Phase 5: 消融与分析（1 周）
- [ ] 逐个组件消融
- [ ] 理论分析：收敛性、泛化界
- [ ] 可视化：子空间演化轨迹

---

## 七、论文定位

**标题候选**:
- "Fisher-Guided Dynamic Subspace Adaptation for Topology-Preserving Fine-Tuning"
- "Beyond Static SVD: Adiabatic Subspace Evolution for Constrained Neural Adaptation"

**对标方法**: LoRA, AdaLoRA, FI-LoRA, Sculpting Subspaces, DoRA, PiSSA

**核心贡献**:
1. 首次将 Fisher 信息与 SVD 子空间分解统一为单一框架
2. 提出绝热子空间演化理论（物理→ML 的迁移）
3. 三空间分解（保护/可塑/中性）解决跨域泛化问题
4. 在物理约束 GAN 微调场景下系统性验证

**目标会议**: ICML 2027 / NeurIPS 2027 / ICLR 2028
