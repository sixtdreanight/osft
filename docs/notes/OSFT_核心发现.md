# OSFT 核心发现：40 倍梯度集中异常

---

## 一、一个未被解释的矛盾

已知：
- τ = 0.80，即主子空间保留 80% 谱能量
- 残差子空间保留 20% 谱能量

如果梯度均匀分布：η ≈ 1 - τ = 0.20

实测：η = 0.005

**实际集中度是谱能量预测的 40 倍。**

```
谱能量预测:     η ≈ 0.20  (20% 梯度在残差)
实际测量:       η = 0.005 (0.5% 梯度在残差)
差距因子:       40×
```

这不是统计波动。是在所有 50 个 epoch 中恒定的结构性现象。

---

## 二、现有理论无法解释

| 理论 | 预测 η | 差距 |
|------|--------|------|
| 谱能量均匀分布 | 0.20 | 40× |
| Spectral Bias（低频偏好） | 不确定 | 仅定性 |
| Fisher 信息几何（EWC） | 无预测 | ρ=0.117→无关 |
| 随机梯度噪声 | ~0 | 不适用 |

现有 PEFT 文献（LoRA/PiSSA/DoRA）把 SVD 当作压缩工具，**没有人问过 SVD 方向和训练域物理结构的关系。**

现有信息几何文献（EWC/NGD）用 Fisher 度量捕获参数的统计重要性，**ρ=0.117 证明 Fisher 几何和 SVD 谱几何几乎正交**——捕获的是两件独立的事。

---

## 三、一个可验证的假说

**假说**：GAN 的大奇异值方向编码了 PDE 的本征模态。

**推理链**：
```
GAN 训练数据 = PDE (弹性力学) 的解的样本
     ↓
生成器学习到的不是像素分布，而是弹性力学模态
     ↓
大奇异值方向 ≈ 低阶本征函数（弯曲模态、应力路径）
小奇异值方向 ≈ 高阶模态 + 噪声（局部孔洞、边界细节）
     ↓
物理损失（柔度最小化）天然激活低阶模态
     ↓
梯度沿大奇异值方向流动 → η 远低于谱能量预测 → 40× 差距
```

**如果成立**，结论就不是"OSFT 是一个 PEFT 方法"，而是：

> **"训练数据驱动的 GAN 在谱结构上隐式编码了物理域的本征模态；OSFT 通过保护谱结构，实际上是在保护物理先验，而非统计先验。"**

---

## 四、最低成本验证实验

**4.1 SVD 方向扰动实验**

```
取 d2 或 d3 层的权重 W
SVD → u₁, u₂, u₃ (前三个左奇异向量)
沿 W + α·u_i 方向扰动（α 从 -0.5 到 +0.5）
观察生成拓扑的变化

预期:
  u₁ → 控制主梁厚度/整体尺度
  u₂ → 控制受力路径/材料分布方向
  u₃ → 控制支撑结构/边界细节
```

**4.2 本征函数相关性**

```
取 Cantilever 弹性力学的前 3 阶本征模态 (分析解或 FEM 计算)
与 GAN 顶层 SVD 的 u₁, u₂, u₃ 做空间相关性
如果 r > 0.5 → 本征模态-谱方向对齐假说初步成立
```

**成本**：无需 GPU 训练，纯推理 + 简单 FEM，3-4 小时。

---

## 五、论文中的呈现方式

### Discussion 新增段落

> **Why does OSFT work?**  
> A naive prediction based on spectral energy distribution would suggest η ≈ 0.20—that is, 20% of the physical gradient should flow into the residual subspace. However, our measurement shows η = 0.005, indicating a 40× stronger concentration than expected. This hyper-concentration cannot be explained by spectral bias alone. We hypothesize that the large singular vectors of the pretrained GAN weights encode the dominant eigenmodes of the underlying PDE (elasticity), and that the physical loss (compliance minimization) naturally activates these modes. OSFT protects these physically meaningful directions by freezing them. The low correlation between SVD directions and Fisher information (ρ = 0.117) further suggests that OSFT and EWC protect fundamentally different types of knowledge: physical eigenmodes (OSFT) vs. statistical loss sensitivity (EWC).

### 对审稿人的价值

- 回答了"为什么 τ=0.80"（不是经验选择，是物理模态编码的自然结果）
- 解释了"为什么 OSFT 有效"（不是压缩技巧，是保护物理先验）
- 建立了与其他方法的区分（不是另一个 PEFT，是独立的知识保护范式）
- 指向了一个新问题（神经网络的谱结构如何编码 PDE 模态）

---

## 六、长期方向

| 时间 | 目标 |
|------|------|
| 本周 | 验证实验（SVD 方向扰动 + 本征函数相关性） |
| 1-2 月 | 多物理域重复（MBB/L-Beam/Bridge），确认跨域一致性 |
| 6-12 月 | 严格推导：为什么 SGD 训练的 GAN 会将 PDE 模态编码到大奇异值方向 |
| 长期 | 谱-物理对齐作为神经网络可解释性的新框架 |

**这个方向的核心价值**：它连接了三个通常分离的领域——深度学习理论、PDE 数值分析、工程拓扑优化。能做这件事的人必须同时懂这三样，竞争者极少。
