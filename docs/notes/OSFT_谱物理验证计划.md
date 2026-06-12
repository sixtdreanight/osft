# OSFT 谱-物理对齐假说验证计划

**核心问题**: 神经网络的 SVD 主方向是否编码了 PDE 的本征模态？  
**关键证据**: η=0.005 vs (1-τ)=0.20 → 40 倍梯度集中 → 物理梯度主动寻找特定方向  
**成本**: 纯推理 + 轻量 FEM，3-4 小时，无需 GPU 训练

---

## 一、假说

```
GAN 训练数据 = 弹性力学 PDE 解的样本
     ↓
生成器学到的不是像素分布，而是弹性力学模态
     ↓
大奇异值方向 ≈ 低阶 PDE 本征函数（弯曲模态、应力路径）
小奇异值方向 ≈ 高阶模态 + 噪声（局部孔洞、边界细节）
     ↓
物理损失天然激活低阶模态 → η 远低于谱能量预测 → 40× 差距
```

## 二、验证实验

### 实验 1：SVD 方向扰动（1.5h）

**问题**: 前 k 个奇异向量是否控制结构的不同物理属性？

**方法**:
```
取 d2, d3 层的权重 W
SVD: W = U Σ V^T
取前 3 个左奇异向量 u₁, u₂, u₃

沿 W + α·u_i 扰动，α ∈ [-0.3, -0.1, 0.1, 0.3]
对每个 α 生成拓扑，观察变化

预期:
  u₁ → 控制主梁厚度/整体尺度
  u₂ → 控制受力路径/材料分布方向  
  u₃ → 控制支撑结构/边界细节
```

**替代方案（更直接）**:
```
逐零化奇异值:
  σ₁→0: 观察拓扑丢失什么
  σ₂→0: 观察拓扑丢失什么
  σ₃→0: 观察拓扑丢失什么
```

**产出**: 3×(4+1) = 15 张拓扑对比图，标注"此方向控制 XX 结构属性"

### 实验 2：本征函数相关性（1.5h）

**问题**: SVD 主方向与 PDE 本征函数有显著空间相关吗？

**方法**:
```
1. 用 Solidspy 计算 Cantilever 的前 3 阶弹性力学本征模态
   - 模态 1: 第一弯曲模态
   - 模态 2: 第二弯曲模态  
   - 模态 3: 轴向拉伸模态

2. 取 d2/d3 层的前 3 个左奇异向量 u₁,u₂,u₃
   重塑为空间域 [H, W] 或适当的特征图尺寸

3. 计算空间相关性:
   r_ij = corr(u_i_reshaped, eigenmode_j)
   
4. 如果 max(r) > 0.5 → 假说初步成立
   如果 max(r) < 0.3 → 假说需要修正或放弃
```

**产出**: 3×3 相关性矩阵 + 本征模态 vs SVD 向量的并排可视化

### 实验 3：跨层一致性（0.5h，可选）

**问题**: 不同层的大奇异值方向是否对应同一物理模态？

**方法**:
```
取 e2, e3, e5, d2, d3 五层的前 3 个奇异向量
计算跨层 CKA (已实现)
如果不同层的 u₁ 之间 CKA > 0.8 → 模态编码是全局的
```

**产出**: 跨层大奇异值方向 CKA 矩阵

---

## 三、执行

```bash
# 实验 1: SVD 方向扰动
python scripts/eigenmode_svd_perturb.py --layers d2,d3 --n-vectors 3

# 实验 2: 本征函数相关性
python scripts/eigenmode_correlation.py --layers d2,d3

# 实验 3: 跨层一致性
python scripts/eigenmode_cross_layer.py
```

## 四、论文中的位置

### 如果验证成功（r > 0.5）

Discussion 新增核心段落:

> **Why does OSFT work? A spectral-physical alignment hypothesis.**  
> A naive prediction based on spectral energy distribution would suggest η ≈ 0.20. However, our measurement shows η = 0.005, a 40× stronger concentration. We find that the dominant singular vectors of the pretrained GAN weights exhibit significant spatial correlation (r = X.XX) with the low-order eigenmodes of the underlying elastic PDE. This suggests that the pretrained GAN has implicitly encoded the physical eigenmodes in its weight spectrum. The physical loss gradient, which activates these modes by design, naturally flows along the principal singular vectors, explaining the hyper-concentration. OSFT works not merely as a compression technique, but as a physical prior protection mechanism: freezing the principal subspace preserves the learned PDE eigenmodes.

### 如果验证失败（r < 0.3）

Discussion 改为:

> The 40× gap between η and (1-τ) remains unexplained by existing theories. While we do not find direct spatial correlation between SVD directions and PDE eigenmodes, the hyper-concentration of physical gradients suggests an alternative encoding mechanism that warrants further investigation. We hypothesize that the encoding may be distributed across multiple layers, or may correspond to composite modes not captured by single eigenfunctions.

**无论成功或失败，40 倍差距本身就是一个值得发表的发现。**

---

## 五、与 E15 并行执行

```
GPU: E15 泛化矩阵 (继续跑, ~3h 剩余)
CPU: 实验 1-3 (3-4h, 纯推理 + 轻量 FEM)

两者互不干扰。
```
