# OSFT 一区冲刺：最终执行计划

**核心叙事**: OSFT 什么时候有效、为什么有效、什么时候失效  
**关键发现**: OSFT 效果 = f(预训练质量)，由 Scaling Law + CKA 动态 + Fisher-SVD 等价三层支撑  
**必须验证**: 真实数据失败 → 预训练不足 → 加训后反超（这是完整闭环的最后一块）

---

## 最终实验清单（7 项，4 天，~14h GPU，5070 Ti 单卡串行）

| # | 实验 | 回答的问题 | 训练量 | 时间 |
|---|------|-----------|--------|------|
| E13-S | 合成 Scaling Law | 合成域 K_c？ | 1×300ep预训练 + 7OSFT + 7FT | 2h |
| E13-R | 真实 Scaling Law | 真实域 K_c？OSFT 反超？ | 1×300ep预训练 + 21OSFT + 21FT | 5h |
| E14 | 逐层冻结定位 | 知识存在哪层？ | 7组×50ep | 3.5h |
| E16 | Fisher-SVD 相关性 | 为什么 τ=0.80？ | 无训练 | 2h |
| E17 | CKA 动态演化 | 表征如何变化？ | 无训练 | 1h |
| E18 | UMAP 可视化 | 多样性变化？ | 无训练 | 1h |
| E15 | 多工况泛化 | 记忆 vs 知识？ | 3工况×50ep | 4.5h |

**关键优化**: 预训练只跑 1 次 300ep，在 5/10/30/50/100/200/300ep 存 checkpoint，然后分别跑 OSFT 和 Full FT。每条 Scaling Law 线同时画 OSFT 和 Full FT 两条曲线，交叉点 = K_c。

---

## E13-S + E13-R 合并：双曲线 Scaling Law

### 数据明确

```
E13-S (合成线): synthetic_train.npy → 7 个预训练 checkpoint → 各 OSFT 1 种子
E13-R (真实线): cantilever_train.npy → 7 个预训练 checkpoint → 各 OSFT 3 种子
```

**为什么不合并为一个实验？** 因为预训练数据不同会影响 K_c 的位置。两条线各自独立跑，最后画在同一张图上对比。

### E13-R 的真实 200ep 验证：3 种子（关键加固）

这是整篇论文的转折点。根据 E1 数据，OSFT 种子方差 CV≈4.5%（MSE），单种子有风险。跑 3 个种子只需 25 分钟，但能回答"结果是否稳健"。

### E13-S 实验设计

```
预训练 7 个 checkpoint: 5, 10, 30, 50, 100, 200, 300 epoch (合成数据)
每个 checkpoint → OSFT 50 epoch × 1 种子
指标: MSE, IOU, SSIM, CKA
```

### E13-R 实验设计

```
预训练 7 个 checkpoint: 5, 10, 30, 50, 100, 200, 300 epoch (真实 Cantilever)
每个 checkpoint → OSFT 50 epoch × 3 种子
指标: MSE, IOU, SSIM, CKA
```

### 主图设计：双曲线交叉图（论文 Figure 1）

**每个 checkpoint 同时跑 OSFT 和 Full FT**，画两条曲线：

```
ΔMSE% (vs pretrained)
  |   
  |        OSFT _______________
  |           /                
  |          /   ← K_c (交叉点)  
  |         /__________________  Full FT
  +--------------------------------→ Pretraining Epochs
     5  10  30  50  100  200  300
```

- 交叉点左侧：Full FT > OSFT（预训练不足，重新训练更划算）
- 交叉点右侧：OSFT > Full FT（预训练充分，知识值得保护）
- K_c = 交叉点对应的 epoch

这张图直接回答"什么时候用 OSFT"。合成数据线 + 真实数据线同框，域复杂度差异可视化。

### 执行细节

```python
# 关键：一次预训练，多次 checkpoint
for epoch in range(1, 301):
    trainer.train_epoch(...)
    if epoch in [5, 10, 30, 50, 100, 200, 300]:
        torch.save(gen.state_dict(), f'checkpoints/scaling_s/ep{epoch}.pt')

# 每个 checkpoint → OSFT + Full FT
for ep in [5, 10, 30, 50, 100, 200, 300]:
    gen = load(f'ep{ep}.pt')
    # OSFT
    osft = train_osft(gen, 50ep)
    # Full FT
    ft = train_fullft(gen, 50ep)
```

脚本: `scripts/run_scaling.py`

### 启动前：5 分钟干跑

```bash
# 验证全链路无报错，确认 checkpoint 保存/加载正常
python scripts/run_scaling.py --data synthetic --mode dry-run --max-epochs 5 --ft-epochs 1
```

### E13-R Full FT 也跑 3 种子

两只曲线都有 ±std 误差棒，对比公平。时间 +100 分钟（14 次 FT × 8.5s × 50ep）。

---

## E14：逐层冻结定位

### 实验设计

基于 E5 发现 d2 层 CKA 差距最大 (+943%), 逐层冻结验证：

```
G1: 标准 OSFT (全冻结, 对照)
G2: 只解冻 d2 (冻结其他)
G3: 只解冻 d3
G4: 只解冻 d4
G5: 解冻 d2+d3
G6: 解冻 d2+d3+d4
G7: Full FT (全解冻, 对照)

每配置 50 epoch
指标: IOU, SSIM, MSE
```

### 产出：Layer Importance Map

```
d2 ████████████  ← 最关键
d3 ███████
d4 ███
e5 ██
```

定位为 **teaser figure**（abstract 旁边那张），突出"哪层被保护了"。

---

## E16：Fisher-SVD 相关性

### 关键：用哪个 loss 算 Fisher？

EWC 的精神是"保护对旧任务最重要的参数"，三个候选 loss：

| loss | 含义 | 是否正确 |
|------|------|---------|
| L_G = -E[log D(G(z))] | 对抗平衡下的重要参数 | ❌ 受 D 影响，不稳定 |
| L_physics（合规性+VF） | 物理约束的重要参数 | ❌ 这是新任务，不是旧任务 |
| **L_pretrain（预训练联合损失）** | **预训练任务的重要参数** | ✅ EWC 的正确对应 |

**实施方案**：
```
1. 用预训练模型, 固定 D 的权重（eval mode）
2. 对预训练数据集做 forward: L_total = L_GAN + L_physics
3. backward → 收集梯度平方 → F = E[(∂L/∂θ)²]
4. 在 SVD 方向投影: F_proj = diag(U^T @ diag(F) @ U)
```

如果 ρ > 0.7：SVD 方向 ≈ 预训练重要方向 → τ 截断有理论依据
如果 ρ 0.4-0.6：SVD 提供了 Fisher 的近似，减少 EWC 的矩阵估计误差

### E4 叙事重述

```
旧: 物理梯度天然对齐残差 → η≈0.005 → 假设被推翻
新: OSFT 的保护机制不是梯度正交性, 而是参数子空间约束本身
    SVD 能量 ≈ Fisher 重要性 (E16)
    → OSFT 硬约束 ≈ EWC 软约束, 但无估计误差
    → η≈0.005 恰好证明约束是必要的（而非梯度是天然的）
```

### ρ < 0.7 备用表述

如果 Fisher-SVD 相关性在 0.4-0.6 之间:
> "SVD 能量截断提供了 Fisher 重要性的近似（ρ = 0.xx），这种近似减少了 EWC 的矩阵估计误差，同时将可训参数压缩到 523K。"

不说"闭环"，说"提供了理论联系的初步证据"，仍然有价值。

### 论文结构（E14 移到前面）

```
1. Introduction
2. Method: OSFT
3. Experiments
   3.1 E14: Where is topology knowledge? (Layer Importance Map, teaser)
   3.2 E13: When can it be protected? (Scaling Law, Figure 1)
   3.3 E5+E17: Why is it protected? (CKA + dynamics)
   3.4 E1: How much improvement? (Main performance table)
   3.5 E_Jac + E18: Diversity preservation
   3.6 E15: Generalization across structures
4. Analysis
   4.1 E4+E16: Mechanism — constraint, not alignment
   4.2 E2: τ selection and phase transitions
   4.3 E6: Spectral dynamics
5. Discussion
   6.1 OSFT = EWC + OWM
   6.2 K_c rises with domain complexity
   6.3 Limitations + Future Work (E20 persistent homology)
```

**逻辑**: 先告诉读者知识在哪（E14），再解释在什么条件下能保护它（E13），然后说明保护机制（E5），最后量化效果（E1）。

---

## E17：CKA 动态演化

利用已有 E5 checkpoint 数据，无需重新训练。

```
从 epoch 0,5,10,15,...,50 的 checkpoint 提取 CKA
画每层 CKA vs Epoch 曲线:
  - Full FT: 解码器 CKA 持续下跌
  - OSFT: 解码器 CKA 保持稳定
  - LoRA: 中间状态
```

动态过程比最终一个数值更有说服力。

---

## E18：UMAP 流形可视化

```
采样 5000 个 z ~ N(0,1)
Pre-trained / Full FT / OSFT 各生成图像
提取中间层特征 → UMAP 降维 → 可视化
预期: Full FT 成簇, OSFT 保持连续流形
```

适合作为论文大图。

---

## E15：多工况泛化

**前提**: 需要 MBB Beam / L-Shape / Bridge 数据（从 MEGA 文件夹 2 或 SIMP 生成）

```
Cantilever 预训练 → 其他工况 OSFT 微调
对比: Full FT (上限), OSFT (知识迁移)
```

验证"是记忆数据 vs 保留知识"。

---

## 执行时间表（4 天，单 GPU 串行）

```
Day 1:
  上午: E14 逐层冻结 (7组×50ep, 3.5h)
  下午: E16 Fisher (2h) + E17 CKA 演化 (1h)
  夜间: E13-S 300ep 预训练 (40min) + E13-R 300ep 预训练 (45min, 串行~1.5h共)

Day 2:
  上午: E13-S OSFT×7 + FT×7 (串行, ~1.5h)
  下午: E13-R OSFT×7×3 种子 + FT×7 (串行, ~3h)
  夜间: E13 双曲线画图 + 分析 (2h)

Day 3:
  上午: E18 UMAP (1h)
  下午: E15 多工况 (如有 MEGA 数据; 否则 SIMP 生成 MBB)
  夜间: 结果整合

Day 4:
  上午: 写论文 Section 3 (E13+E14+E5 为核心)
  下午: 写 Section 4 (Analysis) + Discussion
```

### 备用叙事（如果 E13-R 200ep 仍不够）

```
"K_c rises with domain complexity: synthetic K_c ≈ 30-50ep, real K_c > 200ep,
indicating that higher-quality pretraining is necessary for complex domains."

这把"OSFT 在真实数据上需要更多预训练"变成理论上有意义的观察,
而非实验失败。在 Discussion 预留半页。
```

### E15 备选数据方案

```
pip install solidspy  # 已安装
用 main/utils/fem_solver.py 配合 SIMP 求解器批量生成 MBB Beam 样本:
- 尺寸: 2:1 长宽比, 下边两端固定, 上边中点施力
- 200-300 样本即可跑 E15
- 不需要等 MEGA 下载
```

---

## 论文预期结构

```
1. Introduction
   → GAN 微调是一个持续学习问题
   → 知识保护能力受预训练质量支配

2. Method: OSFT
   → SVD 分解 → 冻结 Wr → 微调 dW
   → E16: SVD 能量 ≈ EWC Fisher (理论依据)

3. Experiments
   3.1 E14: Where is topology knowledge? (Layer Map, teaser)
   3.2 E13: When can it be protected? (Scaling Law, Figure 1)
   3.3 E5+E17: Why is it protected? (CKA + dynamics)
   3.4 E1: How much improvement? (Main performance table)
   3.5 E_Jac+E18: Diversity preservation
   3.6 E15: Generalization across structures

4. Analysis
   4.1 E4: η≈0.005 → 机制是约束, 非对齐
   4.2 E2: τ 扫描 + 相位转变
   4.3 E6: SVD 光谱动力学
   4.4 真实数据: 预训练不足 → OSFT 失效 → 加训后恢复

5. Discussion
   → OSFT = EWC + OWM 的特例
   → 知识保护前提: 预训练质量 > K_c
   → 局限: 合成数据 β1 不可测, 需持久同调 (Future Work)
```

---

## 和原计划的差异

| 变化 | 原因 |
|------|------|
| E13 吞并 E19 | 同一条 Scaling 曲线, 节省 6h GPU |
| 真实 200ep 预训练加入 | 把负结果变成 insight, 是闭环最后一块 |
| E20 降为 Future Work | 4 天冲刺风险太高, 用 E2 相位转变做动机 |
| E4 叙事重述 | "假设错了" → "OSFT 是更干净的形式化" |
| E14 定位为 teaser | 和 CKA 互补, 不是竞争 |
