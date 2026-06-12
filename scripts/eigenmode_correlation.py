"""Eigenmode-SVD correlation — analytical beam eigenmodes.

1. Euler-Bernoulli cantilever beam eigenmodes (analytic, instant)
2. SVD spatial signature: perturb weight along singular vector, observe output change
3. Correlation: Pearson r between eigenmode field and SVD perturbation field

Uses analytical beam modes instead of FEM — exact for the rectangular domain,
no mesh dependency, zero compute time.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, numpy as np
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 128; OUT = "results/eigenmode"; os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Analytical cantilever beam eigenmodes (Euler-Bernoulli)
# ═══════════════════════════════════════════════════════════════
# For a cantilever beam clamped at x=0, length L:
#   φ_n(x) = cosh(βx) - cos(βx) - σ_n·(sinh(βx) - sin(βx))
#   where σ_n = (cosh(βL) + cos(βL)) / (sinh(βL) + sin(βL))
#   β_1·L = 1.87510, β_2·L = 4.69409, β_3·L = 7.85476, β_4·L = 10.99554

print("Computing analytical cantilever beam eigenmodes (Euler-Bernoulli)...")

L = 1.0  # normalized beam length
beta_L = np.array([1.87510407, 4.69409113, 7.85475744, 10.99554073])
sigma = (np.cosh(beta_L) + np.cos(beta_L)) / (np.sinh(beta_L) + np.sin(beta_L))

# 2D grid coordinates
x = np.linspace(0, L, W)      # 128 points along beam length
y = np.linspace(0, 1.0, H)    # 64 points through thickness
X, Y = np.meshgrid(x, y)      # X[H,W], Y[H,W]

omega = []
eigenmodes = []

for n in range(4):
    beta = beta_L[n] / L  # scale back since phi_n depends on βx
    s = sigma[n]

    # 1D beam mode shape along x
    bx = beta * X  # [H,W]
    phi_1d = (np.cosh(bx) - np.cos(bx)
              - s * (np.sinh(bx) - np.sin(bx)))

    # Extend to 2D: plane sections remain plane
    # u_y ∝ φ(x), with linear dependence on y (neutral axis at y=0.5)
    # For pure bending mode, u_y is primarily φ(x)
    uy = phi_1d  # [H,W], the Y-displacement field

    # Normalize to [-1, 1]
    uy = uy / (np.abs(uy).max() + 1e-12)

    eigenmodes.append(uy.astype(np.float32))
    omega.append(beta_L[n])  # dimensionless frequency parameter

    print(f"  Mode {n+1}: βL={beta_L[n]:.4f} ({'1st' if n==0 else '2nd' if n==1 else '3rd' if n==2 else '4th'} bending)")

print(f"  Eigenmodes ready ({len(eigenmodes)} modes, {eigenmodes[0].shape})")
print(f"  All analytical, zero FEM computation needed.")

# ═══════════════════════════════════════════════════════════════
# 2. GAN SVD → spatial perturbation signature
# ═══════════════════════════════════════════════════════════════
print("\nComputing SVD perturbation spatial signatures...")
from main.model.topologygan import TopologyGANGenerator
from main.utils.data_loader import create_dataloaders

state = torch.load("checkpoints/quickstart/pretrained_generator.pt",
                   map_location=device, weights_only=False)
if "generator_state_dict" in state: state=state["generator_state_dict"]
gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                            variant="unet", height=H, width=W).to(device)
gen.load_state_dict(state, strict=False); gen.eval()

_, _, tl = create_dataloaders("data/synthetic_train.npy", height=H, width=W,
                               batch_size=16, num_workers=0)
batch = next(iter(tl))
conds = batch[0][:8].to(device)
z = torch.randn(8, gen.nz, device=device)
with torch.no_grad():
    baseline = gen(conds, z=z).cpu()

target_layers = {}
for name, mod in gen.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        if any(k in name for k in ['d2','d3','e5']):
            target_layers[name] = mod

results = []
for lname, mod in target_layers.items():
    W_orig = mod.weight.data.clone()
    W_2d = W_orig.view(W_orig.size(0), -1).cpu().numpy()
    U, S, _ = np.linalg.svd(W_2d, full_matrices=False)

    for vi in range(min(3, U.shape[1])):
        u = torch.tensor(U[:,vi], dtype=torch.float32, device=device)
        delta = 0.2 * u.view(-1,1,1,1).expand_as(W_orig)
        mod.weight.data = W_orig + delta
        with torch.no_grad():
            pert = gen(conds, z=z).cpu()
        mod.weight.data = W_orig  # restore

        diff = (pert - baseline).abs().mean(dim=0)[0].numpy()  # [H,W]

        for mi, em in enumerate(eigenmodes):
            r, p = pearsonr(diff.ravel(), em.ravel())
            results.append({
                'layer':lname, 'svd_vec':vi+1, 'eigenmode':mi+1,
                'omega':float(omega[mi]), 'pearson_r':float(r), 'p_value':float(p)
            })

    print(f"  {lname}: S[:5]=[{', '.join(f'{s:.0f}' for s in S[:5])}]")

# ═══════════════════════════════════════════════════════════════
# 3. Results
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SVD↔Eigenmode Correlation (analytical beam modes)")
print(f"{'='*70}")
print(f"{'Layer':<15} {'SVD#':>5} {'Mode#':>6} {'βL':>8} {'r':>10} {'p':>10}")
print("-"*55)
for r in sorted(results, key=lambda x: abs(x['pearson_r']), reverse=True)[:10]:
    sig="***" if r['p_value']<0.001 else "**" if r['p_value']<0.01 else "*" if r['p_value']<0.05 else ""
    print(f"{r['layer']:<15} {r['svd_vec']:>5} {r['eigenmode']:>6} {r['omega']:>8.1f} "
          f"{r['pearson_r']:>10.4f} {r['p_value']:>10.4f} {sig}")

best = max(results, key=lambda x: abs(x['pearson_r']))
print(f"\nBest: {best['layer']} SVD#{best['svd_vec']} vs Mode#{best['eigenmode']}"
      f" (βL={best['omega']:.1f}), r={best['pearson_r']:.4f}")

abs_r = abs(best['pearson_r'])
if abs_r > 0.5:   print("✅ |r|>0.5: Eigenmode-SVD alignment SUPPORTED")
elif abs_r > 0.3:  print("⚠️  0.3<|r|<0.5: Weak evidence")
else:              print("❌ |r|<0.3: No spatial correlation. Encoding is distributed/non-spatial.")

# Save visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i in range(4):
    axes[0,i].imshow(eigenmodes[i], cmap='RdBu_r', origin='lower')
    axes[0,i].set_title(f'Mode {i+1} (βL={omega[i]:.1f})', fontsize=10)
    axes[0,i].axis('off')

# Show best-matching diff fields (top 4)
top4 = sorted(results, key=lambda x: abs(x['pearson_r']), reverse=True)[:4]
for j, r in enumerate(top4):
    lname, vi = r['layer'], r['svd_vec']-1
    mod = target_layers[lname]
    W = mod.weight.data.clone()
    W_2d = W.view(W.size(0),-1).cpu().numpy()
    U = np.linalg.svd(W_2d, full_matrices=False)[0]
    u = torch.tensor(U[:,vi], dtype=torch.float32, device=device)
    mod.weight.data = W + 0.2*u.view(-1,1,1,1).expand_as(W)
    with torch.no_grad(): pert = gen(conds, z=z).cpu()
    mod.weight.data = W
    diff_img = (pert-baseline).abs().mean(dim=0)[0].numpy()
    axes[1,j].imshow(diff_img, cmap='hot', origin='lower')
    axes[1,j].set_title(f'{lname} SVD#{vi+1}\nr={r["pearson_r"]:.3f} vs Mode{r["eigenmode"]}', fontsize=9)
    axes[1,j].axis('off')

plt.suptitle('Eigenmode (Analytical Beam) ↔ SVD Spatial Correlation', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/eigenmode_correlation.png', dpi=150); plt.close()

# Also save the eigenmodes as a standalone figure
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 3.5))
for i in range(4):
    axes2[i].imshow(eigenmodes[i], cmap='RdBu_r', origin='lower')
    axes2[i].set_title(f'Mode {i+1}: {"1st" if i==0 else "2nd" if i==1 else "3rd" if i==2 else "4th"} bending\nβL={omega[i]:.4f}', fontsize=11)
    axes2[i].axis('off')
plt.suptitle('Cantilever Beam Analytical Eigenmodes (Euler-Bernoulli)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/analytical_eigenmodes.png', dpi=150); plt.close()

with open(f'{OUT}/eigenmode_correlation.json','w') as f: json.dump(results, f, indent=2)
print(f"\nSaved {OUT}/eigenmode_correlation.png + .json + analytical_eigenmodes.png")
print("Done: analytical eigenmode ↔ SVD correlation")
