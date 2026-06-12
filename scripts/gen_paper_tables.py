"""Generate LaTeX tables for paper."""
import json, os

results = {}
with open("results/final_eval.json") as f:
    results["E1"] = json.load(f)
with open("results/final/E2_tau_scan_results.json") as f:
    results["E2"] = json.load(f)["results"]
with open("results/final/E10_efficiency_results.json") as f:
    results["E10"] = json.load(f)
cka = {}
with open("results/final/E5_cka_results.json") as f:
    cka = json.load(f)
with open("results/final/E6_svd_dynamics_results.json") as f:
    svd = json.load(f)

os.makedirs("results/tables", exist_ok=True)

# === Table 1: Main Performance Comparison (E1) ===
e1 = results["E1"]
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{Main Performance Comparison on Cantilever Dataset}")
lines.append(r"\label{tab:main_results}")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(r"Method & MSE $\downarrow$ & SSIM $\uparrow$ & IOU $\uparrow$ & LPIPS $\downarrow$ & VFAE $\downarrow$ \\")
lines.append(r"\midrule")

pt = e1["Pre-trained"]
lines.append(f"Pre-trained & {pt['mse']:.4f} & {pt['ssim']:.4f} & {pt['iou']:.4f} & {pt['lpips']:.4f} & {pt['vfae']:.4f} \\\\")
for method in ["LoRA-r8", "Full FT", "Adapter", "OSFT"]:
    m = e1[method]["mean"]
    s = e1[method]["std"]
    lines.append(f"{method} & ${m['mse']:.4f}_{{\\pm {s['mse']:.4f}}}$ & "
                 f"${m['ssim']:.4f}_{{\\pm {s['ssim']:.4f}}}$ & "
                 f"${m['iou']:.4f}_{{\\pm {s['iou']:.4f}}}$ & "
                 f"${m['lpips']:.4f}_{{\\pm {s['lpips']:.4f}}}$ & "
                 f"${m['vfae']:.4f}_{{\\pm {s['vfae']:.4f}}}$ \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open("results/tables/table1_main.tex", "w") as f:
    f.write("\n".join(lines))

# === Table 2: Parameter Efficiency (E10) ===
e10 = results["E10"]
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{Parameter Efficiency Comparison}")
lines.append(r"\label{tab:efficiency}")
lines.append(r"\begin{tabular}{lrrrr}")
lines.append(r"\toprule")
lines.append(r"Method & Trainable Params & Fraction & GPU Memory & Time/Epoch \\")
lines.append(r"\midrule")
for method, m in e10.items():
    lines.append(f"{method} & {m['trainable_params']:,} & {m['trainable_pct']:.1f}\\% & "
                 f"{m['gpu_memory_gb']:.2f} GB & {m['time_per_epoch_s']:.1f}s \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open("results/tables/table2_efficiency.tex", "w") as f:
    f.write("\n".join(lines))

# === Table 3: CKA Similarity (E5) ===
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{CKA Representation Similarity}")
lines.append(r"\label{tab:cka}")
lines.append(r"\begin{tabular}{lccccccccc}")
lines.append(r"\toprule")
lines.append(r"Comparison & Avg & e2 & e3 & e4 & e5 & d2 & d3 & d4 & d5 \\")
lines.append(r"\midrule")
for key in ["Pretrain_vs_FullFT", "Pretrain_vs_OSFT", "Pretrain_vs_LoRA"]:
    name = key.replace("Pretrain_vs_", "")
    r = cka[key]
    avg = r["avg_cka"]
    layers = r["per_layer"]
    line = f"{name} & {avg:.3f}"
    for lname in ["e2", "e3", "e4", "e5", "d2", "d3", "d4", "d5"]:
        line += f" & {layers[lname]:.2f}"
    line += r" \\"
    lines.append(line)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open("results/tables/table3_cka.tex", "w") as f:
    f.write("\n".join(lines))

# === Table 4: SVD Dynamics (E6) ===
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{Effective Rank Evolution During Fine-Tuning}")
lines.append(r"\label{tab:svd_dynamics}")
lines.append(r"\begin{tabular}{lcccccccccc}")
lines.append(r"\toprule")
lines.append(r"Method & 0 & 10 & 20 & 30 & 40 & 50 & $\Delta$ \\")
lines.append(r"\midrule")
for method in ["Full_FT", "OSFT"]:
    snaps = svd[method]["snapshots"]
    start = snaps[0]["avg_effective_rank"]
    end = snaps[-1]["avg_effective_rank"]
    delta = (end - start) / start * 100
    line = method.replace("_", " ") + f" & {start:.1f}"
    for s in snaps[2::2]:  # every 10 epochs
        line += f" & {s['avg_effective_rank']:.1f}"
    line += f" & {delta:+.1f}\\% \\\\"
    lines.append(line)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open("results/tables/table4_svd.tex", "w") as f:
    f.write("\n".join(lines))

# === Table 5: Tau Scan Phase Transitions (E2) ===
e2 = results["E2"]
tau = e2["tau"]
mse = e2["mse"]
ssim = e2["ssim"]
b0 = e2["beta0_preservation"]
b1 = e2["beta1_preservation"]
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{$\tau$-Scan: Phase Transitions}")
lines.append(r"\label{tab:tau_scan}")
lines.append(r"\begin{tabular}{cccccc}")
lines.append(r"\toprule")
lines.append(r"$\tau$ & MSE $\downarrow$ & SSIM $\uparrow$ & $\beta_0$ Pres. & $\beta_1$ Pres. & Eff. Rank \\")
lines.append(r"\midrule")
key_taus = [0.10, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
for t in key_taus:
    idx = tau.index(t)
    lines.append(f"{t:.2f} & {mse[idx]:.3f} & {ssim[idx]:.3f} & {b0[idx]:.2f} & "
                 f"{b1[idx]:.2f} & {e2['effective_rank_avg'][idx]:.0f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open("results/tables/table5_tau_scan.tex", "w") as f:
    f.write("\n".join(lines))

# === Summary markdown ===
with open("results/tables/README.md", "w") as f:
    f.write("# OSFT Paper Tables\n\n")
    f.write("| File | Content |\n")
    f.write("|------|--------|\n")
    f.write("| table1_main.tex | Main performance comparison (E1) |\n")
    f.write("| table2_efficiency.tex | Parameter efficiency (E10) |\n")
    f.write("| table3_cka.tex | CKA representation similarity (E5) |\n")
    f.write("| table4_svd.tex | SVD effective rank dynamics (E6) |\n")
    f.write("| table5_tau_scan.tex | Tau-scan phase transitions (E2) |\n")

print("Tables saved to results/tables/")
for fname in ["table1_main", "table2_efficiency", "table3_cka", "table4_svd", "table5_tau_scan"]:
    print(f"  {fname}.tex")
