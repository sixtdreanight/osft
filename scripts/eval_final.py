"""Evaluate all final experiment checkpoints."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from main.model.topologygan import TopologyGANGenerator
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = create_dataloaders(
    "data/synthetic_train.npy", height=64, width=128, batch_size=16, num_workers=0)

def load_and_eval(name, ckpt_path, variant="unet", gf_dim=64):
    if not os.path.exists(ckpt_path):
        return None
    gen = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=gf_dim,
        variant=variant, height=64, width=128,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    gen.load_state_dict(state, strict=False)
    gen.eval()
    m = evaluate_model(gen, test_loader, device)
    del gen; torch.cuda.empty_cache()
    return {k: round(float(v), 6) for k, v in m.items()}

BASE = "results/final"
methods = [
    ("Full FT", "full_ft_Cantilever", "full_ft_best.pt"),
    ("LoRA-r8", "lora_Cantilever", "full_ft_best.pt"),
    ("Adapter", "adapter_Cantilever", "full_ft_best.pt"),
    ("OSFT", "osft_Cantilever", "osft_best.pt"),
]

# Pre-trained baseline
pt = load_and_eval("Pre-trained", "checkpoints/quickstart/pretrained_generator.pt")

all_results = {"Pre-trained": pt}
for method, dir_prefix, best_file in methods:
    vals = []
    for seed in [0, 1, 2]:
        path = f"{BASE}/{dir_prefix}_S{seed}/{best_file}"
        m = load_and_eval(f"{method} S{seed}", path)
        if m:
            vals.append(m)
    if vals:
        avg = {}
        for k in vals[0]:
            avg[k] = round(float(np.mean([v[k] for v in vals])), 6)
        std = {}
        for k in vals[0]:
            std[k] = round(float(np.std([v[k] for v in vals])), 6)
        all_results[method] = {"mean": avg, "std": std, "seeds": vals}

# Print table
print("\n" + "=" * 90)
print(f"{'Method':<15} {'MSE':>12} {'SSIM':>8} {'IOU':>8} {'MAE':>8} {'PSNR':>8} {'VFAE':>8}")
print("-" * 90)
for method, data in all_results.items():
    if method == "Pre-trained":
        m = data
        print(f"{method:<15} {m['mse']:>12.6f} {m['ssim']:>8.4f} {m['iou']:>8.4f} "
              f"{m['mae']:>8.4f} {m.get('psnr',0):>8.2f} {m.get('vfae',0):>8.4f}")
    else:
        m = data["mean"]
        s = data["std"]
        print(f"{method:<15} {m['mse']:>8.6f}±{s['mse']:.4f} {m['ssim']:>8.4f} "
              f"{m['iou']:>8.4f} {m['mae']:>8.4f} {m.get('psnr',0):>8.2f}")
print("=" * 90)

# Improvement vs pretrained
print(f"\n{'Method':<15} {'MSE↓%':>10} {'SSIM↑%':>10} {'IOU↑%':>10}")
print("-" * 50)
pt_mse = pt["mse"]
pt_ssim = pt["ssim"]
pt_iou = pt["iou"]
for method, data in all_results.items():
    if method == "Pre-trained":
        continue
    m = data["mean"]
    mse_drop = (pt_mse - m["mse"]) / pt_mse * 100
    ssim_gain = (m["ssim"] - pt_ssim) / (pt_ssim + 1e-8) * 100
    iou_gain = (m["iou"] - pt_iou) / (pt_iou + 1e-8) * 100
    print(f"{method:<15} {mse_drop:>9.1f}% {ssim_gain:>9.1f}% {iou_gain:>9.1f}%")
print("=" * 50)

with open("results/final_eval.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nSaved to results/final_eval.json")
