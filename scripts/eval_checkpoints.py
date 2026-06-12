"""Quick evaluation of existing checkpoints."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from main.model.topologygan import TopologyGANGenerator
from main.baselines.full_finetune import FullFinetuneTrainer
from main.eval.metrics import evaluate_model, compute_all_image_metrics
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
_, _, test_loader = create_dataloaders(
    "data/synthetic_train.npy", height=64, width=128, batch_size=16, num_workers=0
)

def load_gen(ckpt_path, variant="unet", gf_dim=64):
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
    return gen

def eval_ckpt(name, ckpt_path, variant="unet", gf_dim=64):
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {name}: file not found")
        return None
    gen = load_gen(ckpt_path, variant, gf_dim)
    metrics = evaluate_model(gen, test_loader, device)
    tp = sum(p.numel() for p in gen.parameters())
    tr = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    metrics["trainable_pct"] = 100.0 * tr / tp
    metrics["total_params"] = tp
    print(f"  {name}: MSE={metrics['mse']:.6f}  SSIM={metrics['ssim']:.4f}  "
          f"IOU={metrics['iou']:.4f}  MAE={metrics['mae']:.4f}  "
          f"Trainable={tr:,}/{tp:,} ({metrics['trainable_pct']:.1f}%)")
    del gen
    torch.cuda.empty_cache()
    return metrics

print("\n" + "=" * 60)
print("Checkpoint Evaluation")
print("=" * 60)

results = {}

# 1. Pre-trained
results["Pre-trained"] = eval_ckpt("Pre-trained",
    "checkpoints/quickstart/pretrained_generator.pt")

# 2. Quickstart Full FT (30 epochs)
results["Full FT (qs, 30ep)"] = eval_ckpt("Full FT (qs, 30ep)",
    "checkpoints/quickstart/pretrain/full_ft_best.pt")

# 3. Quickstart OSFT (20 epochs)
results["OSFT (qs, 20ep)"] = eval_ckpt("OSFT (qs, 20ep)",
    "checkpoints/quickstart/osft/osft_best.pt")

# 4. E1 Full FT - best from seed 0
results["Full FT (E1 S0, 50ep)"] = eval_ckpt("Full FT (E1 S0, 50ep)",
    "results/E1/adapter_Cantilever_S0/full_ft_best.pt")

# 5. Overnight Full FT - best
results["Full FT (overnight, 71/200ep)"] = eval_ckpt("Full FT (overnight)",
    "results/overnight/full_ft_Cantilever_S0/full_ft_best.pt")

# Summary table
print("\n" + "=" * 80)
print(f"{'Model':<30} {'MSE':>10} {'SSIM':>8} {'IOU':>8} {'MAE':>8} {'Train%':>8}")
print("-" * 80)
for name, m in results.items():
    if m:
        print(f"{name:<30} {m['mse']:>10.6f} {m['ssim']:>8.4f} {m['iou']:>8.4f} "
              f"{m['mae']:>8.4f} {m['trainable_pct']:>7.1f}%")
print("=" * 80)

# Save
import json
with open("results/eval_checkpoints.json", "w") as f:
    json.dump({k: v for k, v in results.items() if v}, f, indent=2)
print("\nSaved to results/eval_checkpoints.json")
