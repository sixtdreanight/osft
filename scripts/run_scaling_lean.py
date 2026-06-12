"""E13: Pretraining Scaling Law — disk-lean version.

Only saves pretrain checkpoints (deleted after eval). Micro-training (OSFT/FT)
never saves .pt files — evaluates in-memory, records metrics to JSON only.

Usage:
  python scripts/run_scaling_lean.py --data synthetic --tag E13-S
  python scripts/run_scaling_lean.py --data cantilever --tag E13-R --seeds 3
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.optim as optim, numpy as np

from main.model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator
from main.osft.config import OSFTConfig
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINTS = [5, 10, 30, 50, 100, 200, 300]
N_FT_EPOCHS = 50


def get_data_path(name):
    return {"synthetic": "data/synthetic_train.npy",
            "cantilever": "data/cantilever_train.npy"}[name]


def load_gen(path):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    return gen


def pretrain_and_save(data_path, ckpt_dir, max_epochs=300):
    """Pretrain from scratch, save checkpoints at intervals."""
    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader, _, _ = create_dataloaders(
        data_path, height=64, width=128, batch_size=16, num_workers=0)

    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    disc = TopologyGANDiscriminator(condition_dim=6, output_c_dim=1, df_dim=16,
                                     height=64, width=128).to(device)
    cfg = OSFTConfig(n_epochs=max_epochs, checkpoint_dir=ckpt_dir,
                     lr=1e-3, eval_every=50, save_every=999, device="cuda")
    trainer = FullFinetuneTrainer(cfg, generator=gen, discriminator=disc)

    for epoch in range(max_epochs):
        t0 = time.time()
        m = trainer.train_epoch(train_loader, epoch)
        if (epoch + 1) in CHECKPOINTS:
            path = os.path.join(ckpt_dir, f"pretrain_ep{epoch+1:04d}.pt")
            torch.save(gen.state_dict(), path)
        if (epoch + 1) % 50 == 0:
            print(f"  Pretrain {epoch+1}/{max_epochs} G={m['G_loss']:.1f}")

    return ckpt_dir


def train_osft_in_memory(ckpt_path, data_path, n_epochs=50):
    """Train OSFT. Track best val_mse in memory, evaluate on best model."""
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, height=64, width=128, batch_size=16, num_workers=0)
    gen = load_gen(ckpt_path)
    cfg = OSFTConfig(n_epochs=n_epochs, lr=1e-4, save_every=999, eval_every=999)
    trainer = OSFTTrainer(cfg, generator=gen)
    trainer.apply_svd_decomposition()
    trainer.g_optimizer = optim.Adam(
        [p for p in gen.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    best_mse = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        trainer.train_epoch(train_loader, epoch)
        if (epoch + 1) % 5 == 0:
            val_m = trainer.evaluate(val_loader)
            if val_m["val_mse"] < best_mse:
                best_mse = val_m["val_mse"]
                best_state = {k: v.cpu().clone() for k, v in gen.state_dict().items()}
    if best_state:
        gen.load_state_dict(best_state)
    m = evaluate_model(gen, test_loader, device)
    del gen, trainer
    torch.cuda.empty_cache()
    return m


def train_fullft_in_memory(ckpt_path, data_path, n_epochs=50):
    """Train Full FT. Track best val_mse in memory, evaluate on best model."""
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path, height=64, width=128, batch_size=16, num_workers=0)
    gen = load_gen(ckpt_path)
    cfg = OSFTConfig(n_epochs=n_epochs, lr=1e-4, save_every=999, eval_every=999)
    ft = FullFinetuneTrainer(cfg, generator=gen)
    best_mse = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        ft.train_epoch(train_loader, epoch)
        if (epoch + 1) % 5 == 0:
            val_m = ft.evaluate(val_loader)
            if val_m["val_mse"] < best_mse:
                best_mse = val_m["val_mse"]
                best_state = {k: v.cpu().clone() for k, v in ft.generator.state_dict().items()}
    if best_state:
        ft.generator.load_state_dict(best_state)
    m = evaluate_model(ft.generator, test_loader, device)
    del gen, ft
    torch.cuda.empty_cache()
    return m


def evaluate_pretrained(ckpt_path, data_path):
    _, _, test_loader = create_dataloaders(
        data_path, height=64, width=128, batch_size=16, num_workers=0)
    gen = load_gen(ckpt_path)
    gen.eval()
    m = evaluate_model(gen, test_loader, device)
    del gen
    torch.cuda.empty_cache()
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, choices=["synthetic", "cantilever"])
    parser.add_argument("--tag", default="E13")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    data_path = get_data_path(args.data)
    ckpt_dir = args.ckpt_dir or f"checkpoints/scaling/{args.tag}"
    results_dir = args.results_dir or f"results/scaling/{args.tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"E13 Scaling Law — {args.tag} ({args.data}, {args.seeds} seeds)")
    print(f"{'='*60}")

    # Step 1: Pretrain
    print(f"\n--- Pretraining 300ep ---")
    t0 = time.time()
    pretrain_and_save(data_path, ckpt_dir)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    # Step 2: Evaluate each checkpoint (OSFT + FT, no .pt saving)
    summary = {}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        tag_s = f"_S{seed}" if args.seeds > 1 else ""

        for ep in CHECKPOINTS:
            ckpt_path = os.path.join(ckpt_dir, f"pretrain_ep{ep:04d}.pt")
            if not os.path.exists(ckpt_path):
                continue

            # Pretrained evaluation
            pt_key = f"ep{ep:04d}"
            if pt_key not in summary:
                pt_m = evaluate_pretrained(ckpt_path, data_path)
                summary[pt_key] = {"epoch": ep, "pretrained": pt_m}
                print(f"  PT ep{ep:04d}: MSE={pt_m['mse']:.4f}")

            # OSFT
            t1 = time.time()
            os_m = train_osft_in_memory(ckpt_path, data_path)
            elapsed = time.time() - t1
            if "osft" not in summary[pt_key]:
                summary[pt_key]["osft"] = []
            summary[pt_key]["osft"].append(os_m)
            print(f"  OSFT ep{ep:04d}{tag_s}: MSE={os_m['mse']:.4f} ({elapsed:.0f}s)")

            # Full FT
            t1 = time.time()
            ft_m = train_fullft_in_memory(ckpt_path, data_path)
            elapsed = time.time() - t1
            if "full_ft" not in summary[pt_key]:
                summary[pt_key]["full_ft"] = []
            summary[pt_key]["full_ft"].append(ft_m)
            print(f"  FT ep{ep:04d}{tag_s}: MSE={ft_m['mse']:.4f} ({elapsed:.0f}s)")

    # Aggregate and save
    output = []
    for ep in CHECKPOINTS:
        key = f"ep{ep:04d}"
        if key not in summary:
            continue
        s = summary[key]
        pt_mse = s["pretrained"]["mse"]
        os_avg = float(np.mean([m["mse"] for m in s.get("osft", [])]))
        ft_avg = float(np.mean([m["mse"] for m in s.get("full_ft", [])]))
        os_ssim = float(np.mean([m["ssim"] for m in s.get("osft", [])]))
        ft_ssim = float(np.mean([m["ssim"] for m in s.get("full_ft", [])]))
        os_iou = float(np.mean([m["iou"] for m in s.get("osft", [])]))
        ft_iou = float(np.mean([m["iou"] for m in s.get("full_ft", [])]))

        output.append({
            "epoch": ep,
            "pt_mse": float(pt_mse),
            "osft_mse": os_avg,
            "ft_mse": ft_avg,
            "osft_gain": (pt_mse - os_avg) / pt_mse * 100,
            "ft_gain": (pt_mse - ft_avg) / pt_mse * 100,
            "osft_ssim": os_ssim,
            "ft_ssim": ft_ssim,
            "osft_iou": os_iou,
            "ft_iou": ft_iou,
        })

    # Print table
    print(f"\n{'='*70}")
    print(f"Scaling Law: {args.tag}")
    print(f"{'='*70}")
    print(f"{'Ep':>5} {'PT':>8} {'OSFT':>8} {'FT':>8} {'OSFT%':>8} {'FT%':>8}")
    print("-" * 45)
    crossover = None
    for row in output:
        print(f"{row['epoch']:>5} {row['pt_mse']:>8.4f} {row['osft_mse']:>8.4f} "
              f"{row['ft_mse']:>8.4f} {row['osft_gain']:>+7.1f}% {row['ft_gain']:>+7.1f}%")
        if row["osft_gain"] > row["ft_gain"] and crossover is None:
            crossover = row["epoch"]

    if crossover:
        print(f"\nK_c = {crossover}ep (OSFT surpasses Full FT)")

    json_path = f"{results_dir}/scaling_summary.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Cleanup pretrain checkpoints
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"Saved to {json_path} ({len(output)} points, K_c={crossover})")
    print(f"Disk: pretrain checkpoints cleaned")


if __name__ == "__main__":
    main()
