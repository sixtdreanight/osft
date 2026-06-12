"""E13: Pretraining Scaling Law (one pretrain → checkpoint at intervals → OSFT+FT each).

Usage:
  python scripts/run_scaling.py --data synthetic --tag E13-S
  python scripts/run_scaling.py --data cantilever --tag E13-R --seeds 3
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
    if "generator_state_dict" in state: state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    return gen


def pretrain_and_save(data_path, ckpt_dir, max_epochs=300):
    """Pretrain from scratch, save checkpoint at specified intervals."""
    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader, val_loader, _ = create_dataloaders(
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
        train_metrics = trainer.train_epoch(train_loader, epoch)
        elapsed = time.time() - t0

        if (epoch + 1) in CHECKPOINTS:
            path = os.path.join(ckpt_dir, f"pretrain_ep{epoch+1:04d}.pt")
            torch.save(gen.state_dict(), path)
            print(f"  [Checkpoint] epoch {epoch+1} saved ({elapsed:.1f}s)")

        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"  Pretrain epoch {epoch+1}/{max_epochs} "
                  f"G={train_metrics['G_loss']:.1f} ({elapsed:.1f}s)")

    return ckpt_dir


def run_osft(ckpt_path, results_dir, seed=0):
    """Fine-tune with OSFT from a pretrain checkpoint."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        get_data_path(args.data), height=64, width=128, batch_size=16, num_workers=0)

    gen = load_gen(ckpt_path)
    tag = os.path.basename(ckpt_path).replace(".pt", "")
    cfg = OSFTConfig(n_epochs=N_FT_EPOCHS, checkpoint_dir=f"{results_dir}/osft_{tag}",
                     lr=1e-4, eval_every=5, save_every=999)
    trainer = OSFTTrainer(cfg, generator=gen)
    trainer.apply_svd_decomposition()
    trainer.g_optimizer = optim.Adam(
        [p for p in gen.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    trainer.train(train_loader, val_loader, n_epochs=N_FT_EPOCHS)

    m = evaluate_model(gen, test_loader, device)
    tp = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    total = sum(p.numel() for p in gen.parameters())
    m["trainable_pct"] = 100 * tp / total
    del gen, trainer; torch.cuda.empty_cache()
    return m


def run_fullft(ckpt_path, results_dir, seed=0):
    """Fine-tune with Full FT from a pretrain checkpoint."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        get_data_path(args.data), height=64, width=128, batch_size=16, num_workers=0)

    gen = load_gen(ckpt_path)
    tag = os.path.basename(ckpt_path).replace(".pt", "")
    cfg = OSFTConfig(n_epochs=N_FT_EPOCHS, checkpoint_dir=f"{results_dir}/ft_{tag}",
                     lr=1e-4, eval_every=5, save_every=999)
    ft = FullFinetuneTrainer(cfg, generator=gen)
    ft.train(train_loader, val_loader, n_epochs=N_FT_EPOCHS)

    m = evaluate_model(ft.generator, test_loader, device)
    m["trainable_pct"] = 100.0
    del gen, ft; torch.cuda.empty_cache()
    return m


def evaluate_pretrained(ckpt_path):
    """Evaluate pretrained model without fine-tuning."""
    _, _, test_loader = create_dataloaders(
        get_data_path(args.data), height=64, width=128, batch_size=16, num_workers=0)
    gen = load_gen(ckpt_path)
    gen.eval()
    m = evaluate_model(gen, test_loader, device)
    del gen; torch.cuda.empty_cache()
    return m


def main():
    global args
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
    print(f"E13: Pretraining Scaling Law — {args.tag} ({args.data})")
    print(f"{'='*60}")

    # Step 1: Pretrain once, save checkpoints
    print(f"\n--- Step 1: Pretraining 300ep ({args.data}) ---")
    t0 = time.time()
    pretrain_and_save(data_path, ckpt_dir)
    print(f"Pretraining done in {(time.time()-t0)/60:.1f} min")

    # Step 2: For each checkpoint, evaluate pretrained, run OSFT, run Full FT
    all_results = {}
    for seed in range(args.seeds):
        if args.seeds > 1:
            print(f"\n--- Seed {seed+1}/{args.seeds} ---")
        for ep in CHECKPOINTS:
            ckpt_path = os.path.join(ckpt_dir, f"pretrain_ep{ep:04d}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {ckpt_path} not found")
                continue

            tag = f"ep{ep:04d}"
            if args.seeds > 1: tag += f"_S{seed}"

            # Pretrained
            pt_key = f"Pre-trained/{tag}"
            if pt_key not in all_results:
                m_pt = evaluate_pretrained(ckpt_path)
                all_results[pt_key] = m_pt
                print(f"  {pt_key}: MSE={m_pt['mse']:.4f}")

            # OSFT
            m_os = run_osft(ckpt_path, results_dir, seed=seed)
            all_results[f"OSFT/{tag}"] = m_os
            print(f"  OSFT/{tag}: MSE={m_os['mse']:.4f}")

            # Full FT
            m_ft = run_fullft(ckpt_path, results_dir, seed=seed)
            all_results[f"Full FT/{tag}"] = m_ft
            print(f"  Full FT/{tag}: MSE={m_ft['mse']:.4f}")

    # Step 3: Aggregate and save
    # Group by epoch
    import json
    summary = {}
    for ep in CHECKPOINTS:
        tag = f"ep{ep:04d}"
        pt_keys = [k for k in all_results if k.startswith(f"Pre-trained/{tag}")]
        os_keys = [k for k in all_results if k.startswith(f"OSFT/{tag}")]
        ft_keys = [k for k in all_results if k.startswith(f"Full FT/{tag}")]

        summary[ep] = {
            "epoch": ep,
            "pretrained": {k: float(np.mean([all_results[kk][k] for kk in pt_keys]))
                          for k in ["mse","ssim","iou"]} if pt_keys else {},
            "osft": {k: float(np.mean([all_results[kk][k] for kk in os_keys]))
                    for k in ["mse","ssim","iou"]} if os_keys else {},
            "full_ft": {k: float(np.mean([all_results[kk][k] for kk in ft_keys]))
                       for k in ["mse","ssim","iou"]} if ft_keys else {},
        }

        pt_mse = summary[ep]["pretrained"].get("mse", 1.0)
        os_mse = summary[ep]["osft"].get("mse", pt_mse)
        ft_mse = summary[ep]["full_ft"].get("mse", pt_mse)
        summary[ep]["osft_gain"] = (pt_mse - os_mse) / pt_mse * 100
        summary[ep]["fullft_gain"] = (pt_mse - ft_mse) / pt_mse * 100

    # Print table
    print(f"\n{'='*70}")
    print(f"E13 Scaling Law: {args.tag}")
    print(f"{'='*70}")
    print(f"{'Ep':>5} {'PT MSE':>8} {'OSFT MSE':>9} {'FT MSE':>9} {'OSFT Gain':>10} {'FT Gain':>10}")
    print(f"{'-'*55}")
    for ep in CHECKPOINTS:
        s = summary[ep]
        print(f"{ep:>5} {s['pretrained'].get('mse',0):>8.4f} {s['osft'].get('mse',0):>9.4f} "
              f"{s['full_ft'].get('mse',0):>9.4f} {s['osft_gain']:>9.1f}% {s['fullft_gain']:>9.1f}%")

    # Find crossover
    crossover = None
    for ep in sorted(CHECKPOINTS):
        if summary[ep]['osft_gain'] > summary[ep]['fullft_gain']:
            crossover = ep
            break
    if crossover:
        print(f"\nK_c = {crossover}ep (OSFT surpasses Full FT)")

    with open(f"{results_dir}/scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {results_dir}/scaling_summary.json")


if __name__ == "__main__":
    main()
