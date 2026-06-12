"""E15: Multi-structure generalization experiment.

Cross-domain evaluation: pretrain on structure A, fine-tune on structure B.
Measures whether OSFT preserves transferable knowledge better than Full FT.

Usage:
  python scripts/run_e15_generalization.py --source cantilever --target mbb_beam
  python scripts/run_e15_generalization.py --all  # full matrix

Output:
  results/e15/generalization_matrix.json
  results/e15/generalization_matrix.csv
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

# All available datasets
DATASETS = {
    "cantilever": "data/cantilever_train.npy",
    "mbb_beam":   "data/mbb_beam_train.npy",
    "l_beam":     "data/l-beam_train.npy",
    "bridge":     "data/bridge_train.npy",
}

PRETRAIN_EPOCHS = 100      # pretraining from scratch on source domain
FT_EPOCHS = 50             # fine-tuning on target domain
BATCH_SIZE = 16
RESULTS_DIR = "results/e15"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────
def load_gen(path):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    return gen


def pretrain_on(data_path, ckpt_dir, n_epochs=PRETRAIN_EPOCHS):
    """Pretrain GAN from scratch, return checkpoint path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader, _, _ = create_dataloaders(
        data_path, height=64, width=128, batch_size=BATCH_SIZE, num_workers=0)

    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    disc = TopologyGANDiscriminator(condition_dim=6, output_c_dim=1, df_dim=16,
                                     height=64, width=128).to(device)
    cfg = OSFTConfig(n_epochs=n_epochs, checkpoint_dir=ckpt_dir, lr=1e-3,
                     eval_every=999, save_every=999, device="cuda")
    trainer = FullFinetuneTrainer(cfg, generator=gen, discriminator=disc)

    for epoch in range(n_epochs):
        m = trainer.train_epoch(train_loader, epoch)
        if (epoch + 1) % 20 == 0:
            pass  # silent

    ckpt_path = os.path.join(ckpt_dir, "pretrained.pt")
    torch.save(gen.state_dict(), ckpt_path)
    del gen, disc, trainer
    torch.cuda.empty_cache()
    return ckpt_path


def finetune_osft(ckpt_path, target_data, n_epochs=FT_EPOCHS):
    """OSFT fine-tune, evaluate on best val_mse."""
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, height=64, width=128, batch_size=BATCH_SIZE, num_workers=0)
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
            vm = trainer.evaluate(val_loader)
            if vm["val_mse"] < best_mse:
                best_mse = vm["val_mse"]
                best_state = {k: v.cpu().clone() for k, v in gen.state_dict().items()}

    if best_state:
        gen.load_state_dict(best_state)
    m = evaluate_model(gen, test_loader, device)
    del gen, trainer
    torch.cuda.empty_cache()
    return m


def finetune_fullft(ckpt_path, target_data, n_epochs=FT_EPOCHS):
    """Full FT fine-tune, evaluate on best val_mse."""
    train_loader, val_loader, test_loader = create_dataloaders(
        target_data, height=64, width=128, batch_size=BATCH_SIZE, num_workers=0)
    gen = load_gen(ckpt_path)

    cfg = OSFTConfig(n_epochs=n_epochs, lr=1e-4, save_every=999, eval_every=999)
    trainer = FullFinetuneTrainer(cfg, generator=gen)

    best_mse = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        trainer.train_epoch(train_loader, epoch)
        if (epoch + 1) % 5 == 0:
            vm = trainer.evaluate(val_loader)
            if vm["val_mse"] < best_mse:
                best_mse = vm["val_mse"]
                best_state = {k: v.cpu().clone() for k, v in trainer.generator.state_dict().items()}

    if best_state:
        trainer.generator.load_state_dict(best_state)
    m = evaluate_model(trainer.generator, test_loader, device)
    del gen, trainer
    torch.cuda.empty_cache()
    return m


def evaluate_pretrained(ckpt_path, target_data):
    """Direct evaluation (no fine-tuning)."""
    _, _, test_loader = create_dataloaders(
        target_data, height=64, width=128, batch_size=BATCH_SIZE, num_workers=0)
    gen = load_gen(ckpt_path)
    gen.eval()
    m = evaluate_model(gen, test_loader, device)
    del gen
    torch.cuda.empty_cache()
    return m


# ── single cell ───────────────────────────────────────────────
def run_pair(source, target):
    """Pretrain on source, test on target (OSFT + Full FT + direct)."""
    src_data = DATASETS[source]
    tgt_data = DATASETS[target]
    if not os.path.exists(src_data) or not os.path.exists(tgt_data):
        return None

    ckpt_dir = os.path.join(RESULTS_DIR, f"pretrain_{source}_to_{target}")
    ckpt_path = os.path.join(ckpt_dir, "pretrained.pt")

    # Pretrain (reuse if exists)
    if not os.path.exists(ckpt_path):
        print(f"  Pretraining on {source}...")
        pretrain_on(src_data, ckpt_dir)
    else:
        print(f"  Using cached pretrain: {ckpt_path}")

    # Direct evaluation
    pt = evaluate_pretrained(ckpt_path, tgt_data)

    # OSFT
    print(f"  OSFT fine-tuning...")
    osft = finetune_osft(ckpt_path, tgt_data)

    # Full FT
    print(f"  Full FT fine-tuning...")
    ft = finetune_fullft(ckpt_path, tgt_data)

    return {
        "source": source,
        "target": target,
        "pretrained": {"mse": pt["mse"], "ssim": pt["ssim"], "iou": pt["iou"]},
        "osft": {"mse": osft["mse"], "ssim": osft["ssim"], "iou": osft["iou"]},
        "full_ft": {"mse": ft["mse"], "ssim": ft["ssim"], "iou": ft["iou"]},
        "osft_gain": (pt["mse"] - osft["mse"]) / pt["mse"] * 100,
        "ft_gain": (pt["mse"] - ft["mse"]) / pt["mse"] * 100,
    }


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="single source domain")
    parser.add_argument("--target", default=None, help="single target domain")
    parser.add_argument("--all", action="store_true", help="full cross matrix")
    parser.add_argument("--pairs", default=None, help="comma-separated pairs: src1:tgt1,src2:tgt2")
    args = parser.parse_args()

    available = [k for k, v in DATASETS.items() if os.path.exists(v)]
    print(f"Available datasets: {available}")

    if args.source and args.target:
        pairs = [(args.source, args.target)]
    elif args.pairs:
        pairs = [tuple(p.strip().split(":")) for p in args.pairs.split(",")]
    elif args.all:
        available_4 = available[:4]  # use at most 4
        pairs = [(s, t) for s in available_4 for t in available_4]  # full matrix
    else:
        print("Specify --source/--target, --pairs, or --all")
        return

    results = []
    t0 = time.time()
    for i, (src, tgt) in enumerate(pairs):
        print(f"\n{'=' * 55}")
        print(f"[{i+1}/{len(pairs)}] {src} → {tgt}")
        print(f"{'=' * 55}")
        t1 = time.time()
        cell = run_pair(src, tgt)
        if cell:
            results.append(cell)
            print(f"  OSFT: MSE={cell['osft']['mse']:.4f} "
                  f"({cell['osft_gain']:+.1f}%), "
                  f"IOU={cell['osft']['iou']:.4f}")
            print(f"  FT:   MSE={cell['full_ft']['mse']:.4f} "
                  f"({cell['ft_gain']:+.1f}%), "
                  f"IOU={cell['full_ft']['iou']:.4f}")
            print(f"  time: {(time.time() - t1) / 60:.1f} min")

    # Save
    results_path = os.path.join(RESULTS_DIR, "generalization_matrix.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    if len(results) >= 3:
        print(f"\n{'=' * 70}")
        print("Generalization Matrix (OSFT MSE / Full FT MSE)")
        print(f"{'=' * 70}")
        sources = sorted(set(r["source"] for r in results))
        targets = sorted(set(r["target"] for r in results))
        header = f"{'':>12}" + "".join(f"{t:>14}" for t in targets)
        print(header)
        print("-" * len(header))
        for s in sources:
            row = f"{s:>12}"
            for t in targets:
                r = [r for r in results if r["source"] == s and r["target"] == t]
                if r:
                    r = r[0]
                    row += f" {r['osft']['mse']:.3f}/{r['full_ft']['mse']:.3f}"
                else:
                    row += f" {'—':>13}"
            print(row)

        # Generalization Drop
        print(f"\n{'=' * 70}")
        print("Generalization Drop (IOU)")
        print(f"{'=' * 70}")
        print(f"{'Source→Target':>20} {'OSFT IOU':>10} {'FT IOU':>10} {'OSFT better?':>14}")
        print("-" * 55)
        for r in results:
            better = "OSFT ✓" if r["osft"]["iou"] > r["full_ft"]["iou"] else "FT"
            print(f"{r['source']+'→'+r['target']:>20} {r['osft']['iou']:>10.4f} "
                  f"{r['full_ft']['iou']:>10.4f} {better:>14}")

    total_min = (time.time() - t0) / 60
    print(f"\nSaved to {results_path}  ({total_min:.1f} min total)")


if __name__ == "__main__":
    main()
