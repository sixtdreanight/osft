"""E1 on real Cantilever data. Compares Pre-trained, Full FT, Adapter, OSFT."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.optim as optim, numpy as np

from main.model.topologygan import TopologyGANGenerator
from main.osft.config import OSFTConfig
from main.osft.decomposer import SVDWeightDecomposer
from main.osft.subspace_layers import apply_osft_to_generator
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.baselines.lora import apply_lora_to_generator, count_lora_params
from main.baselines.adapter import apply_adapter_to_generator, count_adapter_params
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_DATA = "data/cantilever_train.npy"
PRETRAINED = "checkpoints/quickstart/pretrained_generator.pt"
RESULTS = "results/real"
os.makedirs(RESULTS, exist_ok=True)

def load_gen(path=PRETRAINED):
    state = torch.load(path, map_location=device, weights_only=False)
    if "generator_state_dict" in state: state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=64, width=128).to(device)
    gen.load_state_dict(state, strict=False)
    return gen

def cleanup(*models):
    for m in models:
        if m is not None: del m
    torch.cuda.empty_cache()

def main():
    train_loader, val_loader, test_loader = create_dataloaders(
        REAL_DATA, height=64, width=128, batch_size=16, num_workers=0)
    n_epochs, n_seeds = 50, 3
    all_results = {}

    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n{'='*50}\nSeed {seed+1}/{n_seeds}\n{'='*50}")

        # 1. Pre-trained (eval only)
        gen_pt = load_gen()
        pt_m = evaluate_model(gen_pt, test_loader, device)
        pt_m["trainable_pct"] = 0.0
        all_results[f"Pre-trained/S{seed}"] = pt_m
        print(f"  Pre-trained: MSE={pt_m['mse']:.4f}, SSIM={pt_m['ssim']:.4f}")
        cleanup(gen_pt)

        # 2. Full FT
        gen_ft = load_gen()
        cfg_ft = OSFTConfig(checkpoint_dir=f"{RESULTS}/full_ft_S{seed}", save_every=999)
        ft = FullFinetuneTrainer(cfg_ft, generator=gen_ft)
        ft.train(train_loader, val_loader, n_epochs=n_epochs)
        ft_m = evaluate_model(ft.generator, test_loader, device)
        ft_m["trainable_pct"] = 100.0
        all_results[f"Full FT/S{seed}"] = ft_m
        print(f"  Full FT:    MSE={ft_m['mse']:.4f}, SSIM={ft_m['ssim']:.4f}")
        cleanup(gen_ft, ft)

        # 3. Adapter
        gen_ad = load_gen()
        apply_adapter_to_generator(gen_ad, hidden_dim=32)
        cfg_ad = OSFTConfig(checkpoint_dir=f"{RESULTS}/adapter_S{seed}", save_every=999)
        ad = FullFinetuneTrainer(cfg_ad, generator=gen_ad)
        ad.g_optimizer = optim.Adam([p for p in gen_ad.parameters() if p.requires_grad],
                                     lr=cfg_ad.lr, betas=(cfg_ad.beta1, cfg_ad.beta2))
        ad.train(train_loader, val_loader, n_epochs=n_epochs)
        ad_m = evaluate_model(gen_ad, test_loader, device)
        ad_m["trainable_pct"] = count_adapter_params(gen_ad)["trainable_pct"]
        all_results[f"Adapter/S{seed}"] = ad_m
        print(f"  Adapter:    MSE={ad_m['mse']:.4f}, SSIM={ad_m['ssim']:.4f}")
        cleanup(gen_ad, ad)

        # 4. OSFT
        gen_os = load_gen()
        cfg_os = OSFTConfig(checkpoint_dir=f"{RESULTS}/osft_S{seed}", save_every=999)
        os_trainer = OSFTTrainer(cfg_os, generator=gen_os)
        os_trainer.apply_svd_decomposition()
        os_trainer.train(train_loader, val_loader, n_epochs=n_epochs)
        os_m = evaluate_model(os_trainer.generator, test_loader, device)
        tp = sum(p.numel() for p in gen_os.parameters() if p.requires_grad)
        total = sum(p.numel() for p in gen_os.parameters())
        os_m["trainable_pct"] = 100 * tp / total
        all_results[f"OSFT/S{seed}"] = os_m
        print(f"  OSFT:       MSE={os_m['mse']:.4f}, SSIM={os_m['ssim']:.4f}")
        cleanup(gen_os, os_trainer)

    # Aggregate
    from collections import defaultdict
    methods = defaultdict(list)
    for key, m in all_results.items():
        method = key.rsplit("/S", 1)[0]
        methods[method].append(m)

    print(f"\n{'='*70}")
    print(f"E1 on REAL Cantilever Data ({n_seeds} seeds × {n_epochs} epochs)")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'MSE':>12} {'SSIM':>8} {'IOU':>8} {'PSNR':>8} {'VFAE':>8} {'Train%':>8}")
    print(f"{'-'*70}")
    agg = {}
    for method in ["Pre-trained", "Full FT", "Adapter", "OSFT"]:
        ms = methods[method]
        agg[method] = {}
        for metric in ["mse", "ssim", "iou", "psnr", "vfae", "trainable_pct"]:
            vals = [m[metric] for m in ms if metric in m and isinstance(m[metric], (int, float))]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                agg[method][metric] = f"{mean_v:.4f}±{std_v:.4f}"
                if metric == "mse":
                    print(f"{method:<15} {mean_v:>8.4f}±{std_v:.4f}", end="")
        print()

    with open(f"{RESULTS}/e1_real_results.json", "w") as f:
        json.dump({k: dict(v) for k, v in agg.items()}, f, indent=2)

    # Improvement over pretrained
    pt_mse = float(agg["Pre-trained"]["mse"].split("±")[0])
    print(f"\n{'Method':<15} {'MSE↓':>10}")
    print("-"*30)
    for method in ["Full FT", "Adapter", "OSFT"]:
        mse = float(agg[method]["mse"].split("±")[0])
        delta = (pt_mse - mse) / pt_mse * 100
        print(f"{method:<15} {delta:>9.1f}%")
    print(f"\nSaved to {RESULTS}/e1_real_results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
