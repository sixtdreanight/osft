"""E1 on real data WITH real-pretrained generator. Quick 1-seed comparison."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.optim as optim, numpy as np

from main.model.topologygan import TopologyGANGenerator
from main.osft.config import OSFTConfig
from main.osft.trainer import OSFTTrainer
from main.baselines.full_finetune import FullFinetuneTrainer
from main.baselines.adapter import apply_adapter_to_generator, count_adapter_params
from main.eval.metrics import evaluate_model
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda")
REAL_DATA = "data/cantilever_train.npy"
REAL_PRETRAINED = "checkpoints/real_pretrain/pretrained_generator.pt"
RESULTS = "results/real_pt"
os.makedirs(RESULTS, exist_ok=True)

def load_gen(path=REAL_PRETRAINED):
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
    n_epochs = 50
    torch.manual_seed(42); np.random.seed(42)

    all_results = {}

    # Pre-trained
    gen_pt = load_gen()
    pt_m = evaluate_model(gen_pt, test_loader, device)
    pt_m["trainable_pct"] = 0.0
    all_results["Pre-trained (real)"] = pt_m
    print(f"Pre-trained (real): MSE={pt_m['mse']:.4f}, SSIM={pt_m['ssim']:.4f}, IOU={pt_m['iou']:.4f}")
    cleanup(gen_pt)

    # Full FT
    gen_ft = load_gen()
    cfg_ft = OSFTConfig(checkpoint_dir=f"{RESULTS}/full_ft", save_every=999)
    ft = FullFinetuneTrainer(cfg_ft, generator=gen_ft)
    ft.train(train_loader, val_loader, n_epochs=n_epochs)
    ft_m = evaluate_model(ft.generator, test_loader, device)
    ft_m["trainable_pct"] = 100.0
    all_results["Full FT"] = ft_m
    print(f"Full FT:           MSE={ft_m['mse']:.4f}, SSIM={ft_m['ssim']:.4f}, IOU={ft_m['iou']:.4f}")
    cleanup(gen_ft, ft)

    # Adapter
    gen_ad = load_gen()
    apply_adapter_to_generator(gen_ad, hidden_dim=32)
    cfg_ad = OSFTConfig(checkpoint_dir=f"{RESULTS}/adapter", save_every=999)
    ad = FullFinetuneTrainer(cfg_ad, generator=gen_ad)
    ad.g_optimizer = optim.Adam([p for p in gen_ad.parameters() if p.requires_grad],
                                 lr=cfg_ad.lr, betas=(cfg_ad.beta1, cfg_ad.beta2))
    ad.train(train_loader, val_loader, n_epochs=n_epochs)
    ad_m = evaluate_model(gen_ad, test_loader, device)
    ad_m["trainable_pct"] = count_adapter_params(gen_ad)["trainable_pct"]
    all_results["Adapter"] = ad_m
    print(f"Adapter:           MSE={ad_m['mse']:.4f}, SSIM={ad_m['ssim']:.4f}, IOU={ad_m['iou']:.4f}")
    cleanup(gen_ad, ad)

    # OSFT
    gen_os = load_gen()
    cfg_os = OSFTConfig(checkpoint_dir=f"{RESULTS}/osft", save_every=999)
    os_trainer = OSFTTrainer(cfg_os, generator=gen_os)
    os_trainer.apply_svd_decomposition()
    os_trainer.train(train_loader, val_loader, n_epochs=n_epochs)
    os_m = evaluate_model(os_trainer.generator, test_loader, device)
    tp = sum(p.numel() for p in gen_os.parameters() if p.requires_grad)
    total = sum(p.numel() for p in gen_os.parameters())
    os_m["trainable_pct"] = 100 * tp / total
    all_results["OSFT"] = os_m
    print(f"OSFT:              MSE={os_m['mse']:.4f}, SSIM={os_m['ssim']:.4f}, IOU={os_m['iou']:.4f}")
    cleanup(gen_os, os_trainer)

    # Print comparison
    print(f"\n{'='*80}")
    print(f"Real Data with REAL Pretrained Generator")
    print(f"{'='*80}")
    print(f"{'Method':<22} {'MSE':>10} {'SSIM':>8} {'IOU':>8} {'PSNR':>8}")
    print(f"{'-'*60}")
    pt_mse = all_results["Pre-trained (real)"]["mse"]
    for name, m in all_results.items():
        delta = (pt_mse - m["mse"]) / pt_mse * 100 if name != "Pre-trained (real)" else 0
        d_str = f"({delta:+.1f}%)" if delta != 0 else ""
        print(f"{name:<22} {m['mse']:>10.4f} {m['ssim']:>8.4f} {m['iou']:>8.4f} {m.get('psnr',0):>8.2f}  {d_str}")

    # Vs synthetic pretrained comparison
    print(f"\n{'='*80}")
    print(f"Synth Pretrained  vs  Real Pretrained  (ΔMSE from respective baseline)")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Synth-PT ΔMSE':>16} {'Real-PT ΔMSE':>16}")
    print(f"{'-'*50}")
    # results from earlier run (synth pretrained, mean of 3 seeds)
    synth = {"Full FT": 20.8, "Adapter": 25.2, "OSFT": 24.8}
    for method in ["Full FT", "Adapter", "OSFT"]:
        m = all_results[method]
        real_delta = (pt_mse - m["mse"]) / pt_mse * 100
        print(f"{method:<15} {synth[method]:>15.1f}% {real_delta:>15.1f}%")

    with open(f"{RESULTS}/e1_real_pretrained.json", "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS}/e1_real_pretrained.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total: {(time.time()-t0)/60:.1f} min")
