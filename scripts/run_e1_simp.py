#!/usr/bin/env python3
"""E1 comparison on SIMP-optimized Cantilever data.

Fine-tunes Full FT, OSFT, Adapter, LoRA on SIMP cantilever topologies,
then compares all metrics including FEM compliance validation.
"""

import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.optim as optim
import numpy as np

from main.model.topologygan import TopologyGANGenerator
from main.osft.config import OSFTConfig
from main.osft.decomposer import SVDWeightDecomposer
from main.osft.subspace_layers import apply_osft_to_generator
from main.eval.metrics import evaluate_model, compute_all_image_metrics
from main.eval.fem_validator import FEMValidator
from main.utils.data_loader import create_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 128
DATA = "data/cantilever_train.npy"
PRETRAINED = "checkpoints/quickstart/pretrained_generator.pt"
OUT = "results/simp_e1"
os.makedirs(OUT, exist_ok=True)
EPOCHS = 50
BATCH = 8
LR = 1e-4


def load_gen(ckpt=PRETRAINED):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if "generator_state_dict" in state:
        state = state["generator_state_dict"]
    gen = TopologyGANGenerator(input_c_dim=3, output_c_dim=1, gf_dim=64,
                                variant="unet", height=H, width=W).to(device)
    gen.load_state_dict(state, strict=False)
    return gen


def get_loaders():
    return create_dataloaders(DATA, height=H, width=W, batch_size=BATCH, num_workers=0)


def save_checkpoint(gen, name):
    torch.save({"generator_state_dict": gen.state_dict()}, f"{OUT}/{name}.pt")


def make_trainable_params(gen):
    return [p for p in gen.parameters() if p.requires_grad]


def train_loop(gen, train_loader, optimizer, criterion, name, epochs=EPOCHS):
    best_loss = float("inf")
    for epoch in range(epochs):
        gen.train()
        total_loss = 0.0
        for batch in train_loader:
            cond = batch[0].to(device)
            target = batch[1].to(device)
            optimizer.zero_grad()
            output = gen(cond)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(gen, f"{name}_best")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

    save_checkpoint(gen, f"{name}_latest")
    return gen


# ============================================================
# Full Fine-tuning
# ============================================================
def run_full_ft():
    print("\n" + "=" * 60)
    print("Full Fine-tuning on SIMP Cantilever")
    print("=" * 60)
    gen = load_gen()
    train_loader, _, _ = get_loaders()
    optimizer = optim.Adam(gen.parameters(), lr=LR)
    criterion = torch.nn.L1Loss()
    return train_loop(gen, train_loader, optimizer, criterion, "full_ft")


# ============================================================
# OSFT
# ============================================================
def run_osft():
    print("\n" + "=" * 60)
    print("OSFT on SIMP Cantilever")
    print("=" * 60)
    gen = load_gen()
    train_loader, _, _ = get_loaders()

    config = OSFTConfig(energy_threshold=0.80, lr=LR, n_epochs=EPOCHS)
    decomposer = SVDWeightDecomposer(energy_threshold=config.energy_threshold)
    apply_osft_to_generator(gen, decomposer, freeze_main=True)

    optimizer = optim.Adam(make_trainable_params(gen), lr=LR)
    criterion = torch.nn.L1Loss()
    return train_loop(gen, train_loader, optimizer, criterion, "osft")


# ============================================================
# Adapter
# ============================================================
def run_adapter():
    print("\n" + "=" * 60)
    print("Adapter on SIMP Cantilever")
    print("=" * 60)
    gen = load_gen()
    train_loader, _, _ = get_loaders()

    adapters = []
    for mod in gen.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            mod.requires_grad_(False)
            adapter = torch.nn.Sequential(
                torch.nn.Conv2d(mod.out_channels, max(mod.out_channels // 4, 4), 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(max(mod.out_channels // 4, 4), mod.out_channels, 1),
            ).to(device)
            adapters.append(adapter)
            mod._adapter = adapter

    trainable = [p for a in adapters for p in a.parameters()]
    optimizer = optim.Adam(trainable, lr=LR * 10)
    criterion = torch.nn.L1Loss()
    return train_loop(gen, train_loader, optimizer, criterion, "adapter")


# ============================================================
# LoRA
# ============================================================
def run_lora():
    print("\n" + "=" * 60)
    print("LoRA on SIMP Cantilever")
    print("=" * 60)
    gen = load_gen()
    train_loader, _, _ = get_loaders()

    lora_params = []
    for mod in gen.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            out_c, in_c = mod.out_channels, mod.in_channels
            r = 8
            mod.lora_A = torch.nn.Parameter(torch.randn(r, in_c, 1, 1) * 0.01).to(device)
            mod.lora_B = torch.nn.Parameter(torch.zeros(out_c, r, 1, 1)).to(device)
            lora_params.extend([mod.lora_A, mod.lora_B])
            mod.requires_grad_(False)

    optimizer = optim.Adam(lora_params, lr=LR * 10)
    criterion = torch.nn.L1Loss()

    # Override forward for LoRA
    original_forwards = {}
    for name, mod in gen.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            original_forwards[name] = mod.forward

            def make_lora_forward(m):
                orig = m.forward
                def lora_fwd(x):
                    out = orig(x)
                    delta = m.lora_B @ m.lora_A @ x.reshape(
                        x.size(0), -1, x.size(2), x.size(3))
                    return out + delta.reshape_as(out)
                return lora_fwd

            mod.forward = make_lora_forward(mod)

    return train_loop(gen, train_loader, optimizer, criterion, "lora")


# ============================================================
# Evaluation
# ============================================================
def evaluate_all():
    print("\n" + "=" * 60)
    print("Evaluation on SIMP Cantilever Test Set")
    print("=" * 60)

    _, _, test_loader = get_loaders()
    models = {}

    models["Pre-trained"] = load_gen(PRETRAINED)
    for key, display in [("full_ft", "Full FT"), ("osft", "OSFT"),
                          ("adapter", "Adapter"), ("lora", "LoRA-r8")]:
        ckpt = f"{OUT}/{key}_best.pt"
        if os.path.exists(ckpt):
            models[display] = load_gen(ckpt)

    all_results = {}
    for name, gen in models.items():
        gen.eval()
        metrics_sum = {}
        n = 0

        with torch.no_grad():
            for batch in test_loader:
                if n >= 50:
                    break
                cond = batch[0].to(device)
                target = batch[1]
                output = gen(cond).cpu()

                for i in range(output.size(0)):
                    m = compute_all_image_metrics(output[i:i+1], target[i:i+1])
                    for k, v in m.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    n += 1

        avg = {k: v / n for k, v in metrics_sum.items()}
        all_results[name] = avg

        print(f"\n{name}:")
        for k in ["mse", "ssim", "iou", "psnr", "lpips", "vfae"]:
            if k in avg:
                print(f"  {k}: {avg[k]:.4f}")

    with open(f"{OUT}/e1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def run_fem_validation():
    print("\n" + "=" * 60)
    print("FEM Validation on SIMP Cantilever")
    print("=" * 60)

    _, _, test_loader = get_loaders()
    fem = FEMValidator(height=H, width=W)

    models = {}
    models["Pre-trained"] = load_gen(PRETRAINED)
    for key, display in [("full_ft", "Full FT"), ("osft", "OSFT"),
                          ("adapter", "Adapter"), ("lora", "LoRA-r8")]:
        ckpt = f"{OUT}/{key}_best.pt"
        if os.path.exists(ckpt):
            models[display] = load_gen(ckpt)

    fem_results = {}
    for name, gen in models.items():
        t0 = time.time()
        metrics = fem.validate_dataset(gen, test_loader, device, max_samples=20)
        elapsed = time.time() - t0
        clean = {k: (float(v) if not np.isnan(v) else None) for k, v in metrics.items()}
        fem_results[name] = clean
        print(f"\n{name} ({elapsed:.0f}s):")
        for k, v in clean.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    with open(f"{OUT}/e11_fem_results.json", "w") as f:
        json.dump(fem_results, f, indent=2)
    return fem_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--fem", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.train:
        run_full_ft()
        run_osft()
        run_adapter()
        run_lora()

    if args.all or args.eval:
        evaluate_all()

    if args.all or args.fem:
        run_fem_validation()
