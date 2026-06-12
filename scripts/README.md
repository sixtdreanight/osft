# Scripts

Experiment and utility scripts. Run from the project root:

```bash
python scripts/<script_name>.py
```

## Core Experiments

| Script | Experiment | Description |
|--------|-----------|-------------|
| `run_e1_simp.py` | E1 | Main benchmark on SIMP-generated data |
| `run_e1_real.py` | E1 | Cross-domain: synthetic PT → real data |
| `run_e1_real_pretrained.py` | E1 | In-domain: real PT → real data |
| `run_experiments_phase2.py` | E3, E4, E11, E_Jac, FIX1, FIX2 | Phase 2 suite: Betti ablation, gradient flow, FEM validation, Jacobian analysis |
| `run_scaling_lean.py` | E13 | Pretraining scaling law (7 checkpoints) |
| `run_curriculum.py` | E_EXP1 | Curriculum learning on physics loss weight |
| `run_e15_generalization.py` | E15 | Multi-structure cross-domain generalization matrix |

## Analysis & Evaluation

| Script | Description |
|--------|-------------|
| `eval_final.py` | Batch evaluation across all checkpoints |
| `eval_checkpoints.py` | Single checkpoint evaluation |
| `gen_paper_tables.py` | Generate LaTeX tables from result JSONs |
| `viz_results.py` | Topology comparison visualization |
| `eigenmode_correlation.py` | SVD-eigenmode correlation analysis |
| `eigenmode_svd_perturb.py` | SVD direction perturbation experiment |
| `physics_latent_probe.py` | Physics latent space probing |

## Data Generation

| Script | Description |
|--------|-------------|
| `generate_simp_data.py` | SIMP solver: MBB beam, L-beam, bridge, cantilever |
| `generate_synthetic_data.py` | Synthetic random smooth field data |
| `convert_dataset.py` | Convert CSV datasets to 7-channel .npy format |

## Training Variants (exploratory)

| Script | Description |
|--------|-------------|
| `train_physics_gan.py` | Train GAN with physics loss from scratch |
| `train_physics_latent_gan.py` | Physics-conditioned latent GAN |
| `train_vf_conditioned_gan.py` | Volume-fraction conditioned GAN |
| `train_taylor_gan.py` | Taylor expansion GAN (hierarchical corrections) |
| `test_inverted_osft.py` | Inverted OSFT: freeze residual, train principal |

## Legacy

| Script | Notes |
|--------|-------|
| `quickstart.py` | Initial smoke test — superseded |
| `run_overnight.py` | Early experiment runner — superseded |
| `run_scaling.py` | E13 v1 — use `run_scaling_lean.py` |
| `run_e15_generalization.py` | E15 — completed |
