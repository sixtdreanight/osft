# Scripts Manifest

## Core experiments
| Script | Purpose | Status |
|--------|---------|--------|
| `run_e15_generalization.py` | E15 multi-structure cross-domain matrix | 🔄 running |
| `run_e1_real.py` | E1 on real Cantilever (synth PT) | ✅ done |
| `run_e1_real_pretrained.py` | E1 on real Cantilever (real PT) | ✅ done |
| `run_experiments_phase2.py` | E3/E4/E11/E_Jac/FIX1/FIX2 | ✅ done |
| `run_scaling_lean.py` | E13 scaling law | ✅ done |
| `run_curriculum.py` | E_EXP1 curriculum learning | ✅ done |

## Data generation
| Script | Purpose |
|--------|---------|
| `generate_simp_data.py` | SIMP multi-structure data (MBB/L-Beam/Bridge) |
| `generate_synthetic_data.py` | Synthetic random field data |
| `convert_dataset.py` | CSV → 7-channel .npy converter |

## Analysis & Visualization
| Script | Purpose |
|--------|---------|
| `eval_final.py` | Batch checkpoint evaluation |
| `gen_paper_tables.py` | LaTeX table generation |
| `viz_results.py` | Topology comparison visualization |
| `eval_checkpoints.py` | Single checkpoint evaluation |

## Deprecated (kept for reference)
| Script | Why deprecated |
|--------|---------------|
| `quickstart.py` | Initial smoke test, superseded |
| `run_overnight.py` | Early experiment runner |
| `run_scaling.py` | v1 of scaling, use `run_scaling_lean.py` |
