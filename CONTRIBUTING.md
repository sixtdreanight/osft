# Contributing to OSFT

Thanks for your interest in contributing.

## Ways to Contribute

- **Reproduce experiments**: Run the evaluation suite and verify results
- **Open questions**: The "SVD directions encode PDE eigenmodes" hypothesis (see blog post Section 8) needs verification — two low-cost experiments are designed but not yet run
- **New domains**: Try OSFT on other physics-constrained generation tasks (heat transfer, fluid dynamics, multi-physics)
- **Method improvements**: Better τ selection strategies, topology-aware losses, cross-domain adaptation
- **Bug reports & documentation**: File issues for anything unclear or broken

## Getting Started

```bash
git clone https://github.com/sixtdreanight/osft.git
cd osft
pip install -e ".[dev]"
pytest
```

## Pull Request Process

1. Open an issue describing what you plan to do
2. Keep changes focused and minimal
3. Include tests for new functionality
4. Ensure `pytest` passes before submitting
5. Update relevant documentation

## Code Style

- Python 3.10+ with type annotations
- Formatted with `black` (line length 100), imports sorted with `isort`
- Linted with `ruff`
- Follow existing patterns in `main/osft/` for consistency

## Experiment Naming

Experiments follow the convention `E{n}_{description}`:
- **E1**: Main benchmark
- **E2**: τ scan
- **E3**: Betti number subspace ablation
- **E4**: Gradient projection flow
- **E5**: CKA representational similarity
- **E6**: SVD spectral dynamics
- **E10**: Parameter efficiency
- **E13**: Pretraining scaling law
- **E14**: Per-layer freeze ablation
- **E16**: Fisher-SVD correlation
- **E_Jac**: Jacobian manifold diversity

See `docs/notes/` for complete experiment logs.

## Questions?

Open a [GitHub Discussion](https://github.com/sixtdreanight/osft/discussions) or issue.
