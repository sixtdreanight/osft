"""Spectral Analysis Engine for Generative Topology Models.

Comprehensive analysis toolkit covering:
  - Singular value spectrum & energy distribution
  - CKA (Centered Kernel Alignment) representation similarity
  - Jacobian effective rank, stable rank, participation ratio
  - Gradient projection ratio (physical gradient alignment with subspaces)
  - Layer-wise feature extraction & knowledge localization
  - Betti number approximation (topological feature counting)
  - Singular value dynamics during training

These tools support E2-E7 of the experimental framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, OrderedDict
from ..model.topologygan import TopologyGANGenerator
from ..eval.metrics import compute_all_image_metrics
from ..osft.decomposer import SVDWeightDecomposer
import warnings


# ============================================================
# 1. Singular Value Spectrum Analysis (E2, E6)
# ============================================================

class SingularValueAnalyzer:
    """Analyze singular value distributions of model weights.

    Supports:
      - Full SVD spectrum
      - Energy distribution by rank
      - Effective rank computation
      - Layer-wise singular value tracking over training
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.svd_history: List[Dict[str, np.ndarray]] = []

    @staticmethod
    def extract_conv_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract all Conv2d/ConvTranspose2d/OSFT weight matrices flattened to 2D."""
        weights = OrderedDict()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                w = module.weight.data.detach()
                weights[name] = w.view(w.size(0), -1)
            elif hasattr(module, '_full_weight'):
                w = module._full_weight().detach()
                weights[name] = w.view(w.size(0), -1)
        return weights

    @staticmethod
    def singular_value_spectrum(W: torch.Tensor, top_k: int = None) -> np.ndarray:
        """Compute full singular value spectrum of weight matrix."""
        with torch.no_grad():
            _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        sv = S.cpu().numpy()
        if top_k and top_k < len(sv):
            return sv[:top_k]
        return sv

    def layer_spectra(self) -> Dict[str, np.ndarray]:
        """Compute singular values for each conv layer."""
        weights = self.extract_conv_weights(self.model)
        spectra = {}
        for name, W in weights.items():
            spectra[name] = self.singular_value_spectrum(W)
        return spectra

    def energy_distribution(self, W: torch.Tensor, n_bins: int = 20) -> Dict[str, np.ndarray]:
        """Compute cumulative energy vs rank."""
        sv = self.singular_value_spectrum(W)
        total = (sv ** 2).sum()
        if total < 1e-12:
            return {"ranks": np.array([0]), "cumulative_energy": np.array([1.0])}

        cumsum = np.cumsum(sv ** 2) / total
        ranks = np.arange(1, len(sv) + 1)
        return {"ranks": ranks, "cumulative_energy": cumsum}

    def effective_rank(self, W: torch.Tensor) -> float:
        """Compute effective rank: exp(-sum(p_i * log(p_i))) where p_i = sigma_i^2 / sum(sigma^2).

        Higher effective rank → more uniformly distributed singular values → richer representation.
        """
        sv = self.singular_value_spectrum(W)
        sv_sq = sv ** 2
        total = sv_sq.sum()
        if total < 1e-12:
            return 0.0
        p = sv_sq / total
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))

    def stable_rank(self, W: torch.Tensor) -> float:
        """Compute stable rank: ||W||_F^2 / ||W||_2^2.

        Ratio of Frobenius norm squared to spectral norm squared.
        More robust than effective rank for near-zero singular values.
        """
        frob_sq = torch.norm(W, p='fro').item() ** 2
        spec_norm = torch.linalg.matrix_norm(W, ord=2).item()
        if spec_norm < 1e-12:
            return 0.0
        return frob_sq / (spec_norm ** 2)

    def participation_ratio(self, W: torch.Tensor) -> float:
        """Compute participation ratio: (sum sigma_i)^2 / sum(sigma_i^2).

        Measures how many singular vectors "participate" meaningfully.
        Range: [1, rank(W)]. Higher = more distributed.
        """
        sv = self.singular_value_spectrum(W)
        return float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12))

    def compute_all_ranks(self) -> Dict[str, Dict[str, float]]:
        """Compute all rank metrics for every conv layer."""
        weights = self.extract_conv_weights(self.model)
        results = {}
        for name, W in weights.items():
            results[name] = {
                "effective_rank": self.effective_rank(W),
                "stable_rank": self.stable_rank(W),
                "participation_ratio": self.participation_ratio(W),
                "n_singular_values": min(W.shape),
            }
        return results

    def snapshot_svd(self) -> Dict[str, np.ndarray]:
        """Take a snapshot of current singular values (for SVD dynamics tracking)."""
        spectra = self.layer_spectra()
        self.svd_history.append(spectra)
        return spectra

    def svd_dynamics_report(self) -> Dict[str, Dict[str, float]]:
        """Analyze how singular values changed over training snapshots.

        Returns per-layer metrics comparing first vs last snapshot.
        """
        if len(self.svd_history) < 2:
            return {}

        first = self.svd_history[0]
        last = self.svd_history[-1]
        report = {}

        for name in first:
            if name not in last:
                continue
            sv0 = first[name]
            sv1 = last[name]
            min_len = min(len(sv0), len(sv1))

            # Top-5 singular value relative change
            top_k = min(5, min_len)
            sv_change = np.abs(sv1[:top_k] - sv0[:top_k]) / (sv0[:top_k] + 1e-12)

            report[name] = {
                "top5_sv_mean_change": float(sv_change.mean()),
                "top5_sv_max_change": float(sv_change.max()),
                "frob_norm_change": float(np.abs((sv1**2).sum() - (sv0**2).sum()) / max((sv0**2).sum(), 1e-12)),
            }
        return report


# ============================================================
# 2. CKA (Centered Kernel Alignment) Analysis (E5)
# ============================================================

class CKAAnalyzer:
    """Centered Kernel Alignment for representation similarity.

    CKA(X, Y) = ||cov(X, Y)||_F^2 / (||cov(X, X)||_F * ||cov(Y, Y)||_F)

    Measures similarity between two sets of neural representations.
    Range: [0, 1]. 1 = identical up to orthogonal transformation.
    """

    def __init__(self, kernel: str = "linear"):
        """
        Args:
            kernel: 'linear' or 'rbf'. Linear is standard for CKA.
        """
        self.kernel = kernel

    @staticmethod
    def _gram_linear(X: torch.Tensor) -> torch.Tensor:
        """Linear kernel: K = X @ X^T."""
        return X @ X.T

    @staticmethod
    def _gram_rbf(X: torch.Tensor, sigma: float = None) -> torch.Tensor:
        """RBF kernel: K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))."""
        sq_dists = torch.cdist(X, X, p=2) ** 2
        if sigma is None:
            sigma = sq_dists.median().sqrt().item()
            if sigma < 1e-8:
                sigma = 1.0
        return torch.exp(-sq_dists / (2 * sigma ** 2))

    @staticmethod
    def _center(K: torch.Tensor) -> torch.Tensor:
        """Center a kernel matrix: H @ K @ H where H = I - (1/n)11^T."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    @staticmethod
    def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Hilbert-Schmidt Independence Criterion.

        HSIC = tr(HKH @ HLH) / (n-1)^2 where H = I - (1/n)11^T.
        """
        K_centered = CKAAnalyzer._center(K)
        L_centered = CKAAnalyzer._center(L)
        n = K.shape[0]
        return torch.trace(K_centered @ L_centered) / (n - 1) ** 2

    def compute(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> float:
        """Compute CKA between two sets of features.

        Args:
            X: [n_samples, d1] feature matrix
            Y: [n_samples, d2] feature matrix

        Returns:
            CKA similarity score in [0, 1]
        """
        if self.kernel == "linear":
            K = self._gram_linear(X)
            L = self._gram_linear(Y)
        else:
            K = self._gram_rbf(X)
            L = self._gram_rbf(Y)

        hsic_xy = self._hsic(K, L)
        hsic_xx = self._hsic(K, K)
        hsic_yy = self._hsic(L, L)

        denom = torch.sqrt(max(hsic_xx * hsic_yy, torch.tensor(1e-12)))
        cka = hsic_xy / denom
        return float(torch.clamp(cka, 0.0, 1.0).item())


class FeatureExtractor:
    """Extract intermediate features from a model for CKA analysis.

    Hooks into specified layers and collects activations during forward pass.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        self._hooks = []

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            # Flatten spatial dims: [B, C, H, W] -> [B, C*H*W]
            if isinstance(output, torch.Tensor):
                feat = output.detach()
                if feat.dim() == 4:
                    feat = feat.view(feat.size(0), -1)
                self.features[name].append(feat)
        return hook

    def register_hooks(self):
        """Register forward hooks on specified layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_features(self):
        """Clear accumulated features."""
        for name in self.features:
            self.features[name].clear()

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get concatenated features for each layer."""
        return {
            name: torch.cat(feats, dim=0) if feats else torch.tensor([])
            for name, feats in self.features.items()
        }

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


def compute_cka_between_models(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader,
    layer_names: List[str],
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Compute per-layer CKA between two models.

    Args:
        model_a: Reference model (e.g., pre-trained)
        model_b: Comparison model (e.g., fine-tuned)
        dataloader: DataLoader providing (real_A, real_B, ...)
        layer_names: Names of layers to compare
        device: torch.device
        max_batches: Max batches for feature extraction

    Returns:
        Dict[layer_name -> CKA score]
    """
    cka = CKAAnalyzer(kernel="linear")
    model_a.eval()
    model_b.eval()

    extractor_a = FeatureExtractor(model_a, layer_names)
    extractor_b = FeatureExtractor(model_b, layer_names)

    with extractor_a, extractor_b:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                real_A = batch[0].to(device)
                _ = model_a(real_A)
                _ = model_b(real_A)

    features_a = extractor_a.get_features()
    features_b = extractor_b.get_features()

    results = {}
    max_samples = 2000
    for name in layer_names:
        X = features_a.get(name)
        Y = features_b.get(name)
        if X is not None and Y is not None and X.numel() > 0 and Y.numel() > 0:
            X, Y = X.float(), Y.float()
            if X.shape[0] > max_samples:
                idx = torch.randperm(X.shape[0])[:max_samples]
                X, Y = X[idx], Y[idx]
            results[name] = cka.compute(X, Y)
        else:
            results[name] = float("nan")

    return results


# ============================================================
# 3. Gradient Projection Analysis (E4)
# ============================================================

class GradientProjectionAnalyzer:
    """Analyze gradient alignment with principal/residual subspaces.

    Tracks:
      - η = ||G_res||_F^2 / ||G_phy||_F^2
        (physical gradient energy in residual subspace)

      - θ = angle between physical gradient and residual subspace

    Key claim to verify:
      "Physical gradients predominantly lie in the residual subspace."
    """

    def __init__(self, decomposition_results: Dict[str, dict]):
        """
        Args:
            decomposition_results: Output of SVDWeightDecomposer.results
                Each entry has 'Ur', 'Vr', 'Unr', 'Vnr' tensors.
        """
        self.decomp = decomposition_results
        self.history: List[Dict[str, float]] = []

    def gradient_projection_ratio(
        self,
        layer_name: str,
        grad: torch.Tensor,
    ) -> float:
        """Compute η = ||Vnr^T @ grad @ Unr||_F^2 / ||grad||_F^2.

        Measures fraction of gradient energy that projects onto the residual subspace.
        """
        if layer_name not in self.decomp:
            return float("nan")

        d = self.decomp[layer_name]
        Vnr = d["Vnr"].to(grad.device)
        Unr = d["Unr"].to(grad.device)

        # Flatten gradient to match SVD shape
        grad_2d = grad.view(grad.size(0), -1) if grad.dim() == 4 else grad

        # Project onto residual subspace: Unr.T @ grad_2d @ Vnr
        # Unr: [m, k-r], grad_2d: [m, n], Vnr: [n, k-r]
        grad_res = Unr.T @ grad_2d @ Vnr
        grad_phy = grad_2d  # total physical gradient

        norm_res = torch.norm(grad_res, p="fro") ** 2
        norm_phy = torch.norm(grad_phy, p="fro") ** 2

        if norm_phy < 1e-12:
            return 0.0
        return float((norm_res / norm_phy).item())

    def snapshot_gradients(
        self,
        model: nn.Module,
        loss_fn: Callable,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Dict[str, float]:
        """Take one snapshot of gradient projection ratios for all OSFT layers.

        Reads the full-weight gradient from OSFT modules (retained via
        retain_grad() in forward), then projects onto residual subspace.

        Args:
            model: OSFT-augmented generator
            loss_fn: Function that takes (model, batch) and returns scalar loss
            batch: Input batch
            device: torch device

        Returns:
            Dict[layer_name -> η (gradient projection ratio)]
        """
        from ..osft.subspace_layers import (
            OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d,
            OrthogonalSubspaceLinear,
        )
        _OSFT_TYPES = (OrthogonalSubspaceConv2d, OrthogonalSubspaceConvTranspose2d,
                       OrthogonalSubspaceLinear)

        model.train()
        model.zero_grad()

        loss = loss_fn(model, batch)
        loss.backward()

        ratios = {}
        for module_name, module in model.named_modules():
            if (isinstance(module, _OSFT_TYPES) and module_name in self.decomp
                    and hasattr(module, '_weight_ref')):

                w_ref = module._weight_ref
                w_grad = w_ref.grad  # populated by retain_grad() in forward

                if w_grad is None:
                    continue

                ratio = self.gradient_projection_ratio(module_name, w_grad)
                ratios[module_name] = ratio

        self.history.append(ratios)
        return ratios

    def average_ratio(self) -> float:
        """Average η across all layers in the latest snapshot."""
        if not self.history:
            return float("nan")
        latest = self.history[-1]
        valid = [v for v in latest.values() if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    def gradient_flow_report(self) -> Dict[str, List[float]]:
        """Return per-epoch gradient projection history for each layer."""
        report = defaultdict(list)
        for snapshot in self.history:
            for name, ratio in snapshot.items():
                report[name].append(ratio)
        return dict(report)


# ============================================================
# 4. Jacobian Manifold Dimension (E7)
# ============================================================

class JacobianAnalyzer:
    """Analyze Jacobian of generator to measure manifold dimension.

    Jacobian J = ∂G(z)/∂z has singular values that indicate
    the local dimensionality of the generated manifold.

    Metrics:
      - Effective Rank: exp(-sum(p_i log p_i)) where p_i = sigma_i^2 / sum(sigma^2)
      - Stable Rank: ||J||_F^2 / ||J||_2^2
      - Participation Ratio: (sum sigma_i)^2 / sum(sigma_i^2)
    """

    def __init__(self, generator: nn.Module, nz: int = 100):
        self.generator = generator
        self.nz = nz

    def compute_jacobian_svd(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        n_directions: int = 200,
    ) -> np.ndarray:
        """Approximate Jacobian singular values via finite-difference JVPs.

        For each random unit direction v_i, computes ||J @ v_i|| ≈ ||G(z+εv_i) - G(z)||/ε.
        The set of ||J@v_i|| values approximates the singular value spectrum
        (randomized range-finding, not full SVD).

        Args:
            z: Latent vector [1, nz] or [B, nz]
            condition: Input conditions [1, C, H, W] or [B, C, H, W]
            n_directions: Number of random probing directions

        Returns:
            Approximate singular values [n_directions]
        """
        self.generator.eval()
        device = z.device
        B = z.size(0)

        # Random directions
        directions = torch.randn(n_directions, self.nz, device=device)
        directions /= directions.norm(dim=1, keepdim=True)

        sv_approx = torch.zeros(n_directions, device=device)
        eps = 1e-3

        with torch.no_grad():
            for i in range(n_directions):
                v = directions[i:i+1].expand(B, -1)
                # Compute J @ v via finite differences: (G(z + εv) - G(z)) / ε
                G_plus = self.generator(condition, z + eps * v)
                G_base = self.generator(condition, z)
                Jv = (G_plus - G_base) / eps
                # ||J @ v|| approximates a singular value
                sv_approx[i] = Jv.norm()

        return sv_approx.cpu().numpy()

    def manifold_dimension_metrics(
        self,
        dataloader,
        device: torch.device,
        n_samples: int = 50,
        n_directions: int = 200,
    ) -> Dict[str, float]:
        """Compute manifold dimension metrics averaged over samples.

        Returns:
            Dict with effective_rank, stable_rank, participation_ratio
        """
        self.generator.eval()
        all_eff_rank, all_stable_rank, all_part_ratio = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= n_samples:
                    break
                real_A = batch[0][:1].to(device)  # Take 1 sample per batch
                z = torch.randn(1, self.nz, device=device)

                sv = self.compute_jacobian_svd(z, real_A, n_directions)

                sv_sq = sv ** 2
                total = sv_sq.sum()
                if total > 1e-12:
                    p = sv_sq / total
                    p = p[p > 1e-12]
                    entropy = -np.sum(p * np.log(p))
                    eff_r = np.exp(entropy)
                else:
                    eff_r = 0.0

                frob_sq = sv_sq.sum()
                spec_sq = sv[0] ** 2 if len(sv) > 0 else 0
                stable_r = frob_sq / max(spec_sq, 1e-12)

                part_r = sv.sum() ** 2 / max(sv_sq.sum(), 1e-12)

                all_eff_rank.append(eff_r)
                all_stable_rank.append(stable_r)
                all_part_ratio.append(part_r)

        return {
            "effective_rank": float(np.mean(all_eff_rank)),
            "stable_rank": float(np.mean(all_stable_rank)),
            "participation_ratio": float(np.mean(all_part_ratio)),
            "n_samples": len(all_eff_rank),
        }


# ============================================================
# 5. Betti Number Approximation (E3 supplementary)
# ============================================================

class TopologyFeatureAnalyzer:
    """Approximate topological features (Betti numbers β0, β1) for 2D density fields.

    β0 = number of connected components
    β1 = number of holes

    Uses simple image processing (connected components + hole counting)
    as a lightweight alternative to persistent homology.
    """

    @staticmethod
    def betti_numbers(density: np.ndarray, threshold: float = 0.5) -> Tuple[int, int]:
        """Compute β0 and β1 for a 2D binary topology.

        Args:
            density: [H, W] density field
            threshold: Binarization threshold

        Returns:
            (beta0, beta1) = (connected components, holes)
        """
        binary = (density > threshold).astype(np.uint8)

        # β0: Count connected components in solid phase
        beta0 = TopologyFeatureAnalyzer._count_components(binary)

        # β1: Count holes (connected components in void phase within the bounding box,
        #      minus 1 for the exterior)
        void = 1 - binary
        # Pad with solid border to eliminate exterior void component
        padded = np.pad(void, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        beta1 = TopologyFeatureAnalyzer._count_components(padded) - 1
        beta1 = max(beta1, 0)

        return beta0, beta1

    @staticmethod
    def _count_components(binary: np.ndarray) -> int:
        """Count connected components using flood-fill."""
        try:
            from scipy.ndimage import label
            labeled, n_components = label(binary)
            return n_components
        except ImportError:
            # Fallback: BFS connected components
            from collections import deque
            visited = np.zeros_like(binary, dtype=bool)
            count = 0
            H, W = binary.shape
            for i in range(H):
                for j in range(W):
                    if binary[i, j] and not visited[i, j]:
                        count += 1
                        q = deque([(i, j)])
                        visited[i, j] = True
                        while q:
                            ci, cj = q.popleft()
                            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                ni, nj = ci + di, cj + dj
                                if 0 <= ni < H and 0 <= nj < W and binary[ni, nj] and not visited[ni, nj]:
                                    visited[ni, nj] = True
                                    q.append((ni, nj))
            return count

    @classmethod
    def betti_change(
        cls,
        generated: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute Betti number changes between generated and reference topology.

        Returns:
            Dict with delta_beta0, delta_beta1, beta0_ref, beta0_gen, beta1_ref, beta1_gen
        """
        b0_gen, b1_gen = cls.betti_numbers(generated, threshold)
        b0_ref, b1_ref = cls.betti_numbers(reference, threshold)

        return {
            "beta0_reference": b0_ref,
            "beta0_generated": b0_gen,
            "delta_beta0": b0_gen - b0_ref,
            "beta1_reference": b1_ref,
            "beta1_generated": b1_gen,
            "delta_beta1": b1_gen - b1_ref,
        }

    @classmethod
    def batch_betti_analysis(
        cls,
        fake_batch: torch.Tensor,
        real_batch: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Analyze Betti numbers over a batch.

        Args:
            fake_batch: [B, 1, H, W]
            real_batch: [B, 1, H, W]

        Returns:
            Averaged Betti metrics
        """
        B = fake_batch.size(0)
        deltas_b0, deltas_b1 = [], []
        b0_refs, b1_refs, b0_gens, b1_gens = [], [], [], []

        for i in range(B):
            fake = fake_batch[i, 0].cpu().numpy()
            real = real_batch[i, 0].cpu().numpy()
            result = cls.betti_change(fake, real, threshold)
            deltas_b0.append(result["delta_beta0"])
            deltas_b1.append(result["delta_beta1"])
            b0_refs.append(result["beta0_reference"])
            b0_gens.append(result["beta0_generated"])
            b1_refs.append(result["beta1_reference"])
            b1_gens.append(result["beta1_generated"])

        return {
            "mean_abs_delta_beta0": float(np.mean(np.abs(deltas_b0))),
            "mean_abs_delta_beta1": float(np.mean(np.abs(deltas_b1))),
            "beta0_preservation": float(np.mean(np.array(b0_gens) == np.array(b0_refs))),
            "beta1_preservation": float(np.mean(np.array(b1_gens) == np.array(b1_refs))),
        }


# ============================================================
# 6. Comprehensive Analysis Suite
# ============================================================

class SpectralAnalysisSuite:
    """Unified interface for all spectral analyses.

    Usage:
        suite = SpectralAnalysisSuite(generator, decomposition_results)
        # E2: Singular value scan
        report = suite.tau_scan_analysis(tau_values=[0.1, 0.3, ..., 0.99])
        # E4: Gradient flow
        suite.track_gradient_flow(trainer, dataloader, n_epochs=100)
        # E5: CKA comparison
        cka_results = suite.compare_representations(model_a, model_b, dataloader)
        # E6: SVD dynamics
        suite.track_svd_dynamics(trainer, dataloader, snapshot_every=5)
        # E7: Manifold dimension
        dim_metrics = suite.manifold_dimensions(dataloader)
    """

    def __init__(
        self,
        generator: nn.Module,
        decomposition_results: Optional[Dict[str, dict]] = None,
    ):
        self.generator = generator
        self.decomp_results = decomposition_results or {}
        self.svd_analyzer = SingularValueAnalyzer(generator)
        self.grad_analyzer = GradientProjectionAnalyzer(self.decomp_results)
        self.jacobian_analyzer = JacobianAnalyzer(generator)

    def tau_scan_analysis(
        self,
        tau_values: List[float],
        dataloader,
        device: torch.device,
        pretrained_path: str,
    ) -> Dict[str, List[float]]:
        """E2: Scan τ from low to high, measure topology preservation.

        For each τ (energy threshold):
          - Decompose generator with that threshold
          - Generate samples
          - Compute Betti preservation, compliance, SSIM

        Returns time series for each metric across τ values.
        """
        results = {
            "tau": tau_values,
            "compliance": [],
            "ssim": [],
            "mse": [],
            "beta0_preservation": [],
            "beta1_preservation": [],
            "trainable_pct": [],
            "effective_rank": [],
        }

        for tau in tau_values:
            # Fresh generator for each tau
            gen = TopologyGANGenerator(
                input_c_dim=3, output_c_dim=1, gf_dim=128,
                variant="se_res_unet", height=64, width=128,
            ).to(device)
            gen.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

            # Decompose at this tau
            decomposer = SVDWeightDecomposer(energy_threshold=tau)
            decomposer.decompose_model(gen, verbose=False)
            summary = decomposer.summary()

            # Replace each conv layer's weight with Wr-only (principal subspace)
            # to evaluate information retained at this tau
            for name, module in gen.named_modules():
                if name in decomposer.results and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    Wr = decomposer.results[name]["Wr"]
                    with torch.no_grad():
                        module.weight.data = Wr.view(module.weight.shape)

            trainable_p = summary["total_residual_dim"]
            total_p = summary["total_params_orig"]
            results["trainable_pct"].append(100 * trainable_p / max(total_p, 1))

            # Generate and evaluate (using frozen generator, no fine-tuning)
            gen.eval()
            all_mse, all_ssim = [], []
            all_b0_pres, all_b1_pres = [], []

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 10:
                        break
                    real_A, real_B = batch[0].to(device), batch[1].to(device)
                    fake_B = gen(real_A)

                    metrics = compute_all_image_metrics(fake_B, real_B)
                    all_mse.append(metrics["mse"])
                    all_ssim.append(metrics["ssim"])

                    betti = TopologyFeatureAnalyzer.batch_betti_analysis(fake_B, real_B)
                    all_b0_pres.append(betti["beta0_preservation"])
                    all_b1_pres.append(betti["beta1_preservation"])

            results["mse"].append(float(np.mean(all_mse)))
            results["ssim"].append(float(np.mean(all_ssim)))
            results["beta0_preservation"].append(float(np.mean(all_b0_pres)))
            results["beta1_preservation"].append(float(np.mean(all_b1_pres)))
            results["effective_rank"].append(summary["avg_rank_ratio"])

        return results

    def track_gradient_flow(
        self,
        trainer,
        dataloader,
        n_epochs: int,
        snapshot_every: int = 1,
    ) -> Dict[str, List[float]]:
        """E4: Track gradient projection ratio η during training.

        Returns per-epoch average η across layers.
        """
        device = trainer.device
        eta_history = []

        for epoch in range(n_epochs):
            epoch_etas = []
            for batch in dataloader:
                def phys_loss(model, batch_in):
                    real_A = batch_in[0].to(device)
                    real_B = batch_in[1].to(device)
                    bc = batch_in[2].to(device) if len(batch_in) > 2 else None
                    lx = batch_in[3].to(device) if len(batch_in) > 3 else None
                    ly = batch_in[4].to(device) if len(batch_in) > 4 else None
                    fake_B = model(real_A)
                    phys = trainer.physics_loss(fake_B, real_B, bc, lx, ly)
                    return phys["total"]

                ratios = self.grad_analyzer.snapshot_gradients(
                    trainer.generator, phys_loss, batch, device,
                )
                avg = np.mean([v for v in ratios.values() if not np.isnan(v)])
                epoch_etas.append(avg)
                break  # One batch per epoch for efficiency

            mean_eta = float(np.mean(epoch_etas))
            eta_history.append(mean_eta)

            # Train one epoch
            trainer.train_epoch(dataloader, epoch)

        return {"epoch": list(range(n_epochs)), "eta": eta_history}

    def compare_representations(
        self,
        pretrained_model: nn.Module,
        fine_tuned_model: nn.Module,
        dataloader,
        device: torch.device,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """E5: CKA representation similarity between models.

        Returns per-layer CKA scores.
        """
        if layer_names is None:
            layer_names = []
            for name, _ in pretrained_model.named_modules():
                if isinstance(_, (nn.Conv2d, nn.ConvTranspose2d)):
                    layer_names.append(name)

        return compute_cka_between_models(
            pretrained_model, fine_tuned_model,
            dataloader, layer_names, device,
        )

    def track_svd_dynamics(
        self,
        trainer,
        dataloader,
        n_epochs: int,
        snapshot_every: int = 5,
    ) -> Dict[str, List[Dict]]:
        """E6: Track singular value changes during training.

        Returns list of per-snapshot SVD analysis results.
        """
        snapshots = []
        svd_analyzer = SingularValueAnalyzer(trainer.generator)

        # Initial snapshot
        svd_analyzer.snapshot_svd()
        snapshots.append({"epoch": 0, "ranks": svd_analyzer.compute_all_ranks()})

        for epoch in range(n_epochs):
            trainer.train_epoch(dataloader, epoch)
            if (epoch + 1) % snapshot_every == 0:
                svd_analyzer.snapshot_svd()
                snapshots.append({
                    "epoch": epoch + 1,
                    "ranks": svd_analyzer.compute_all_ranks(),
                })

        dynamics_report = svd_analyzer.svd_dynamics_report()
        return {
            "snapshots": snapshots,
            "dynamics_report": dynamics_report,
        }

    def manifold_dimensions(
        self,
        dataloader,
        device: torch.device,
        n_samples: int = 50,
    ) -> Dict[str, float]:
        """E7: Measure Jacobian manifold dimension."""
        return self.jacobian_analyzer.manifold_dimension_metrics(
            dataloader, device, n_samples,
        )
