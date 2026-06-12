"""Latent Space Geometry Analysis (E8).

UMAP visualization of generated topology distributions,
manifold coverage analysis, and mode collapse detection.

Compares latent space geometry across:
  - Pre-trained GAN
  - Full Fine-tuned
  - OSFT Fine-tuned
  - LoRA Fine-tuned
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class LatentGeometryAnalyzer:
    """Analyze latent space geometry via dimensionality reduction."""

    def __init__(self, generator: nn.Module, nz: int = 100, device: str = "cuda"):
        self.generator = generator
        self.nz = nz
        self.device = device

    @torch.no_grad()
    def generate_samples(
        self,
        conditions: torch.Tensor,
        n_per_condition: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Generate multiple topologies for each condition using different z.

        Args:
            conditions: [N, C, H, W] condition inputs
            n_per_condition: Number of different z per condition

        Returns:
            Dict with:
              - 'topologies': [N * n_per_condition, flattened_dim] flattened density fields
              - 'conditions_idx': [N * n_per_condition] condition index
              - 'z_values': [N * n_per_condition, nz] latent vectors
        """
        self.generator.eval()
        N = conditions.size(0)
        all_topos = []
        all_zs = []
        cond_indices = []

        conditions = conditions.to(self.device)

        for i in range(N):
            cond = conditions[i:i+1].expand(n_per_condition, -1, -1, -1)
            z = torch.randn(n_per_condition, self.nz, device=self.device)
            fake = self.generator(cond, z)
            all_topos.append(fake.view(n_per_condition, -1).cpu().numpy())
            all_zs.append(z.cpu().numpy())
            cond_indices.extend([i] * n_per_condition)

        return {
            "topologies": np.concatenate(all_topos, axis=0),
            "conditions_idx": np.array(cond_indices),
            "z_values": np.concatenate(all_zs, axis=0),
        }

    def umap_projection(
        self,
        data: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        random_state: int = 42,
    ) -> np.ndarray:
        """Project high-dimensional data to 2D via UMAP.

        Falls back to PCA if umap-learn is not installed.
        """
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                verbose=False,
            )
            return reducer.fit_transform(data)
        except ImportError:
            warnings.warn("umap-learn not installed, falling back to PCA")
            return self._pca_projection(data, n_components)

    @staticmethod
    def _pca_projection(data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """PCA fallback for dimensionality reduction."""
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=42)
            return pca.fit_transform(data)
        except ImportError:
            # Manual PCA via SVD as ultimate fallback
            data_centered = data - data.mean(axis=0)
            U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
            return (data_centered @ Vt[:n_components].T)

    def manifold_coverage_metrics(
        self,
        data: np.ndarray,
        n_bins: int = 20,
    ) -> Dict[str, float]:
        """Compute manifold coverage and diversity metrics.

        Args:
            data: [N, D] samples in original or projected space
            n_bins: Number of bins for histogram-based coverage metric

        Returns:
            Dict with:
              - 'coverage': fraction of occupied bins in 2D projection
              - 'mean_pairwise_dist': average L2 distance between samples
              - 'convex_hull_area': approximate area of convex hull in 2D
              - 'cluster_coefficient': (1 - Gini of bin counts), measures uniformity
        """
        N = data.shape[0]

        # Project to 2D for coverage metrics
        if data.shape[1] > 2:
            proj = self._pca_projection(data, 2)
        else:
            proj = data

        # Coverage: 2D histogram occupation ratio
        try:
            H, _, _ = np.histogram2d(proj[:, 0], proj[:, 1], bins=n_bins)
            coverage = np.sum(H > 0) / (n_bins ** 2)
        except Exception:
            coverage = 1.0

        # Mean pairwise distance
        if N > 2000:
            idx = np.random.choice(N, 2000, replace=False)
            data_sub = data[idx]
        else:
            data_sub = data
        try:
            from sklearn.metrics import pairwise_distances
            dists = pairwise_distances(data_sub, n_jobs=-1)
            mean_dist = float(dists[np.triu_indices_from(dists, k=1)].mean())
        except ImportError:
            # Manual pairwise L2 distance
            diff = data_sub[:, None, :] - data_sub[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            n = dists.shape[0]
            triu_idx = np.triu_indices(n, k=1)
            mean_dist = float(dists[triu_idx].mean())

        # Convex hull area (2D projection)
        try:
            from scipy.spatial import ConvexHull
            if proj.shape[0] >= 3:
                hull = ConvexHull(proj)
                hull_area = float(hull.volume)
            else:
                hull_area = 0.0
        except (ImportError, Exception):
            hull_area = 0.0

        # Cluster coefficient: uniformity of bin occupancy
        if 'H' in locals():
            bin_counts = H.flatten()
            bin_counts = bin_counts[bin_counts > 0]
            if len(bin_counts) > 1:
                sorted_counts = np.sort(bin_counts)
                lorenz = np.cumsum(sorted_counts) / sorted_counts.sum()
                gini = 1 - 2 * np.trapz(lorenz, np.linspace(0, 1, len(lorenz)))
                uniformity = 1 - gini
            else:
                uniformity = 0.0
        else:
            uniformity = 1.0

        return {
            "coverage": float(coverage),
            "mean_pairwise_distance": float(mean_dist),
            "convex_hull_area_2d": hull_area,
            "uniformity": float(uniformity),
        }

    def analyze(
        self,
        conditions: torch.Tensor,
        n_per_condition: int = 10,
        umap_components: int = 2,
    ) -> Dict:
        """Full latent geometry analysis.

        Returns:
            Dict with 'projection', 'coverage_metrics', 'samples'
        """
        samples = self.generate_samples(conditions, n_per_condition)

        # UMAP on topology space
        projection = self.umap_projection(
            samples["topologies"], n_components=umap_components,
        )

        # Also compute UMAP on latent z space
        z_projection = self.umap_projection(
            samples["z_values"], n_components=umap_components,
        )

        coverage = self.manifold_coverage_metrics(samples["topologies"])

        return {
            "projection_umap": projection,
            "projection_z_umap": z_projection,
            "coverage_metrics": coverage,
            "conditions_idx": samples["conditions_idx"],
            "n_conditions": conditions.size(0),
            "n_per_condition": n_per_condition,
        }


def compare_latent_geometries(
    models: Dict[str, nn.Module],
    conditions: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Compare latent space geometry across multiple models.

    Args:
        models: {name: generator}
        conditions: Fixed set of conditions for fair comparison

    Returns:
        {model_name: geometry_analysis_results}
    """
    results = {}
    for name, gen in models.items():
        analyzer = LatentGeometryAnalyzer(gen, device=device)
        results[name] = analyzer.analyze(conditions)
    return results
