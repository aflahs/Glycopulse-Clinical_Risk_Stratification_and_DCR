"""
Mahalanobis Latent Manifold Out-Of-Distribution (OOD) Detector
Author: MUHAMMED AFLAH S (RA2411026050147)
Reference: Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples", NeurIPS 2018.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn


class MahalanobisOODDetector:
    """
    Detects anomalous or out-of-distribution clinical presentations in the
    model's latent bottleneck manifold h in R^d.
    
    Fits a multivariate Gaussian N(mu, Sigma) over training latent vectors:
        D_M(h) = sqrt( (h - mu)^T Sigma^{-1} (h - mu) )
    
    Clinical values with extreme Mahalanobis distances are flagged before clinical triage.
    """
    def __init__(self, alpha: float = 0.95, reg_cov: float = 1e-4):
        self.alpha = alpha
        self.reg_cov = reg_cov
        self.mean: np.ndarray = None
        self.inv_cov: np.ndarray = None
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def fit(self, latent_vectors: np.ndarray) -> "MahalanobisOODDetector":
        """
        Fits empirical mean and regularized inverse covariance matrix.
        
        Args:
            latent_vectors: Array of shape (N, d_model) representing training latent codes.
        """
        assert latent_vectors.ndim == 2, f"Expected 2D array, got shape {latent_vectors.shape}"
        N, D = latent_vectors.shape

        self.mean = np.mean(latent_vectors, axis=0)
        cov = np.cov(latent_vectors, rowvar=False)

        # Regularize covariance to ensure well-conditioned inversion
        cov_reg = cov + np.eye(D) * self.reg_cov
        self.inv_cov = np.linalg.pinv(cov_reg)

        # Compute empirical distances on training set to establish threshold
        diff = latent_vectors - self.mean
        dist_sq = np.sum(diff @ self.inv_cov * diff, axis=1)
        distances = np.sqrt(np.maximum(dist_sq, 0.0))

        # Decision boundary at alpha-th percentile (e.g. 95%)
        self.threshold = float(np.percentile(distances, self.alpha * 100))
        self.is_fitted = True
        return self

    def score(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Computes Mahalanobis distances for query latent vectors.
        
        Args:
            latent_vectors: Array of shape (M, d_model).
        Returns:
            distances: 1D array of shape (M,).
        """
        if not self.is_fitted:
            raise RuntimeError("MahalanobisOODDetector must be fitted before scoring.")
        if latent_vectors.ndim == 1:
            latent_vectors = latent_vectors.reshape(1, -1)

        diff = latent_vectors - self.mean
        dist_sq = np.sum(diff @ self.inv_cov * diff, axis=1)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def predict(self, latent_vectors: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates OOD status for query latent vectors.
        
        Returns:
            Dict containing:
              - 'distances': List[float]
              - 'is_ood': List[bool]
              - 'threshold': float
        """
        distances = self.score(latent_vectors)
        is_ood = distances > self.threshold
        return {
            "distances": [float(d) for d in distances],
            "is_ood": [bool(b) for b in is_ood],
            "threshold": self.threshold
        }
