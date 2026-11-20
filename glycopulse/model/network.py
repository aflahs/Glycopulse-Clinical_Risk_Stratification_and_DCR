"""
GLYCOPULSE-NET Master Network Architecture
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from glycopulse.config import ModelConfig
from glycopulse.model.tokenization import BiomarkerGatedTokenizer
from glycopulse.model.attention import ResidualCrossAttentionBlock, LatentBottleneckPooling


class DecoupledTaskHead(nn.Module):
    """
    Decoupled Task Prediction Head with Monte Carlo Variational Dropout.
    Maps latent patient vector h -> logit for specific clinical endpoint.
    """
    def __init__(self, in_features: int = 128, hidden_dim: int = 64, mc_dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=mc_dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, in_features) -> (B, 1)
        return self.net(h)


class GlycoPulseNet(nn.Module):
    """
    GLYCOPULSE-NET: Multi-Task Clinical Risk Stratification Model.
    
    Architecture Pipeline:
      1. Biomarker Gated Tokenization (BGT): Projects 24 scalar biomarkers to d_model tokens via SwiGLU.
      2. Residual Cross-Attention (ResCross-Net): L layers of multi-head cross/self-attention.
      3. Latent Bottleneck Pooling: Compresses sequence into unified patient representation vector h.
      4. Decoupled Latent Heads: Computes calibrated risks for T2DM, Nephropathy, and ASCVD.
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # 1. Biomarker Gated Tokenizer
        self.tokenizer = BiomarkerGatedTokenizer(
            num_features=self.config.num_features,
            d_model=self.config.d_model,
            dropout=self.config.dropout
        )

        # 2. Residual Cross-Attention Blocks
        self.layers = nn.ModuleList([
            ResidualCrossAttentionBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout
            )
            for _ in range(self.config.num_layers)
        ])

        # 3. Latent Bottleneck Pooling
        self.bottleneck_pool = LatentBottleneckPooling(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout
        )

        # 4. Decoupled Multi-Task Heads
        self.head_t2dm = DecoupledTaskHead(
            in_features=self.config.d_model,
            hidden_dim=self.config.d_model // 2,
            mc_dropout=self.config.mc_dropout
        )
        self.head_nephropathy = DecoupledTaskHead(
            in_features=self.config.d_model,
            hidden_dim=self.config.d_model // 2,
            mc_dropout=self.config.mc_dropout
        )
        self.head_ascvd = DecoupledTaskHead(
            in_features=self.config.d_model,
            hidden_dim=self.config.d_model // 2,
            mc_dropout=self.config.mc_dropout
        )

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the latent patient representation vector h (B, d_model)
        for manifold geometry and Mahalanobis OOD density estimation.
        """
        tokens = self.tokenizer(x)  # (B, N, d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        h = self.bottleneck_pool(tokens)  # (B, d_model)
        return h

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Raw or normalized clinical biomarker tensor of shape (B, 24).
            return_latent: If True, also returns the latent representation h.
        Returns:
            logits: Tensor of shape (B, 3) containing logits for [T2DM, Nephropathy, ASCVD].
            h: Optional latent vector of shape (B, d_model).
        """
        h = self.extract_latent(x)
        logit_t2dm = self.head_t2dm(h)
        logit_neph = self.head_nephropathy(h)
        logit_ascvd = self.head_ascvd(h)

        logits = torch.cat([logit_t2dm, logit_neph, logit_ascvd], dim=-1)  # (B, 3)

        if return_latent:
            return logits, h
        return logits

    def predict_monte_carlo(
        self,
        x: torch.Tensor,
        num_samples: int = 30,
        temperature: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Bayesian Monte Carlo Variational Dropout Inference.
        Keeps dropout active during evaluation to compute epistemic uncertainty.
        
        Args:
            x: Input tensor (B, 24).
            num_samples: Number of forward stochastic samples M.
            temperature: Optional temperature scaling vector (3,).
        Returns:
            Dict containing:
              - 'mean_probs': (B, 3) Predictive posterior mean probability.
              - 'epistemic_std': (B, 3) Epistemic uncertainty (std across passes).
              - 'samples': (M, B, 3) All raw probability samples.
        """
        self.eval()
        # Force dropout layers to remain in training mode for MC sampling
        def enable_mc_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        self.apply(enable_mc_dropout)

        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                if temperature is not None:
                    logits = logits / temperature
                probs = torch.sigmoid(logits)
                samples.append(probs.unsqueeze(0))

        # Reset back to full eval mode
        self.eval()

        samples_tensor = torch.cat(samples, dim=0)  # (M, B, 3)
        mean_probs = torch.mean(samples_tensor, dim=0)  # (B, 3)
        epistemic_std = torch.std(samples_tensor, dim=0)  # (B, 3)

        return {
            "mean_probs": mean_probs,
            "epistemic_std": epistemic_std,
            "samples": samples_tensor
        }
