"""
Biomarker Gated Tokenization (BGT) Module
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUGating(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) feedforward layer:
    SwiGLU(x) = (Silu(x W_gate) * (x W_value)) W_out
    Provides non-linear feature interaction and sparse gating.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_val = nn.Linear(in_features, hidden_features, bias=False)
        self.w_out = nn.Linear(hidden_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        gate = F.silu(self.w_gate(x))
        val = self.w_val(x)
        out = self.w_out(gate * val)
        return self.dropout(out)


class BiomarkerGatedTokenizer(nn.Module):
    """
    Transforms raw continuous clinical biomarkers into high-dimensional tokens.
    Input: (B, num_features)
    Output: (B, num_features, d_model)
    
    Each scalar x_i is projected to d_model, enriched with a learned biomarker
    identity embedding, and passed through SwiGLU gating with residual normalization.
    """
    def __init__(self, num_features: int = 24, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Scalar projection weight and bias per biomarker (num_features independent linear heads)
        self.weight_proj = nn.Parameter(torch.Tensor(num_features, d_model))
        self.bias_proj = nn.Parameter(torch.Tensor(num_features, d_model))

        # Learned categorical identity embedding per biomarker
        self.biomarker_embed = nn.Embedding(num_features, d_model)

        # SwiGLU gating block applied per token
        self.swiglu = SwiGLUGating(
            in_features=d_model,
            hidden_features=d_model * 2,
            out_features=d_model,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_proj, a=math.sqrt(5))
        nn.init.zeros_(self.bias_proj)
        nn.init.normal_(self.biomarker_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, num_features) containing continuous clinical values.
        Returns:
            Tensor of shape (B, num_features, d_model).
        """
        B, N = x.shape
        assert N == self.num_features, f"Expected {self.num_features} features, got {N}"

        # Expand x to (B, N, 1) and project with per-feature weights:
        # x_expanded: (B, N, 1), weight_proj: (1, N, d_model) -> (B, N, d_model)
        x_expanded = x.unsqueeze(-1)
        z = x_expanded * self.weight_proj.unsqueeze(0) + self.bias_proj.unsqueeze(0)

        # Add learned biomarker identity embedding
        token_indices = torch.arange(self.num_features, device=x.device).unsqueeze(0).expand(B, -1)
        id_embeds = self.biomarker_embed(token_indices)
        z = z + id_embeds

        # Apply SwiGLU non-linear gating with residual skip connection
        gated = self.swiglu(z)
        tokens = self.layer_norm(z + gated)

        return tokens
