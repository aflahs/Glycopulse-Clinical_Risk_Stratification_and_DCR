"""
Residual Cross-Attention and Latent Bottleneck Pooling Blocks
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCrossAttentionBlock(nn.Module):
    """
    Residual Cross-Attention Block (ResCross-Net) with Pre-LayerNorm.
    Features:
      1. Pre-LN Multi-Head Self/Cross Attention across biomarker tokens.
      2. Residual skip connections.
      3. Position-wise Feedforward Network with GeLU/Swish activation.
      4. Dropout regularization.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dim_feedforward: int = 256, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-Head Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        # Feedforward MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, num_tokens, d_model)
        Returns:
            (B, num_tokens, d_model)
        """
        # Pre-LN Self-Attention with Residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask
        )
        x = x + self.drop1(attn_out)

        # Pre-LN FFN with Residual
        x = x + self.ffn(self.norm2(x))
        return x


class LatentBottleneckPooling(nn.Module):
    """
    Compresses a sequence of biomarker tokens (B, N, d_model) into a unified
    patient representation vector (B, d_model) using a learnable latent query.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # Learnable clinical summary token
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_tokens = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, d_model)
        Returns:
            patient_rep: (B, d_model)
        """
        B = tokens.shape[0]
        q = self.query_token.expand(B, -1, -1)  # (B, 1, d_model)

        norm_q = self.norm_query(q)
        norm_kv = self.norm_tokens(tokens)

        # Cross-attention: query pools from tokens
        out, _ = self.cross_attn(query=norm_q, key=norm_kv, value=norm_kv)
        out = self.proj(out.squeeze(1))  # (B, d_model)
        return out
