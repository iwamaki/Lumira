"""Transformer Encoder implementation."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            src_mask: Source padding mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection (pre-norm)
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, normed, normed, src_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection (pre-norm)
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x


class Encoder(nn.Module):
    """Transformer Encoder stack."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            src_mask: Source padding mask

        Returns:
            Encoded output [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)
