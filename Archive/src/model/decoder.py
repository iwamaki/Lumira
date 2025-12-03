"""Transformer Decoder implementation."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .encoder import FeedForward


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            tgt_mask: Target causal mask
            src_mask: Source padding mask

        Returns:
            Output tensor [batch, tgt_len, d_model]
        """
        # Masked self-attention
        normed = self.norm1(x)
        self_attn_out = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout(self_attn_out)

        # Cross-attention
        normed = self.norm2(x)
        cross_attn_out = self.cross_attn(normed, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_out)

        # Feed-forward
        normed = self.norm3(x)
        ff_out = self.feed_forward(normed)
        x = x + self.dropout(ff_out)

        return x


class Decoder(nn.Module):
    """Transformer Decoder stack."""

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
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.

        Args:
            x: Decoder input [batch, tgt_len, d_model]
            encoder_output: Encoder output [batch, src_len, d_model]
            tgt_mask: Target causal mask
            src_mask: Source padding mask

        Returns:
            Decoded output [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return self.norm(x)
