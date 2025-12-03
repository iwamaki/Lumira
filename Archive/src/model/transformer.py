"""Lumira Transformer - Encoder-Decoder architecture for translation."""

import torch
import torch.nn as nn

from .config import ModelConfig
from .encoder import Encoder
from .decoder import Decoder
from .attention import PositionalEncoding


class LumiraTransformer(nn.Module):
    """Encoder-Decoder Transformer for Japanese <-> Lumira translation.

    This model follows the original Transformer architecture with some
    modern improvements (pre-norm, GELU activation).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings (shared or separate based on task)
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout
        )

        # Encoder and Decoder
        self.encoder = Encoder(
            n_layers=config.n_encoder_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

        self.decoder = Decoder(
            n_layers=config.n_decoder_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

        # Output projection (can tie weights with tgt_embedding)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence.

        Args:
            src: Source tokens [batch, src_len]

        Returns:
            Padding mask [batch, 1, src_len]
        """
        return (src != self.config.pad_token_id).unsqueeze(1)

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create causal mask for target sequence.

        Args:
            tgt: Target tokens [batch, tgt_len]

        Returns:
            Combined padding and causal mask [batch, tgt_len, tgt_len]
        """
        batch_size, tgt_len = tgt.size()
        device = tgt.device

        # Padding mask
        padding_mask = (tgt != self.config.pad_token_id).unsqueeze(1)  # [batch, 1, tgt_len]

        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0)  # [1, tgt_len, tgt_len]

        # Combine masks
        combined_mask = padding_mask & causal_mask

        return combined_mask

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: Source tokens [batch, src_len]
            src_mask: Source padding mask

        Returns:
            Encoder output [batch, src_len, d_model]
        """
        src_embedded = self.positional_encoding(self.src_embedding(src))
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target sequence.

        Args:
            tgt: Target tokens [batch, tgt_len]
            encoder_output: Encoder output [batch, src_len, d_model]
            tgt_mask: Target causal mask
            src_mask: Source padding mask

        Returns:
            Decoder output [batch, tgt_len, d_model]
        """
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))
        return self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            src: Source tokens [batch, src_len]
            tgt: Target tokens [batch, tgt_len]
            src_mask: Source padding mask (optional, will be created if None)
            tgt_mask: Target causal mask (optional, will be created if None)

        Returns:
            Logits [batch, tgt_len, vocab_size]
        """
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)

        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)

        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate translation autoregressively.

        Args:
            src: Source tokens [batch, src_len]
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated tokens [batch, gen_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode source
        src_mask = self.create_src_mask(src)
        encoder_output = self.encode(src, src_mask)

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device
        )

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = self.create_tgt_mask(generated)
            decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)

            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])

            # Apply temperature
            logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if temperature > 0:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            done = done | (next_token.squeeze(-1) == self.config.eos_token_id)
            if done.all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
