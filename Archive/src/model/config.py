"""Model configuration for Lumira Transformer."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Lumira Transformer model.

    Default configuration yields ~60M parameters suitable for
    Google Colab T4 GPU training.
    """
    # Model architecture
    vocab_size: int = 8000
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 128

    # Regularization
    dropout: float = 0.1

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """Estimate total number of parameters."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model * 2  # src + tgt

        # Encoder layer params (per layer)
        # Self-attention: 4 * d_model^2 (Q, K, V, O projections)
        # FFN: 2 * d_model * d_ff
        # LayerNorm: 2 * 2 * d_model
        encoder_layer = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff + 4 * self.d_model
        encoder_params = encoder_layer * self.n_encoder_layers

        # Decoder layer params (per layer)
        # Self-attention + Cross-attention + FFN
        decoder_layer = 8 * self.d_model**2 + 2 * self.d_model * self.d_ff + 6 * self.d_model
        decoder_params = decoder_layer * self.n_decoder_layers

        # Output projection
        output_params = self.d_model * self.vocab_size

        total = embed_params + encoder_params + decoder_params + output_params
        return total


# Preset configurations
TINY_CONFIG = ModelConfig(
    vocab_size=8000,
    d_model=256,
    n_heads=4,
    n_encoder_layers=4,
    n_decoder_layers=4,
    d_ff=1024,
)

SMALL_CONFIG = ModelConfig(
    vocab_size=8000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
)

BASE_CONFIG = ModelConfig(
    vocab_size=8000,
    d_model=768,
    n_heads=12,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=3072,
)
