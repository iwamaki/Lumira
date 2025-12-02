"""Training configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training the Lumira Transformer.

    Default values are optimized for Google Colab T4 GPU.
    """
    # Data
    train_data: str = "data/processed/train.jsonl"
    val_data: str = "data/processed/val.jsonl"
    tokenizer_model: str = "data/vocab/lumira.model"

    # Model (see ModelConfig for architecture)
    model_config: str = "small"  # tiny, small, base

    # Training
    epochs: int = 20
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    max_seq_len: int = 128

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # linear, cosine, constant

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Checkpointing
    output_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100

    # Resume training
    resume_from: str | None = None

    # Mixed precision
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    def __post_init__(self):
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ColabConfig(TrainingConfig):
    """Optimized config for Google Colab free tier (T4 GPU, 12GB VRAM)."""
    batch_size: int = 24
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    save_every: int = 500
    eval_every: int = 250
