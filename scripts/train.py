#!/usr/bin/env python3
"""Script to train Lumira Transformer."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Lumira Transformer")
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--val-data", default="data/processed/val.jsonl")
    parser.add_argument("--tokenizer", default="data/vocab/lumira.model")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--model-config", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = TrainingConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        tokenizer_model=args.tokenizer,
        output_dir=args.output_dir,
        model_config=args.model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
        device=args.device,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
