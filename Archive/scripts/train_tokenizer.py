#!/usr/bin/env python3
"""Script to train SentencePiece tokenizer."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LumiraTokenizer
from src.data.dataset import BilingualTextDataset


def main():
    parser = argparse.ArgumentParser(description="Train Lumira tokenizer")
    parser.add_argument("--data", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="data/vocab/lumira")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare training data
    print("Preparing training data...")
    dataset = BilingualTextDataset(args.data)
    temp_file = output_path.parent / "spm_train.txt"
    dataset.save_for_spm(temp_file)
    print(f"Prepared {len(dataset)} text samples")

    # Train tokenizer
    print(f"\nTraining tokenizer (vocab_size={args.vocab_size})...")
    tokenizer = LumiraTokenizer()
    tokenizer.train(
        input_files=[temp_file],
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    # Save vocabulary for inspection
    tokenizer.save_vocab(output_path.parent / "vocab.json")

    # Test tokenizer
    print("\nTokenizer test:")
    test_sentences = [
        "こんにちは",
        "Sola!",
        "私はあなたを愛しています",
        "Mi ama tu",
    ]
    for sent in test_sentences:
        tokens = tokenizer.tokenize(sent)
        ids = tokenizer.encode(sent)
        print(f"  {sent}")
        print(f"    Tokens: {tokens}")
        print(f"    IDs: {ids}")
        print()

    # Clean up temp file
    temp_file.unlink()

    print(f"Tokenizer saved to: {args.output}.model")
    print(f"Vocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
