#!/usr/bin/env python3
"""Script to generate Lumira training data.

Can be run standalone without installing dependencies.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for direct import (avoiding __init__.py dependencies)
src_path = Path(__file__).parent.parent / "src" / "data"
sys.path.insert(0, str(src_path.parent.parent))

# Import generate module directly to avoid tokenizer dependency
import importlib.util
spec = importlib.util.spec_from_file_location("generate", src_path / "generate.py")
generate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_module)
VocabularyExpander = generate_module.VocabularyExpander
SentenceGenerator = generate_module.SentenceGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Lumira training data")
    parser.add_argument(
        "--vocab-input",
        default="docs/vocabulary.json",
        help="Path to existing vocabulary JSON",
    )
    parser.add_argument(
        "--vocab-output",
        default="data/vocab/vocabulary_expanded.json",
        help="Path to save expanded vocabulary",
    )
    parser.add_argument(
        "--data-output",
        default="data/processed",
        help="Directory to save training data",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=100000,
        help="Number of translation pairs to generate",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of training data (rest is validation)",
    )
    args = parser.parse_args()

    # Create output directories
    Path(args.vocab_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.data_output).mkdir(parents=True, exist_ok=True)

    # Step 1: Expand vocabulary
    print("=" * 50)
    print("Step 1: Expanding vocabulary")
    print("=" * 50)

    expander = VocabularyExpander(args.vocab_input)
    vocab = expander.expand_vocabulary(args.vocab_size)
    expander.save_vocabulary(vocab, args.vocab_output)
    print(f"Vocabulary expanded: {len(vocab)} words")
    print(f"Saved to: {args.vocab_output}")

    # Step 2: Generate translation pairs
    print("\n" + "=" * 50)
    print("Step 2: Generating translation pairs")
    print("=" * 50)

    generator = SentenceGenerator(vocab)
    pairs = generator.generate_dataset(args.data_size)
    generator.save_dataset(pairs, args.data_output, args.train_ratio)

    # Show some examples
    print("\n" + "=" * 50)
    print("Sample pairs:")
    print("=" * 50)
    import random
    for pair in random.sample(pairs, min(5, len(pairs))):
        print(f"  JA: {pair['ja']}")
        print(f"  LU: {pair['lumira']}")
        print()

    print("Data generation complete!")


if __name__ == "__main__":
    main()
