"""BPE Tokenizer for Lumira language."""

import json
from pathlib import Path
from typing import List

import sentencepiece as spm


class LumiraTokenizer:
    """Tokenizer for Japanese <-> Lumira translation.

    Uses SentencePiece BPE model trained on both languages.
    """

    SPECIAL_TOKENS = {
        'pad': '<pad>',
        'bos': '<bos>',
        'eos': '<eos>',
        'unk': '<unk>',
    }

    def __init__(self, model_path: str | Path | None = None):
        """Initialize tokenizer.

        Args:
            model_path: Path to trained SentencePiece model (.model file)
        """
        self.sp = None
        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        input_files: List[str | Path],
        model_prefix: str | Path,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        model_type: str = 'bpe',
    ):
        """Train SentencePiece model.

        Args:
            input_files: List of text files for training
            model_prefix: Output model prefix (will create .model and .vocab)
            vocab_size: Target vocabulary size
            character_coverage: Character coverage (higher for Japanese)
            model_type: Model type ('bpe', 'unigram', 'char', 'word')
        """
        input_str = ','.join(str(f) for f in input_files)

        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece=self.SPECIAL_TOKENS['pad'],
            bos_piece=self.SPECIAL_TOKENS['bos'],
            eos_piece=self.SPECIAL_TOKENS['eos'],
            unk_piece=self.SPECIAL_TOKENS['unk'],
            # For Japanese support
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_digits=True,
            treat_whitespace_as_suffix=False,
        )

        self.load(f"{model_prefix}.model")

    def load(self, model_path: str | Path):
        """Load trained SentencePiece model.

        Args:
            model_path: Path to .model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end

        Returns:
            List of token IDs
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")

        ids = self.sp.encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special: Skip special tokens in output

        Returns:
            Decoded text
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")

        if skip_special:
            ids = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]

        return self.sp.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to subword pieces.

        Args:
            text: Input text

        Returns:
            List of subword tokens
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or train() first.")

        return self.sp.encode(text, out_type=str)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def save_vocab(self, path: str | Path):
        """Save vocabulary to JSON for inspection."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")

        vocab = {
            self.sp.id_to_piece(i): i
            for i in range(self.vocab_size)
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
