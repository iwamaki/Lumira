"""Dataset classes for translation training."""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import LumiraTokenizer


class TranslationDataset(Dataset):
    """Dataset for Japanese <-> Lumira translation pairs."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: LumiraTokenizer,
        max_len: int = 128,
        src_lang: str = 'ja',
        tgt_lang: str = 'lumira',
    ):
        """Initialize dataset.

        Args:
            data_path: Path to JSONL file with translation pairs
            tokenizer: Trained tokenizer
            max_len: Maximum sequence length
            src_lang: Source language key in data
            tgt_lang: Target language key in data
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.data = self._load_data(data_path)

    def _load_data(self, path: str | Path) -> List[dict]:
        """Load translation pairs from JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single translation pair.

        Returns:
            Tuple of (src_ids, tgt_ids) as tensors
        """
        item = self.data[idx]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        # Encode
        src_ids = self.tokenizer.encode(src_text, add_bos=True, add_eos=True)
        tgt_ids = self.tokenizer.encode(tgt_text, add_bos=True, add_eos=True)

        # Truncate if needed
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """Collate function for DataLoader.

    Pads sequences to same length within batch.

    Args:
        batch: List of (src_ids, tgt_ids) tuples
        pad_id: Padding token ID

    Returns:
        Dictionary with padded tensors and lengths
    """
    src_list, tgt_list = zip(*batch)

    # Get max lengths
    src_max_len = max(len(s) for s in src_list)
    tgt_max_len = max(len(t) for t in tgt_list)

    batch_size = len(batch)

    # Create padded tensors
    src_padded = torch.full((batch_size, src_max_len), pad_id, dtype=torch.long)
    tgt_padded = torch.full((batch_size, tgt_max_len), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': torch.tensor([len(s) for s in src_list]),
        'tgt_lengths': torch.tensor([len(t) for t in tgt_list]),
    }


class BilingualTextDataset(Dataset):
    """Simple dataset for tokenizer training.

    Yields raw text lines from both languages.
    """

    def __init__(self, data_path: str | Path, src_lang: str = 'ja', tgt_lang: str = 'lumira'):
        self.texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.texts.append(item[src_lang])
                    self.texts.append(item[tgt_lang])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def save_for_spm(self, output_path: str | Path):
        """Save texts to plain text file for SentencePiece training."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(text + '\n')
