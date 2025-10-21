import os
import torch
from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence
from typing import List, Tuple
from utils import encode_text, get_tokenizer, PAD_ID


class TranslationDataset(torch.utils.data.Dataset):
    """
    Translation dataset class.
    Parameters:
    - data: List of (source_text, target_text) tuples
    - tokenizer: Tokenizer
    - max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for src_text, tgt_text in data:
            src_tokens = encode_text(src_text, tokenizer, max_seq_len)
            tgt_tokens = encode_text(tgt_text, tokenizer, max_seq_len)
            if src_tokens.numel() < 2 or tgt_tokens.numel() < 2:
                # In edge cases, skip samples with insufficient length (less than BOS/EOS)
                continue
            self.data.append((src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_sequence(sequences):
    """Pad sequences to the same length using PAD_ID."""
    return torch_pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)


def create_dataloader(
    data: List[Tuple[str, str]],
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create data loader.
    Parameters:
    - data: List of (source_text, target_text) tuples
    - batch_size: Batch size
    - max_seq_len: Maximum sequence length
    - shuffle: Whether to shuffle data
    Returns:
    - DataLoader instance
    """
    tokenizer = get_tokenizer()
    dataset = TranslationDataset(data, tokenizer, max_seq_len)

    def collate_fn(batch):
        src_batch = pad_sequence([item[0] for item in batch])
        tgt_batch = pad_sequence([item[1] for item in batch])

        if tgt_batch.size(1) < 2:
            raise ValueError(
                "Target sequence length must be at least 2 (including BOS and EOS)"
            )

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        return src_batch, tgt_input, tgt_output

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader


def load_data_from_file(
    file_path: str,
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Load data from file and create data loader.
    Parameters:
    - file_path: File path
    - batch_size: Batch size
    - max_seq_len: Maximum sequence length
    - shuffle: Whether to shuffle data
    Returns:
    - DataLoader instance
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                src, tgt = line.split("\t")
                data.append((src, tgt))
            except ValueError:
                print(f"Skipping invalid line: {line}")

    return create_dataloader(data, batch_size, max_seq_len, shuffle)
