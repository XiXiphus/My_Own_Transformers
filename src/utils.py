import torch
import tiktoken
from typing import Tuple, List, Optional
import collections
import math
import os

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SPECIAL_TOKEN_COUNT = 3
TOKEN_OFFSET = SPECIAL_TOKEN_COUNT


_TOKENIZER: Optional[tiktoken.Encoding] = None


def _get_or_create_tokenizer() -> tiktoken.Encoding:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask.
    Parameters:
    - seq: Input sequence
    Returns:
    - Padding mask
    """
    return (seq != PAD_ID).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    Create look-ahead mask.
    Parameters:
    - size: Sequence length
    Returns:
    - Look-ahead mask
    """
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
    return mask


def create_tgt_mask(tgt):
    """Create target sequence mask (combining look-ahead mask and padding mask)."""
    seq_len = tgt.size(1)
    look_ahead = create_look_ahead_mask(seq_len).to(tgt.device)
    look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)

    padding_mask = create_padding_mask(tgt)
    padding_mask = padding_mask.expand(-1, -1, seq_len, -1)

    return (~look_ahead) & padding_mask


def create_masks(src, tgt):
    """
    Create source and target sequence masks.
    Parameters:
    - src: Source sequence with shape [batch_size, seq_len]
    - tgt: Target sequence with shape [batch_size, seq_len]
    Returns:
    - src_mask: Source sequence mask with shape [batch_size, 1, 1, seq_len]
    - tgt_mask: Target sequence mask with shape [batch_size, 1, seq_len, seq_len]
    """
    src_mask = create_padding_mask(src)
    tgt_mask = create_tgt_mask(tgt)
    return src_mask, tgt_mask


def get_tokenizer():
    return _get_or_create_tokenizer()


def encode_text(
    text: str, tokenizer, max_seq_len: Optional[int] = None
) -> torch.Tensor:
    content_tokens = [token + TOKEN_OFFSET for token in tokenizer.encode(text)]
    tokens = [BOS_ID] + content_tokens + [EOS_ID]
    if max_seq_len is not None and max_seq_len > 0:
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            if tokens[-1] != EOS_ID:
                tokens[-1] = EOS_ID
    return torch.tensor(tokens, dtype=torch.long)


def decode_text(tokens: torch.Tensor, tokenizer) -> str:
    if tokens.dim() > 1:
        tokens = tokens.squeeze(0)
    tokens = tokens.cpu().numpy().tolist()
    if EOS_ID in tokens:
        tokens = tokens[: tokens.index(EOS_ID)]
    tokens = [t for t in tokens if t not in [PAD_ID, BOS_ID]]
    content_tokens = [t - TOKEN_OFFSET for t in tokens if t >= TOKEN_OFFSET]
    return tokenizer.decode(content_tokens)


def calculate_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score.
    Parameters:
    - reference: Reference translation
    - hypothesis: Model-generated translation
    - max_n: Maximum n-gram length
    Returns:
    - BLEU score in range [0, 1]
    """
    # Tokenize text into character list (friendly for Chinese)
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    # If generated translation is empty, return 0
    if len(hyp_tokens) == 0:
        return 0.0

    # Calculate n-gram precision
    precisions = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        # Calculate n-grams in reference translation
        ref_ngrams = collections.Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i : i + n])
            ref_ngrams[ngram] += 1

        # Calculate n-grams in generated translation
        hyp_ngrams = collections.Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i : i + n])
            hyp_ngrams[ngram] += 1

        # Calculate number of matching n-grams
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        # Calculate precision
        precision = matches / max(1, len(hyp_tokens) - n + 1)
        precisions.append(precision)

    # Calculate brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        brevity_penalty = 1.0

    # Calculate BLEU score
    if any(p > 0 for p in precisions):
        s = math.log(brevity_penalty)
        s += sum(math.log(p) if p > 0 else float("-inf") for p in precisions) / len(
            precisions
        )
        bleu = math.exp(s)
    else:
        bleu = 0.0

    return bleu


def evaluate_translations(references: List[str], hypotheses: List[str]) -> float:
    """
    Evaluate BLEU score for a set of translations.
    Parameters:
    - references: List of reference translations
    - hypotheses: List of model-generated translations
    Returns:
    - Average BLEU score
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of reference and generated translations must be equal")

    total_bleu = 0.0
    for ref, hyp in zip(references, hypotheses):
        total_bleu += calculate_bleu(ref, hyp)

    return total_bleu / len(references)


def get_demo_data(
    tokenizer: tiktoken.Encoding,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """
    Get demo training, validation, and test data.
    Parameters:
    - tokenizer: Tokenizer
    Returns:
    - Tuple of (train_data, val_data, test_data), each element is a (source_text, target_text) tuple
    """
    # Training data
    train_data = [
        ("Learning is the best reward.", "学习是旅途的意义。"),
        ("Knowledge is power.", "知识就是力量。"),
        ("Practice makes perfect.", "熟能生巧。"),
        ("Time is money.", "时间就是金钱。"),
        ("Where there is a will, there is a way.", "有志者事竟成。"),
        ("Actions speak louder than words.", "行动胜于言语。"),
        ("The early bird catches the worm.", "早起的鸟儿有虫吃。"),
        (
            "A journey of a thousand miles begins with a single step.",
            "千里之行，始于足下。",
        ),
        ("Failure is the mother of success.", "失败是成功之母。"),
        ("Rome was not built in a day.", "罗马不是一天建成的。"),
    ]

    # Validation data
    val_data = [
        ("All roads lead to Rome.", "条条大路通罗马。"),
        ("Better late than never.", "亡羊补牢，为时未晚。"),
        ("Every cloud has a silver lining.", "黑暗中总有一线光明。"),
        ("A friend in need is a friend indeed.", "患难见真情。"),
        ("Honesty is the best policy.", "诚实是最好的策略。"),
    ]

    # Test data
    test_data = [
        ("The grass is always greener on the other side.", "这山望着那山高。"),
        ("Don't put all your eggs in one basket.", "不要把所有鸡蛋放在一个篮子里。"),
        ("When in Rome, do as the Romans do.", "入乡随俗。"),
        ("A penny saved is a penny earned.", "省一分就是赚一分。"),
        ("Birds of a feather flock together.", "物以类聚，人以群分。"),
    ]

    return train_data, val_data, test_data


def save_data_to_file(data: List[Tuple[str, str]], file_path: str) -> None:
    """
    Save data to file.
    Parameters:
    - data: List of (source_text, target_text) tuples
    - file_path: File path
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for src, tgt in data:
            f.write(f"{src}\t{tgt}\n")
