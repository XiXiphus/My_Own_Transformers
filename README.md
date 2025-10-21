# My Own Transformers

PyTorch re-implementation of the original Transformer architecture from *Attention Is All You Need*. The codebase stays close to the paper’s specification—stacked multi-head attention, position-wise feed-forward layers, sinusoidal positional encodings, residual connections, and the warmup-based learning rate schedule.

Use it as a concise reference implementation, a starting point for experiments, or an educational resource to understand how the model is wired end-to-end.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d549173-450a-484a-af29-47152805800d" width="55%">
</div>

If you read Chinese, you can find additional notes here: [Transformer不完全指北](https://zhuanlan.zhihu.com/p/1901912964216358105).

---

## Contents

- [My Own Transformers](#my-own-transformers)
  - [Contents](#contents)
  - [Highlights](#highlights)
  - [Project Layout](#project-layout)
  - [Setup](#setup)
  - [Running Training](#running-training)
    - [Built-in demo dataset](#built-in-demo-dataset)
    - [Custom dataset](#custom-dataset)
    - [Resume training](#resume-training)
  - [Configuration](#configuration)
  - [Data Preparation](#data-preparation)
  - [Implementation Notes](#implementation-notes)
  - [Development Tips](#development-tips)
  - [References](#references)

---

## Highlights

- Canonical encoder–decoder Transformer with 6 layers × 8 heads (configurable)
- Inverse-sqrt warmup learning rate schedule via `TransformerOptimizer`
- `tiktoken` tokenizer shared by encoder/decoder with BOS/EOS/PAD handling baked in
- Data loader automatically prepares `(src, tgt_input, tgt_output)` tensors for teacher forcing
- Character-level BLEU evaluation for train/val/test splits
- Checkpoint management (model + optimizer state + metrics) and optional early stopping
- Demo dataset bundled for quick sanity checks; easily swap in your own parallel corpus

---

## Project Layout

```
src/
├── args.py         # CLI flags definition
├── config.py       # Config wrapper around parsed args
├── data.py         # Dataset, padding, shifting, dataloader helpers
├── main.py         # Train/eval entry point
├── models.py       # Encoder, decoder, attention blocks, FFNs, positional encoding
├── optimizer.py    # Warmup-based optimizer wrapper
└── utils.py        # Masks, tokenizer cache, BLEU score, demo data, helpers
```

Dependencies: `torch`, `tiktoken`, `tqdm` (see `requirements.txt`).

---

## Setup

```bash
# Replace with your actual repository URL
git clone https://github.com/XiXiphus/My_Own_Transformers
cd My_Own_Transformers

# optional virtualenv
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Running Training

### Built-in demo dataset

```bash
python src/main.py --use_demo_data --device cpu
```

Runs a short training session on the toy English↔Chinese corpus and prints per-epoch BLEU scores.

### Custom dataset

```bash
python src/main.py \
    --train_file data/train.tsv \
    --val_file data/val.tsv \
    --batch_size 64 \
    --num_epochs 20 \
    --save_best_only \
    --save_dir checkpoints/exp1
```

### Resume training

```bash
python src/main.py --resume checkpoints/exp1/checkpoint_epoch_10.pt
```

Each epoch: teacher-forced training, BLEU evaluation on train/val, optional early stopping, and checkpoint saving (`epoch`, metrics, weights, optimizer state).

If you run the demo workflow, the script evaluates on the held-out test set at the end.

---

## Configuration

All CLI arguments live in `src/args.py`. Common options:

| Argument                      | Description                                             | Default |
| ----------------------------- | ------------------------------------------------------- | ------- |
| `--d_model`                   | Embedding/hidden size                                   | 512     |
| `--d_ff`                      | Feed-forward dimension                                  | 2048    |
| `--num_heads`                 | Multi-head attention heads                              | 8       |
| `--num_layers`                | Encoder & decoder layers                                | 6       |
| `--dropout`                   | Dropout rate                                            | 0.1     |
| `--batch_size`                | Minibatch size                                          | 32      |
| `--num_epochs`                | Training epochs                                         | 10      |
| `--learning_rate`             | Scale factor applied to scheduler                       | 1.0     |
| `--warmup_steps`              | Warmup steps before inverse-sqrt decay                  | 4000    |
| `--device`                    | `cuda`/`cpu` (auto-fallback to CPU if CUDA unavailable) | `cuda`  |
| `--train_file` / `--val_file` | Tab-separated corpus paths                              | `None`  |
| `--max_seq_len`               | Maximum tokenized length (includes BOS/EOS)             | 512     |
| `--use_demo_data`             | Toggle demo dataset                                     | `False` |
| `--save_best_only`            | Persist checkpoints that improve val BLEU               | `False` |
| `--patience`                  | Early-stop patience (epochs)                            | 5       |
| `--max_eval_batches`          | Max batches to evaluate (None = all)                    | `None`  |

Values are exposed via the global `config = Config()` instance in `src/config.py`.

---

## Data Preparation

Provide a UTF-8 TSV with `<source>	<target>` per line. Example:

```
Learning is the best reward.	学习是旅途的意义。
Knowledge is power.	知识就是力量。
Practice makes perfect.	熟能生巧。
```

`TranslationDataset` handles:

- Tokenization with `tiktoken` (`cl100k_base`)
- Injecting `BOS/EOS`, truncating to `max_seq_len`
- Padding batches via `torch.nn.utils.rnn.pad_sequence`
- Producing `src`, `tgt_input` (left-shifted) and `tgt_output` (right-shifted)

Swap in your own tokenizer by editing `utils.get_tokenizer` / `encode_text`.

---

## Implementation Notes

- **Positional encoding**: sinusoidal tables registered as buffers (length 5000)
- **Masking**: `create_masks` combines padding and look-ahead masks compatible with multi-head attention
- **Residual + LayerNorm**: dropout before residual, post-residual layer norm (paper’s layout)
- **Optimizer**: wraps Adam (`betas=(0.9, 0.98)`, `eps=1e-9`), stores `step_num`, exposes `state_dict`
- **BLEU**: character-level metric for language-agnostic evaluation
- **Device convenience**: `Transformer.device` property keeps tensors on the right accelerator/CPU

---

## Development Tips

- `python -m compileall src` catches syntax errors quickly (used in health checks)
- Add unit tests around masking/data shapes when extending the model (e.g., beam search)
- Set `torch.manual_seed(...)` before constructing datasets/model for reproducibility
- Adjust `--log_interval` or pipe stdout to your logger for longer runs

---

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Harvard NLP annotated Transformer walkthrough: <https://nlp.seas.harvard.edu/2018/04/03/attention.html>
- `tiktoken` tokenizer: <https://github.com/openai/tiktoken>

If this project helps you, consider starring the repo or citing the paper. Happy translating!
