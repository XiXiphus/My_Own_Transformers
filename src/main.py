import torch
import torch.nn as nn
import os
from tqdm import tqdm
from models import Encoder, Decoder
from config import config
from optimizer import get_optimizer
from utils import (
    get_tokenizer,
    decode_text,
    evaluate_translations,
    get_demo_data,
    BOS_ID,
    EOS_ID,
    create_masks,
    create_padding_mask,
    create_tgt_mask,
)
from data import create_dataloader, load_data_from_file


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

    @property
    def device(self):
        return next(self.parameters()).device

    def translate(self, src, tokenizer, max_length=None):
        """
        Translate a single sentence.
        Parameters:
        - src: Source sequence
        - tokenizer: Tokenizer
        - max_length: Maximum generation length (defaults to max_seq_len from config)
        Returns:
        - Translation result
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            bos_token = BOS_ID
            eos_token = EOS_ID

            decode_max_length = max_length or config.max_seq_len or 50
            decode_max_length = max(2, decode_max_length)

            # Only translate the first sentence (first in batch)
            single_src = src[0:1]

            # Move source sequence to device
            device = self.device
            single_src = single_src.to(device)

            # Create source sequence mask
            src_mask = create_padding_mask(single_src)

            # Encode source sequence
            enc_output = self.encoder(single_src, src_mask)

            # Initialize target sequence with BOS token
            tgt = torch.tensor([[bos_token]], dtype=torch.long, device=device)

            # Autoregressive generation
            for _ in range(decode_max_length - 1):
                tgt_mask = create_tgt_mask(tgt)

                # Decode
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

                # Get next token
                next_token = output[:, -1].argmax(dim=-1, keepdim=True)

                # Append new token to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)

                # Stop if EOS token is generated
                if next_token.item() == eos_token:
                    break

            # Decode generated sequence
            translation = decode_text(tgt[0], tokenizer)
        if was_training:
            self.train()
        return translation


def train_step(model, optimizer, criterion, src, tgt_input, tgt_output):
    """
    Execute a single training step.
    Parameters:
    - model: Transformer model
    - optimizer: Optimizer
    - criterion: Loss function
    - src: Source sequence
    - tgt_input: Decoder input sequence
    - tgt_output: Decoder output target
    Returns:
    - Loss value
    """
    # Forward pass
    src_mask, tgt_mask = create_masks(src, tgt_input)

    output = model(src, tgt_input, src_mask, tgt_mask)

    # Calculate loss
    output = output.view(-1, output.size(-1))
    tgt_output = tgt_output.view(-1)
    loss = criterion(output, tgt_output)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, tokenizer, eval_data, max_batches=None):
    """
    Evaluate model performance.
    Parameters:
    - model: Transformer model
    - tokenizer: Tokenizer
    - eval_data: Evaluation data
    - max_batches: Maximum number of batches to evaluate
    Returns:
    - Average BLEU score
    """
    if eval_data is None:
        return 0.0

    if max_batches is not None and max_batches <= 0:
        return 0.0

    was_training = model.training
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch_idx, (src, _, tgt_output) in enumerate(eval_data):
            if max_batches is not None and batch_idx >= max_batches:
                break

            src = src.to(model.device)
            batch_size = src.size(0)

            for sample_idx in range(batch_size):
                single_src = src[sample_idx : sample_idx + 1]
                translation = model.translate(single_src, tokenizer)
                reference = decode_text(tgt_output[sample_idx], tokenizer)

                references.append(reference)
                hypotheses.append(translation)

    # If no successful translations, return 0
    if len(references) == 0:
        return 0.0

    bleu_score = evaluate_translations(references, hypotheses)
    if was_training:
        model.train()

    return bleu_score


def train(model, optimizer, criterion, train_loader, val_loader, tokenizer):
    """
    Train the model.
    Parameters:
    - model: Transformer model
    - optimizer: Optimizer
    - criterion: Loss function
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - tokenizer: Tokenizer
    """
    # Create save directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize best BLEU score
    best_bleu = 0.0
    patience_counter = 0

    # Try initial evaluation to ensure everything works
    print("Initial evaluation...")
    try:
        # Evaluate on a small batch only
        initial_src, initial_tgt_input, initial_tgt_output = next(iter(train_loader))
        initial_src = initial_src[:1].to(model.device)
        initial_tgt_input = initial_tgt_input[:1].to(model.device)
        src_mask, tgt_mask = create_masks(initial_src, initial_tgt_input)

        # Try forward pass
        with torch.no_grad():
            output = model(initial_src, initial_tgt_input, src_mask, tgt_mask)
        print("Initial evaluation successful!")
    except Exception as e:
        print(f"Initial evaluation failed: {e}")
        raise e

    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        for i, (src, tgt_input, tgt_output) in enumerate(train_loader):
            src = src.to(model.device)
            tgt_input = tgt_input.to(model.device)
            tgt_output = tgt_output.to(model.device)

            loss = train_step(model, optimizer, criterion, src, tgt_input, tgt_output)
            total_loss += loss

            # Print training information
            if (i + 1) % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}"
                )
                total_loss = 0

        # Evaluate model
        print("Evaluating training set BLEU score...")
        train_bleu = evaluate(
            model, tokenizer, train_loader, max_batches=config.max_eval_batches
        )
        val_bleu = 0.0
        if val_loader is not None:
            print("Evaluating validation set BLEU score...")
            val_bleu = evaluate(
                model, tokenizer, val_loader, max_batches=config.max_eval_batches
            )
            print(
                f"Epoch {epoch + 1}/{config.num_epochs}, "
                f"Train BLEU: {train_bleu:.4f}, "
                f"Val BLEU: {val_bleu:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{config.num_epochs}, "
                f"Train BLEU: {train_bleu:.4f}"
            )

        # Save model
        if val_loader is None:
            improved = True
        else:
            improved = val_bleu > best_bleu

        if not config.save_best_only or improved:
            # Update best BLEU score
            if val_loader is not None:
                if val_bleu > best_bleu:
                    best_bleu = val_bleu
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_bleu": train_bleu,
                "val_bleu": val_bleu,
                "best_bleu": best_bleu,
            }
            torch.save(
                checkpoint,
                os.path.join(config.save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # Early stopping check (only enabled when validation set is available)
        if val_loader is not None and patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load checkpoint.
    Parameters:
    - model: Transformer model
    - optimizer: Optimizer
    - checkpoint_path: Checkpoint file path
    Returns:
    - Starting epoch
    - Best BLEU score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_bleu"]


def main():
    # Print configuration
    print(config)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Prepare data
    if config.use_demo_data or not config.train_file:
        print("Using demo data")
        train_data, val_data, test_data = get_demo_data(tokenizer)
        train_loader = create_dataloader(
            train_data, config.batch_size, config.max_seq_len
        )
        val_loader = create_dataloader(val_data, config.batch_size, config.max_seq_len)
        test_loader = create_dataloader(
            test_data, config.batch_size, config.max_seq_len
        )
    else:
        print(f"Loading training data from file: {config.train_file}")
        train_loader = load_data_from_file(
            config.train_file, config.batch_size, config.max_seq_len
        )
        val_loader = (
            load_data_from_file(config.val_file, config.batch_size, config.max_seq_len)
            if config.val_file
            else None
        )
        test_loader = None

    # Create model
    model = Transformer(vocab_size=tokenizer.n_vocab).to(config.device)

    # Create optimizer and loss function
    optimizer = get_optimizer(
        model=model,
        model_size=config.d_model,
        factor=config.learning_rate,
        warmup=config.warmup_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Load checkpoint if specified
    if config.resume:
        load_checkpoint(model, optimizer, config.resume)

    # Train model
    train(model, optimizer, criterion, train_loader, val_loader, tokenizer)

    # Evaluate on test set if using demo data
    if test_loader is not None:
        test_bleu = evaluate(
            model, tokenizer, test_loader, max_batches=config.max_eval_batches
        )
        print(f"Test BLEU: {test_bleu:.4f}")


if __name__ == "__main__":
    main()
