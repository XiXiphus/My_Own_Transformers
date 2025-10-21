import argparse


def get_args():
    """
    Parse command line arguments.
    Returns:
    - Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Transformer model training arguments")

    # Model parameters
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model dimension",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of encoder and decoder layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Learning rate scaling factor",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval in batches",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint file path to resume training",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="Only save checkpoints that improve validation BLEU",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Maximum number of batches to evaluate (None = all)",
    )

    # Data parameters
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Training data file path",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="Validation data file path",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--use_demo_data",
        action="store_true",
        help="Use demo dataset",
    )

    return parser.parse_args()
