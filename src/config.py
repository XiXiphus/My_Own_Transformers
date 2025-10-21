import torch
from args import get_args


class Config:
    def __init__(self):
        # Parse command line arguments
        args = get_args()

        # Model parameters
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # Training parameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.warmup_steps = args.warmup_steps
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.save_dir = args.save_dir
        self.log_interval = args.log_interval
        self.resume = args.resume
        self.save_best_only = args.save_best_only
        self.patience = args.patience
        self.max_eval_batches = args.max_eval_batches

        # Data parameters
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.max_seq_len = args.max_seq_len
        self.use_demo_data = args.use_demo_data

        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            self.device = "cpu"

    def __str__(self):
        """Return string representation of configuration"""
        config_str = "Model Configuration:\n"
        config_str += f"  d_model: {self.d_model}\n"
        config_str += f"  d_ff: {self.d_ff}\n"
        config_str += f"  num_heads: {self.num_heads}\n"
        config_str += f"  num_layers: {self.num_layers}\n"
        config_str += f"  dropout: {self.dropout}\n"
        config_str += "\nTraining Configuration:\n"
        config_str += f"  batch_size: {self.batch_size}\n"
        config_str += f"  num_epochs: {self.num_epochs}\n"
        config_str += f"  learning_rate: {self.learning_rate}\n"
        config_str += f"  warmup_steps: {self.warmup_steps}\n"
        config_str += f"  device: {self.device}\n"
        config_str += f"  save_dir: {self.save_dir}\n"
        config_str += f"  log_interval: {self.log_interval}\n"
        config_str += f"  resume: {self.resume}\n"
        config_str += f"  save_best_only: {self.save_best_only}\n"
        config_str += f"  patience: {self.patience}\n"
        config_str += f"  max_eval_batches: {self.max_eval_batches}\n"
        config_str += "\nData Configuration:\n"
        config_str += f"  train_file: {self.train_file}\n"
        config_str += f"  val_file: {self.val_file}\n"
        config_str += f"  max_seq_len: {self.max_seq_len}\n"
        config_str += f"  use_demo_data: {self.use_demo_data}\n"
        return config_str


# Create global configuration instance
config = Config()
