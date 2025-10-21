import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding module.
        Parameters:
        - d_model: Feature dimension (embedding dimension)
        - max_len: Maximum sequence length
        """
        super().__init__()

        # Initialize zero positional encoding matrix with shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension, shape becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as persistent buffer, no gradients needed
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass.
        Parameters:
        - x: Input tensor with shape [batch_size, seq_len, d_model]
        Returns:
        - Tensor with positional encoding added, same shape as input
        """
        # Get input sequence length
        seq_len = x.size(1)

        # Extract positional encoding matching input sequence length
        position_encoding = self.pe[:, :seq_len]

        # Add positional encoding to input tensor using broadcasting
        return x + position_encoding


# Multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Initialize multi-head attention module.
        Parameters:
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass.
        Parameters:
        - q: Query tensor with shape [batch_size, seq_len, d_model]
        - k: Key tensor with shape [batch_size, seq_len, d_model]
        - v: Value tensor with shape [batch_size, seq_len, d_model]
        - mask: Mask tensor with shape [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        device = q.device

        # Linear transformation and split heads [batch_size, num_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention scores [batch_size, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask
        if mask is not None:
            # Ensure mask is on the correct device
            mask = mask.to(device)

            # Handle mask dimensions - ensure mask has 4 dimensions
            if mask.dim() > 4:
                mask = mask.squeeze(2)

            # Expand mask to all heads
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.num_heads, -1, -1)

            # Apply mask to scores (masked positions get -inf before softmax)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Calculate attention weights [batch_size, num_heads, q_len, k_len]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Calculate output [batch_size, num_heads, q_len, d_k]
        output = torch.matmul(attn, v)

        # Rearrange dimensions and merge heads [batch_size, q_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output


# Position-wise feed-forward network module
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize position-wise feed-forward network module.
        Parameters:
        - d_model: Input feature dimension
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.
        Parameters:
        - x: Input tensor with shape [batch_size, seq_len, d_model]
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# Encoder layer module
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer module.
        Parameters:
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass.
        Parameters:
        - x: Input tensor with shape [batch_size, seq_len, d_model]
        - mask: Mask tensor
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        # Self-attention layer
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# Encoder module
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.1,
        max_len=5000,
        device="cpu",
    ):
        """
        Initialize encoder module.
        Parameters:
        - vocab_size: Vocabulary size
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - num_layers: Number of encoder layers
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        - max_len: Maximum sequence length
        - device: Device (e.g., "cpu" or "cuda")
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass.
        Parameters:
        - x: Input tensor with shape [batch_size, seq_len]
        - mask: Mask tensor
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        # Word embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# Decoder layer module
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer module.
        Parameters:
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        """
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass.
        Parameters:
        - x: Decoder input tensor with shape [batch_size, seq_len, d_model]
        - enc_output: Encoder output tensor with shape [batch_size, seq_len, d_model]
        - src_mask: Source sequence mask
        - tgt_mask: Target sequence mask
        Returns:
        - Output tensor with shape [batch_size, seq_len, d_model]
        """
        # Masked self-attention layer
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Encoder-decoder attention layer
        # src_mask shape: [batch_size, 1, 1, src_len]
        # This allows broadcasting across attention heads and query positions
        attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


# Decoder module
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.1,
        max_len=5000,
        device="cpu",
    ):
        """
        Initialize decoder module.
        Parameters:
        - vocab_size: Vocabulary size
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - num_layers: Number of decoder layers
        - d_ff: Feed-forward hidden layer dimension
        - dropout: Dropout probability
        - max_len: Maximum sequence length
        - device: Device (e.g., "cpu" or "cuda")
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass.
        Parameters:
        - x: Decoder input tensor with shape [batch_size, seq_len]
        - enc_output: Encoder output tensor with shape [batch_size, seq_len, d_model]
        - src_mask: Source sequence mask
        - tgt_mask: Target sequence mask
        Returns:
        - Output tensor with shape [batch_size, seq_len, vocab_size]
        """
        # Word embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through N decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        # Final layer normalization
        x = self.norm(x)

        # Linear transformation to vocabulary size
        return self.linear(x)
