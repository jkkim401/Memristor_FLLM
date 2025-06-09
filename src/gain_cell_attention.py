import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Assuming quantization.py is in the same directory or accessible in the Python path
from .quantization import BitLinear, BitActivationQuant, quantize_activations_8bit

class GainCellAttention(nn.Module):
    """Simulated Gain-Cell Attention Layer with BitNet Quantization."""
    def __init__(self, config):
        """
        Initializes the GainCellAttention module.

        Args:
            config: A configuration object containing model parameters like:
                    n_embd (int): Embedding dimension.
                    n_head (int): Number of attention heads.
                    bias (bool): Whether to use bias in linear layers.
                    dropout (float): Dropout probability.
                    # Add any gain-cell specific parameters if needed (e.g., noise levels for simulation)
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Use BitLinear for Key, Query, Value projections as per BitNet integration
        # W1.58A8 weights, potentially A8 activations (handled by BitActivationQuant or wrapper)
        self.key = BitLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = BitLinear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = BitLinear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection layer - also BitLinear
        self.proj = BitLinear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layer
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Activation quantization layer (if activations need to be quantized before/after attention)
        # Placement depends on the exact BitNet architecture variant (W1.58A8 implies 8-bit activations)
        self.act_quant = BitActivationQuant() # Placeholder for 8-bit activation quantization

        # Causal mask (for decoder architectures)
        # Using register_buffer for non-parameter tensors that should be part of the model's state
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        # Note: Masking logic might need adjustment based on how gain-cell handles sequences.
        # The paper focuses on the dot-product computation aspect.

    def forward(self, x):
        """
        Forward pass for Gain-Cell Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd).
        """
        B, T, C = x.size() # Batch size, sequence length, embedding dimension

        # --- Input Quantization (if applicable) ---
        # Apply 8-bit quantization to the input tensor if required by the architecture variant
        x = self.act_quant(x)

        # --- Project Key, Query, Value --- 
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Input x: (B, T, C)
        # Output k, q, v: (B, n_head, T, head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- Activation Quantization (Post-Projection) ---
        # Quantize K, Q, V if activations are quantized after linear layers
        k = self.act_quant(k)
        q = self.act_quant(q)
        v = self.act_quant(v)

        # --- Simulated Analog Dot-Product (Gain-Cell Behavior) ---
        # Standard attention: att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Gain-cell simulation: This is where the core analog computation happens.
        # For simulation purposes, we can perform the standard dot product,
        # potentially adding noise or applying constraints if modeling hardware effects.
        # The key aspect is that the K, Q, V projections used BitLinear (W1.58).
        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask (if decoder)
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))

        # Softmax
        att_weights = F.softmax(att_scores, dim=-1)

        # --- Attention Weight Quantization (if applicable) ---
        # Some low-bit architectures might quantize attention weights too.
        # Assuming 8-bit quantization for consistency with activations.
        att_weights = self.act_quant(att_weights)

        # Dropout on attention weights
        att_weights = self.attn_dropout(att_weights)

        # --- Weighted Sum of Values --- 
        # Weighted aggregation of values
        # Output y: (B, n_head, T, head_dim)
        y = att_weights @ v

        # --- Output Quantization (Before Projection) ---
        y = self.act_quant(y)

        # Re-assemble all head outputs side by side
        # y shape: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- Output Projection --- 
        # Apply final projection layer (BitLinear)
        y = self.proj(y)

        # --- Final Activation Quantization & Dropout ---
        y = self.act_quant(y)
        y = self.proj_dropout(y)

        return y

class SlidingWindowAttention(GainCellAttention):
    def __init__(self, config, window_size: int = 64, stride: int = 32):
        super().__init__(config)
        assert window_size <= config.block_size, "window size must be < block size"
        self.window_size = window_size
        self.stride = stride

    def forward(self, x):
        B, T, C = x.shape
        device= x.device
        
        out = torch.zeros((T,), device=device)
        
        for start in range(0, T, self.stride):
            end = min(start + self.window_size, T)
            x_win = x[:, start:end, :]
            
            q = self.q.proj(x_win) # Batch size, sequence length, embedding dimension
            k = self.k.proj(x_win)
            v = self.v.proj(x_win)
            
            q = BitActivationQuant()(q)
            k = BitActivationQuant()(k)
            v = BitActivationQuant()(v)
            
            att_scores = (q @ k.transpose(-2, -1)) * self.scale
            att_weights = F.softmax(att_scores, dim=-1)
            att_weights = self.attn_dropout(att_weights)

            win_out = att_weights @ v
            win_out = self.out_proj(win_out)

            out[:, start:end] += win_out
            coverage[start:end] += 1
            
        coverage = coverage.clamp(min=1.0).unsqueeze(0).unsqueeze(-1)    
        out = out / coverage
        return out
# Example Configuration (replace with actual config)
class BitnetGainCellConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    attention: str = "SlidingWindowAttention"
    window_size: int = 64
    stride: int = 32
    
    # block_size = 64 # Sequence length for mask, if needed

# Example Usage (Conceptual)
if __name__ == '__main__':
    config = BitnetGainCellConfig()
    gain_cell_attn = SlidingWindowAttention(config)
    print(gain_cell_attn)

    # Example input tensor
    batch_size = 4
    seq_len = 16
    input_tensor = torch.randn(batch_size, seq_len, config.n_embd)

    # Forward pass
    output = gain_cell_attn(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)

