import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.utils.checkpoint as checkpoint

# Assuming quantization.py and gain_cell_attention.py are accessible
from .quantization import BitLinear, BitActivationQuant
from .gain_cell_attention import GainCellAttention, SlidingWindowAttention

# --- Layer Normalization ---
# Standard LayerNorm is used here. BitNet papers sometimes mention specific RMSNorm or other variants.
# For simplicity, we start with standard LayerNorm. Quantizing LayerNorm is also possible but adds complexity.
class LayerNorm(nn.Module):
    """최적화된 LayerNorm"""
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# --- MLP ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = BitLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = BitLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# --- Transformer Block --- 
class Block(nn.Module):
    """ Transformer block combining Gain-Cell Attention and BitLinear MLP."""

    def __init__(self, config: BitNetGainCellConfig) -> None:
        super().__init__()
        self.config = config
        # Layer Normalization before attention
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        # Gain-Cell Attention layer (슬라이딩 윈도우 어텐션 적용)
        self.attn = SlidingWindowAttention(config, window_size=config.window_size, stride=config.stride)
        # Layer Normalization before MLP
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        # MLP using BitLinear layers
        self.mlp = MLP(config)
        self.use_checkpoint = config.use_checkpoint

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass for the Transformer Block."""
        if self.use_checkpoint and self.training:
            x = x + checkpoint.checkpoint(self.attn, self.ln_1(x), mask)
            x = x + checkpoint.checkpoint(self.mlp, self.ln_2(x))
        else:
            x = x + self.attn(self.ln_1(x), mask)
            x = x + self.mlp(self.ln_2(x))
        return x

# --- Model Configuration --- 
@dataclass
class BitNetGainCellConfig:
    block_size: int = 4096 # Max sequence length
    vocab_size: int = 128256
    n_layer: int = 30
    n_head: int = 20
    n_embd: int = 2560
    dropout: float = 0.0
    bias: bool = True
    window_size: int = 64
    stride: int = 32
    use_flash_attention: bool = True
    use_checkpoint: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

# --- Full Model --- 
class BitNetGainCellLLM(nn.Module):
    """ The full BitNet-based LLM with Gain-Cell Attention."""

    def __init__(self, config: BitNetGainCellConfig) -> None:
        super().__init__()
        self.config = config
        assert config.vocab_size is not None
        assert config.block_size is not None

        # Token and Position Embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout for embeddings
            drop = nn.Dropout(config.dropout),
            # Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final Layer Normalization
            ln_f = LayerNorm(config.n_embd, config.bias),
        ))
        # Final linear layer (Language Model Head)
        # Note: The final layer might or might not be quantized depending on the specific BitNet variant.
        # Using standard nn.Linear for now, but could be replaced with BitLinear if needed.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between token embedding and final linear layer
        self.transformer.wte.weight = self.lm_head.weight

        # Optional: Activation quantization for embeddings/final output
        self.emb_act_quant = BitActivationQuant()
        self.head_act_quant = BitActivationQuant()

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("mlp.3.weight"): # Second linear layer in MLP
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()/1e6:.2f} M")

        # 혼합 정밀도 훈련 설정
        self.mixed_precision = config.mixed_precision
        if self.mixed_precision:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

    def get_num_params(self, non_embedding=True):
        """ Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude embedding parameters if requested (often done for reporting core model size)
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ Initialize weights."""
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the LLM.

        Args:
            idx (torch.Tensor): Input sequence of token indices (B, T).
            targets (torch.Tensor, optional): Target sequence of token indices (B, T). If provided, computes loss.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Logits and Loss (if targets provided).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"입력 시퀀스 길이 {t}가 모델의 최대 컨텍스트 길이 {self.config.block_size}를 초과합니다."
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Get token and position embeddings
        with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
            tok_emb = self.transformer.wte(idx) # shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # shape (1, t, n_embd)
            
            # Quantize embeddings if needed
            x = self.emb_act_quant(tok_emb + pos_emb)
            x = self.transformer.drop(x)
            
            # Pass through transformer blocks
            for block in self.transformer.h:
                x = block(x, mask)
            
            # Final layer normalization
            x = self.transformer.ln_f(x)

            # Quantize output of final layer norm
            x = self.head_act_quant(x)

            if targets is not None:
                # If we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # Inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """자동 회귀적 텍스트 생성"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Example Usage (Conceptual)
if __name__ == '__main__':
    config = BitNetGainCellConfig(
        block_size=4096,
        vocab_size=128256,
        n_layer=30,
        n_head=20,
        n_embd=2560,
        dropout=0.0,
        bias=True,
        window_size=64,
        stride=32
    )
    model = BitNetGainCellLLM(config)
    print(model)

    # Example input
    batch_size = 4
    seq_len = config.block_size
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print("Input indices shape:", idx.shape)

    # Forward pass (training mode)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    print("Logits shape (training):", logits.shape)
    print("Loss:", loss.item() if loss is not None else "N/A")

    # Forward pass (inference mode)
    logits_inf, _ = model(idx)
    print("Logits shape (inference):", logits_inf.shape)

