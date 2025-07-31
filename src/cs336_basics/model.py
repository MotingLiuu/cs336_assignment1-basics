import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange, einsum
from math import sqrt
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional
from collections.abc import Iterable, Callable
import logging
import math


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.infeatures = in_features
        self.outfeatures = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), device=device, dtype=dtype)
        self.reset_parameter()
            
    def reset_parameter(self):
        sigma = sqrt(2.0 / (self.infeatures + self.outfeatures))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        if hasattr(self, "bias"):
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(self.weight, x, "dout din, ... din -> ... dout")
        if hasattr(self, "bias"):
            output += self.bias
        return output
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    
    def reset_parameters(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x.pow(2)
        mean_of_squares = x_squared.mean(dim=-1, keepdim=True)
        rms_val = torch.sqrt(mean_of_squares + self.eps)
        result = (x / rms_val) * self.weight
        return result.to(in_dtype)
    

class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, inner_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtye = dtype
        self.infeatures = in_features
        self.outfeatures = out_features
        self.inner_features = inner_features
        self.w1 = Linear(in_features, inner_features, bias=False, device=device, dtype=dtype)
        self.w2 = Linear(inner_features, out_features, bias=False, device=device, dtype=dtype)
        self.w3 = Linear(in_features, inner_features, bias=False, device=device, dtype=dtype)

    def register_parameter(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.w3(x)
        x_1 = self.w1(x)
        return self.w2(SiLU(x_1) * x_3)
    
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, buffer = True ,device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.buffer = buffer
        if buffer:
            self.get_buffer()
    
    def get_buffer(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device, dtype=self.dtype) / self.d_k))
        position_tensor = torch.arange(0, self.max_seq_len, device=self.device, dtype=self.dtype)
        theta_cached = einsum(position_tensor, inv_freq, "max_len, d -> max_len d")
        cos_cached = torch.cos(theta_cached)
        sin_cached = torch.sin(theta_cached)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
    
    def forward(self, x: Float[Tensor, "... seq_len d_k"]) -> Float[Tensor, "... seq_len d_k"]:
        seq_len = x.shape[-2] # the shape of x is (..., length, d_k), usually b, length, d_k
        if self.buffer:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device, dtype=self.dtype) / self.d_k))
            position_tensor = torch.arange(0, seq_len, device=self.device, dtype=self.dtype)
            theta_cached = einsum(position_tensor, inv_freq, "seq_len, d -> seq_len d")
            cos = torch.cos(theta_cached)
            sin = torch.sin(theta_cached)
        x_reshaped = rearrange(x, "... (d2 c) -> ... d2 c", c=2)
        logging.debug(f"x_reshaped shape: {x_reshaped.shape}, cos shape: {cos.shape}, sin shape: {sin.shape}")
        x_0, x_1 = x_reshaped[..., 0], x_reshaped[..., 1]
        x_0_rotated = x_0 * cos - x_1 * sin
        x_1_rotated = x_0 * sin + x_1 * cos
        x_rotated = rearrange(torch.stack((x_0_rotated, x_1_rotated), dim=-1), "... d2 c -> ... (d2 c)")
        return x_rotated
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, self.d_k * self.num_heads, bias=False, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.d_k * self.num_heads, bias=False, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.d_v * self.num_heads, bias=False, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
    
    def reset_paramerters(self):
        pass
    
    def forward(
        self, x: Float[Tensor, "... seq_len d_model"],
        mask: Float[Tensor, "... seq_len seq_len"] | None = None,
        ROPE: RotaryPositionalEmbedding | None = None
        ) -> Float[Tensor, "... seq_len d_model"]:
        Q = rearrange(self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        logging.debug(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        if isinstance(ROPE, RotaryPositionalEmbedding):
            Q = ROPE(Q)
            K = ROPE(K)
        result = scaled_dot_product_attention(Q, K, V, mask=mask)
        result = rearrange(result, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.output_proj(result)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiHeadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model, d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
    def reset_parameters(self):
        pass
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"], 
                mask: Float[Tensor, "... seq_len seq_len"] | None = None, 
                ROPE: RotaryPositionalEmbedding | None = None) -> Float[Tensor, "... seq_len d_model"]:
        x1 = self.ln1(x)
        x1 = self.attn(x1, mask=mask, ROPE=ROPE)
        x = x + x1
        x2 = self.ln2(x)
        x2 = self.ffn(x2)
        x = x + x2
        return x
    
    
class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        num_layers: int, 
        vocab_size: int, 
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
        ): 
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, buffer=True, device=device, dtype=dtype)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=dtype)),
            persistent=False
        )  
        
    def reset_parameters(self):
        pass
    
    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        seq_len = x.shape[-1]
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x, mask=self.causal_mask[:seq_len, :seq_len], ROPE=self.rope)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0) 
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
        

def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_v"],
    mask: Float[Tensor, "... seq_len seq_len"] | None = None,
) -> Float[Tensor, "... seq_len d_v"]:
    d_k = Q.shape[-1]
    dot_result = einsum(Q, K, "... seq_len0 d_k, ... seq_len1 d_k -> ... seq_len0 seq_len1") / sqrt(d_k)
    if mask is not None:
        dot_result = dot_result.masked_fill_(mask == False, float("-inf"))
    return einsum(softmax(dot_result), V, "... seq_len0 seq_len1, ... seq_len1 d_v -> ... seq_len0 d_v")
    
    
def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    max_value, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_value
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"],
    targets: Float[Tensor, "... 1"],
) -> Float[Tensor, "..."]:
    max_value, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - max_value
    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_sum_exp_logits = torch.log(sum_exp_logits)
    loss = - logits.gather(dim=-1, index=targets) + log_sum_exp_logits
    return loss.mean()
    
