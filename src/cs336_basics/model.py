import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange, einsum
from math import sqrt
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.infeatures = in_features
        self.outfeatures = out_features
        self.weights = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), device=device, dtype=dtype)
        self.reset_parameter()
            
    def reset_parameter(self):
        sigma = sqrt(2.0 / (self.infeatures + self.outfeatures))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        if hasattr(self, "bias"):
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(self.weights, x, "dout din, ... din -> ... dout")
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
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weights, mean=0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
    
    def reset_parameters(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x.pow(2)
        mean_of_squares = x_squared.mean(dim=-1, keepdim=True)
        rms_val = torch.sqrt(mean_of_squares + self.eps)
        result = (x / rms_val) * self.scale
        return result.to(in_dtype)
    

class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, inner_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtye = dtype
        self.infeatures = in_features
        self.outfeatures = out_features
        self.inner_features = inner_features
        self.linear1 = Linear(in_features, inner_features, bias=False, device=device, dtype=dtype)
        self.linear2 = Linear(inner_features, out_features, bias=False, device=device, dtype=dtype)
        self.linear3 = Linear(in_features, inner_features, bias=False, device=device, dtype=dtype)

    def register_parameter(self):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.linear3(x)
        x_1 = self.linear1(x)
        return self.linear2(SiLU(x_1) * x_3)
    
    
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
    
    def forward(self, x: Float[Tensor, "... seq_len d_k"]) -> Float[Tensor, "..."]:
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
        x_0, x_1 = x_reshaped[..., 0], x_reshaped[..., 1]
        x_0_rotated = x_0 * cos - x_1 * sin
        x_1_rotated = x_0 * sin + x_1 * cos
        x_rotated = rearrange(torch.stack((x_0_rotated, x_1_rotated), dim=-1), "... d2 c -> ... (d2 c)")
        return x_rotated
    

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
    return einsum(Softmax(dot_result), V, "... seq_len0 seq_len1, ... seq_len1 d_v -> ... seq_len0 d_v")
    
    
def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


def Softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    max_value, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_value
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x
