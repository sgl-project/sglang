# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)).mul_(self.w1(x))
        x = self.w2(x)
        return x


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer("zero_k_bias", torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim**-0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(
            input=x,
            weight=self.qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        q, k, v = (
            qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        x = scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x


class AttnProjection(nn.Module):
    def __init__(
        self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2
    ):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)
        # self.mlp = FeedForward(out_dim, out_dim)

    def forward(self, x):
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return self.mlp(self.norm2(x)).add_(x)
