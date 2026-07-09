# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_buckets,
        shared_pos=True,
        dropout=0.1,
    ):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1)
        )
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn) ** -0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn) ** -0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads) ** -0.5
        )


class WanTextEncoder(torch.nn.Module):

    def __init__(
        self,
        vocab: int | nn.Embedding = 256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1,
        text_encoder_pretrained_path: str = None,
    ):
        super(WanTextEncoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.text_encoder_pretrained_path = text_encoder_pretrained_path

        # layers
        if isinstance(vocab, int):
            self.token_embedding = nn.Embedding(vocab, dim)
        else:
            self.token_embedding = vocab
        if shared_pos:
            self.pos_embedding = T5RelativeEmbedding(
                num_buckets, num_heads, bidirectional=True
            )
        else:
            self.pos_embedding = None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        if self.shared_pos:
            assert self.pos_embedding is not None
            e = self.pos_embedding(x.size(1), x.size(1))
        else:
            e = None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x

    @staticmethod
    def state_dict_converter():
        return WanTextEncoderStateDictConverter()


class WanTextEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict

    def from_civitai(self, state_dict):
        return state_dict
