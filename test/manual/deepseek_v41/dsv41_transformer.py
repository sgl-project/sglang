"""Text-only DeepSeek V4.1 forward built from the reference-style modules:
embed -> hc_mult copies -> blocks -> collapse -> logits."""

import torch
import torch.nn.functional as F
from dsv41_args import DeepseekV41Args
from dsv41_block import Block
from dsv41_engram import EngramLayout, NgramHashState
from dsv41_hc import hc_pre, make_identity_pre_mix
from dsv41_norm import RMSNorm
from dsv41_shared import SharedAttentionRuntime
from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class Head(nn.Module):
    """bf16 in the checkpoint, held in fp32 so the logits come out in fp32."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


class Transformer(nn.Module):
    def __init__(self, args: DeepseekV41Args, tokenizer=None):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash = None
        if self.engram_layout is not None:
            self.engram_hash = NgramHashState(args, self.engram_layout, tokenizer)
        self.embed = Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            Block(args, layer_id, self.engram_layout)
            for layer_id in range(args.n_layers)
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = Head(args.vocab_size, args.dim)
        self.shared = SharedAttentionRuntime()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        token_types: torch.Tensor | None = None,
        full_logits: bool = False,
    ) -> torch.Tensor:
        """input_ids [b, s]; token_types [b, s] with -1 for text, >= 0 inside an image span
        (selects the VL routing bias and keeps those tokens out of n-grams)."""
        image_mask = None if token_types is None else token_types >= 0
        engram_mask = None if image_mask is None else ~image_mask
        hashes = None
        if self.engram_hash is not None:
            hashes = self.engram_hash(input_ids, start_pos, engram_mask)
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(
                    h, hashes[:, :, layer.engram.layer_hash_index, :], engram_mask
                )
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask, self.shared)
        h = hc_pre(h, pre_mix)
        return self.head(self.norm(h), full_logits=full_logits)
