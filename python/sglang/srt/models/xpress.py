# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's Qwen3 XPress implementation.

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel, _radix_topk
from sglang.srt.speculative import xpress_kernels


class XPressRefinerHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_hidden: int = 512,
        topc: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.topc = int(topc)
        if self.rank < 32 or self.rank & (self.rank - 1):
            raise ValueError("XPress rank must be a power of two and at least 32.")
        if mlp_hidden <= 0 or mlp_hidden % 64:
            raise ValueError("XPress MLP width must be a positive multiple of 64.")
        if not 0 <= self.topc <= self.vocab_size:
            raise ValueError("XPress top-C must be between 0 and vocab_size.")
        self.w1 = nn.Embedding(vocab_size, rank)
        self.down_h = nn.Linear(hidden_size, rank, bias=False)
        self.down_g = nn.Linear(hidden_size, rank, bias=False)
        self.in_proj = nn.Linear(3 * rank, rank, bias=False)
        self.mix_L = nn.Parameter(
            torch.eye(block_size).expand(rank, -1, -1).contiguous()
        )
        self.mlp_gate = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, rank, bias=False)
        self.w2 = nn.Linear(rank, vocab_size, bias=False)
        self._fused_buf = None

    @torch.no_grad()
    def fold_from_raw_(self, raw: torch.Tensor) -> None:
        if tuple(raw.shape) != tuple(self.mix_L.shape):
            raise ValueError(
                f"XPress mixer shape {tuple(raw.shape)} does not match "
                f"{tuple(self.mix_L.shape)}."
            )
        b = self.block_size
        tril = torch.tril(torch.ones(b, b, dtype=raw.dtype, device=raw.device))
        eye = torch.eye(b, dtype=raw.dtype, device=raw.device)
        self.mix_L.copy_((raw * tril + eye).to(self.mix_L))

    def hidden_cache(self, hidden: torch.Tensor) -> torch.Tensor:
        global_hidden = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
        return torch.cat([self.down_h(hidden), self.down_g(global_hidden)], dim=-1)

    def refine_bias(self, prev_ids: torch.Tensor, hcache: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(torch.cat([hcache, self.w1(prev_ids)], dim=-1))
        x = torch.bmm(self.mix_L.to(x.dtype), x.permute(2, 1, 0)).permute(2, 1, 0)
        x = x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))
        return self.w2(x)

    @torch.no_grad()
    def build_fused_buffers(self) -> None:
        r = self.rank
        w = self.in_proj.weight.detach()
        self._fused_buf = {
            "whc_t": w[:, : 2 * r].t().contiguous(),
            "wlat_t": w[:, 2 * r :].t().contiguous(),
            "mix_kjc": self.mix_L.detach().permute(1, 2, 0).contiguous(),
            "wg_t": self.mlp_gate.weight.detach().t().contiguous(),
            "wu_t": self.mlp_up.weight.detach().t().contiguous(),
            "wd_t": self.mlp_down.weight.detach().t().contiguous(),
            "w2_t": self.w2.weight.detach().t().contiguous(),
        }

    def jacobi_refine_greedy(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        anchor: torch.Tensor,
        num_passes: int,
    ) -> torch.Tensor:
        n, b, v = base_logits.shape
        if b != self.block_size or v != self.vocab_size:
            raise ValueError(
                "XPress base logits have an incompatible block or vocabulary."
            )
        buf = self._fused_buf
        if buf is None:
            raise RuntimeError("XPress weights must be loaded before drafting.")
        rows = n * (b - 1)
        blk = torch.empty((n, b), dtype=torch.int64, device=hidden.device)
        blk[:, 0] = anchor
        blk[:, 1:] = base_logits[:, 1:].argmax(dim=-1)
        hcache = self.hidden_cache(hidden)
        xh = torch.mm(hcache.reshape(n * b, -1), buf["whc_t"]).view(n, b, self.rank)
        lat = torch.empty(
            (n, b - 1, self.rank), dtype=hidden.dtype, device=hidden.device
        )

        if self.topc:
            c = min(self.topc, v)
            base_c, cand = _radix_topk(base_logits[:, 1:].reshape(rows, v), c)
            w2c = self.w2.weight[cand].reshape(rows, c, self.rank)
            for _ in range(num_passes):
                xpress_kernels.xpress_latent_pass(
                    blk,
                    anchor,
                    xh,
                    lat,
                    self.w1.weight,
                    buf["wlat_t"],
                    buf["mix_kjc"],
                    buf["wg_t"],
                    buf["wu_t"],
                    buf["wd_t"],
                )
                bias = torch.bmm(w2c, lat.reshape(rows, self.rank, 1)).view(n, b - 1, c)
                xpress_kernels.fused_topc_argmax_to_blk(
                    base_c.view(n, b - 1, c), bias, cand.view(n, b - 1, c), blk
                )
        else:
            base = base_logits[:, 1:].reshape(rows, v)
            bias = torch.empty_like(base)
            chunks = (v + 4095) // 4096
            out_val = torch.empty(
                (rows, chunks), dtype=torch.float32, device=hidden.device
            )
            out_idx = torch.empty(
                (rows, chunks), dtype=torch.int64, device=hidden.device
            )
            for _ in range(num_passes):
                xpress_kernels.xpress_latent_pass(
                    blk,
                    anchor,
                    xh,
                    lat,
                    self.w1.weight,
                    buf["wlat_t"],
                    buf["mix_kjc"],
                    buf["wg_t"],
                    buf["wu_t"],
                    buf["wd_t"],
                )
                torch.mm(lat.reshape(rows, self.rank), buf["w2_t"], out=bias)
                xpress_kernels.fused_add_argmax_to_blk(
                    base, bias, out_val, out_idx, blk
                )
        return blk[:, 1:]


_HEAD_NAMES = {
    "w1.weight": "w1.weight",
    "down_h.weight": "down_h.weight",
    "down_g.weight": "down_g.weight",
    "in_proj.weight": "in_proj.weight",
    "mlp.gate_proj.weight": "mlp_gate.weight",
    "mlp.up_proj.weight": "mlp_up.weight",
    "mlp.down_proj.weight": "mlp_down.weight",
    "w2.weight": "w2.weight",
}


class Qwen3XPressModel(DFlashDraftModel):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        dflash_config = getattr(config, "dflash_config", None) or {}
        if not isinstance(dflash_config, dict):
            dflash_config = vars(dflash_config)
        if getattr(config, "sample_from_anchor", False) or dflash_config.get(
            "sample_from_anchor", False
        ):
            raise ValueError(
                "XPress requires fill-in layout, with the anchor at slot 0."
            )
        if (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        ) != config.vocab_size:
            raise ValueError("XPress requires a full-vocabulary draft.")
        block_size = int(getattr(config, "xpress_block_size", None) or self.block_size)
        if block_size != self.block_size:
            raise ValueError(
                "XPress mixer block size must equal the DFlash block size."
            )
        self.xpress_head = XPressRefinerHead(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            block_size=block_size,
            rank=int(getattr(config, "xpress_rank", 256)),
            mlp_hidden=int(getattr(config, "xpress_mlp_hidden", 512)),
            topc=int(getattr(config, "xpress_topc", 512)),
        )
        self.num_jacobi_passes = int(getattr(config, "xpress_num_passes", 6))
        if self.num_jacobi_passes < 1:
            raise ValueError("XPress requires at least one Jacobi pass.")

    def set_block_size(self, block_size: int) -> None:
        if int(block_size) != self.xpress_head.block_size:
            raise ValueError("XPress block size must match the checkpoint mixer.")
        super().set_block_size(block_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mapped = []
        seen = set()
        raw_mix = None
        for name, weight in weights:
            stripped = name.removeprefix("model.")
            if stripped.startswith("lm_head."):
                raise ValueError(
                    "XPress checkpoints with a separate LM head are not supported."
                )
            if stripped.startswith("xpress_head."):
                suffix = stripped[len("xpress_head.") :]
                if suffix == "mix.L":
                    raw_mix = weight
                    continue
                if suffix not in _HEAD_NAMES:
                    raise ValueError(f"Unknown XPress head weight {name!r}.")
                mapped_name = "xpress_head." + _HEAD_NAMES[suffix]
                seen.add(mapped_name)
                mapped.append((mapped_name, weight))
            else:
                mapped.append((name, weight))
        missing = {"xpress_head." + name for name in _HEAD_NAMES.values()} - seen
        if raw_mix is None or missing:
            raise ValueError(
                f"XPress checkpoint is missing mixer or head weights: {sorted(missing)}."
            )
        super().load_weights(mapped)
        self.xpress_head.fold_from_raw_(raw_mix)
        self.post_load_weights()

    def post_load_weights(self) -> None:
        # Some SGLang loaders copy an already-folded state directly into the
        # parameters instead of calling load_weights(). Their transposed mixer
        # and projection buffers must follow the newly loaded parameters.
        self.xpress_head.build_fused_buffers()


class DFlashQwen3XPressModel(Qwen3XPressModel):
    pass


EntryClass = [Qwen3XPressModel, DFlashQwen3XPressModel]
