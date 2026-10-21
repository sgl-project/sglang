"""Opt-in SID arithmetic over the ordinary GR paged KV implementation.

This example adapter changes attention arithmetic only. The core backend still
owns KV writes and metadata, and the scheduler still owns every allocation.
Gathered tensors are temporary copies in logical token order, not another cache.
"""

import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.layers.attention.gr_paged_backend import (
    GRPagedAttentionBackend,
    _context_locations,
)
from sglang.srt.runtime_context import get_parallel

from .gr_kernel import beam_decode_with_reference_splits, load_gr_kernel


class SIDPagedAttentionBackend(GRPagedAttentionBackend):
    def __init__(self, runner):
        super().__init__(runner)
        self.tp_size = get_parallel().attn_tp_size
        # Fail at startup if the explicitly requested reference kernel is absent.
        load_gr_kernel()

    def _prefill_attention(self, q, key_pool, value_pool, meta, scale):
        locations = _context_locations(
            meta.context_pages, meta.context_len, self.page_size
        )
        query = q.transpose(0, 1)[None]
        key = key_pool[locations].transpose(0, 1)[None]
        value = value_pool[locations].transpose(0, 1)[None]
        groups = q.shape[1] // key_pool.shape[1]
        # Cold prefill uses SID's exact SDPA call. Cached suffixes require an
        # absolute-position causal mask: SDPA is_causal starts at position zero.
        mask = None
        if meta.prefix_len:
            positions = meta.prefix_len + torch.arange(q.shape[0], device=q.device)
            mask = (
                torch.arange(meta.context_len, device=q.device)[None, :]
                <= positions[:, None]
            )
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(groups, dim=1),
                value.repeat_interleave(groups, dim=1),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=meta.prefix_len == 0,
                scale=scale,
            )
        return output.squeeze(0).transpose(0, 1)

    def _decode_attention(self, q, key_pool, value_pool, meta, scale):
        width, heads, dim = q.shape
        if scale != dim**-0.5:
            raise ValueError("SID reference decode requires inverse-sqrt head scaling")
        context_locations = _context_locations(
            meta.context_pages, meta.context_len, self.page_size
        )
        key_context = key_pool[context_locations][None].contiguous()
        value_context = value_pool[context_locations][None].contiguous()
        generated_locations = meta.beam_locations()
        steps = meta.step + 1

        def gather(pool):
            # Selected ancestors may be shared. Repack them into a step-major
            # temporary tensor and give the kernel an identity path through it.
            return (
                pool[generated_locations]
                .transpose(0, 1)
                .reshape(1, steps * width, pool.shape[1], pool.shape[2])
                .contiguous()
            )

        paths = (
            torch.arange(steps * width, device=q.device, dtype=torch.int32)
            .reshape(1, 1, 1, steps, width)
            .expand(1, 1, heads, steps, width)
            .contiguous()
        )
        output = beam_decode_with_reference_splits(
            q.reshape(1, 1, width, heads, dim).contiguous(),
            key_context,
            value_context,
            gather(key_pool),
            gather(value_pool),
            paths,
            steps,
            reference_heads=heads * self.tp_size,
        )
        return output.reshape_as(q)


def register_paged_reference_backend():
    if os.environ.get("SGLANG_SID_GR_PAGED_REFERENCE") == "1":
        # Importing attention_registry above completes the default registrations
        # before this explicit example override is installed.
        register_attention_backend("gr_paged")(SIDPagedAttentionBackend)
