"""Opt-in exact FP32 top-2048 for DSA decode and speculative verify on B200.

DeepGEMM's paged MQA logits kernel scores every live KV slot of each query token and, given ``histogram``, also
counts the scores into 1024 coarse bins; ``select.cuh`` then selects each token's exact FP32 top-2048 and maps it to
physical KV slots. Needs a DeepGEMM build whose ``fp8_fp4_paged_mqa_logits`` takes ``histogram``.
"""

from __future__ import annotations

from pathlib import Path

import deep_gemm
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

TOPK = 2048


@cache_once
def _load_selector():
    return load_jit(
        "litetopk_decode_select",
        cuda_files=[str(Path(__file__).resolve().parent / "select.cuh")],
        cuda_wrappers=[("select", "litetopk::select")],
    )


class FusedDecodePlan:
    """Buffers of one batch shape, reused across calls and CUDA graph replays.

    The selector clears the histogram and candidates and balances its hand-off counters, so no reset runs between calls. A token whose
    crossing bin holds more than ``candidate_capacity`` scores, or whose histogram disagrees with its scores, is still
    selected exactly by a slower whole-row path.
    """

    def __init__(
        self,
        batch: int,
        next_n: int,
        device: torch.device,
        candidate_capacity: int = 8192,
    ):
        rows = batch * next_n
        self.next_n = next_n
        self.candidate_capacity = candidate_capacity
        self.histogram = torch.zeros((rows, 1024), dtype=torch.int32, device=device)
        # Per row: 16 bytes of hand-off state (litetopk::RowState) and the 8-byte candidate entries
        self.workspace = torch.zeros(
            rows * (16 + 8 * candidate_capacity), dtype=torch.uint8, device=device
        )
        self.output = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
        self.selector = _load_selector()

    def __call__(
        self,
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len,
    ):
        """Physical KV slots of each token's top-2048: int32 [B * next_n, 2048], unsorted, padded with -1.

        Arguments are those of ``deep_gemm.fp8_fp4_paged_mqa_logits``: ``context_lens`` is int32 [B, next_n] and a
        request's tokens share its ``block_table`` row. For speculative verify, supply each token's causal length
        (and build ``schedule_metadata`` from those lengths). The output buffer is overwritten by the next call.
        """
        scores = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, None),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_metadata,
            max_context_len,
            clean_logits=False,
            histogram=self.histogram,
        )
        self.selector.select(
            scores,
            context_lens.view(-1),
            self.histogram,
            block_table,
            self.next_n,
            self.output,
            self.workspace,
            self.candidate_capacity,
        )
        return self.output
