"""Greedy draft tokens on the vocab-parallel TP LM head without the logits all-to-all.

A row gather collects the draft hidden rows MAX_LEN-style with a multimem one-shot store instead
of zero-fill + all-reduce; a shard select reduces each row on this rank's vocab shard and gathers
the (value, id) pairs. Tokens equal the argmax over the full vocab, ties to the lowest
id, with the NaN rule of the argmax each draft call replaces.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import torch

from sglang.kernels.ops.speculative.k75p_argmax import k75p_select, k75p_shard_argmax
from sglang.kernels.ops.speculative.topk1 import draft_topk1_argmax_only
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# One (fp32 bits, int32 id) pair as bf16 elements for the multimem gather.
_PAIR_BF16 = 4


class K75PDraftSelect:
    def __init__(self, *, max_rows: int, hidden_size: int):
        tp_group = get_parallel().tp_group
        self.world = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        # Flat one-token buffers: rank r's block lands at offset r * block, so the
        # gathered rows come out rank-major ([world * rows, ...]).
        self.hidden_state = triton_symm_mem_ag.create_state(
            group=tp_group.device_group,
            rank_in_group=self.rank,
            max_tokens=1,
            hidden_size=self.world * max_rows * hidden_size,
        )
        self.pairs_state = triton_symm_mem_ag.create_state(
            group=tp_group.device_group,
            rank_in_group=self.rank,
            max_tokens=1,
            hidden_size=self.world * self.world * max_rows * _PAIR_BF16,
        )
        self.pairs = torch.empty(
            (self.world * max_rows, 2),
            dtype=torch.int32,
            device=self.hidden_state.device,
        )

    @classmethod
    def create(cls, *, max_rows: int, hidden_size: int) -> Optional[K75PDraftSelect]:
        select = cls(max_rows=max_rows, hidden_size=hidden_size)
        if (
            select.hidden_state.symm_mem_hdl.multicast_ptr == 0
            or select.pairs_state.symm_mem_hdl.multicast_ptr == 0
        ):
            logger.warning("K75P draft LM head disabled: no multicast on the TP group.")
            return None
        logger.info(
            "K75P draft LM head active (tp=%d, rows/rank <= %d).",
            select.world,
            max_rows,
        )
        return select

    def fits(self, hidden_states: torch.Tensor) -> bool:
        return (
            0 < hidden_states.shape[0] <= self.max_rows
            and hidden_states.shape[1] == self.hidden_size
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        )

    def gather_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No entry barrier: every rank passes this call's pair gather, which follows
        # its GEMM, before any rank stores the next call's rows (and vice versa).
        rows = hidden_states.shape[0]
        return triton_symm_mem_ag.all_gather_inner(
            self.hidden_state,
            hidden_states.view(1, rows * self.hidden_size),
            tp_hidden_dim=self.world * rows * self.hidden_size,
            skip_entry_sync=True,
            safe=False,
        ).view(self.world * rows, self.hidden_size)

    def select(
        self, *, logits: torch.Tensor, num_local_rows: int, torch_nan: bool
    ) -> torch.Tensor:
        total_rows = logits.shape[0]
        pairs = self.pairs[:total_rows]
        k75p_shard_argmax(
            logits=logits,
            pairs=pairs,
            vocab_start=self.rank * logits.shape[1],
            torch_nan=torch_nan,
        )
        gathered = triton_symm_mem_ag.all_gather_inner(
            self.pairs_state,
            pairs.view(torch.bfloat16).view(1, total_rows * _PAIR_BF16),
            tp_hidden_dim=self.world * total_rows * _PAIR_BF16,
            skip_entry_sync=True,
            safe=False,
        )
        return k75p_select(
            gathered=gathered.view(torch.int32).view(self.world, total_rows, 2),
            row_start=self.rank * num_local_rows,
            num_rows=num_local_rows,
            torch_nan=torch_nan,
        )


class K75PProof:
    """Check every K75P draft pick against the stock LM-head path, eager or in CUDA graphs.

    Counters live on the device and are mirrored into pinned host memory by a copy that is part
    of the check, so the reporter thread logs them without any CUDA call (as CombineProof).
    """

    PERIOD_S = 30.0
    KINDS = ("decode", "extend")

    def __init__(self):
        # [kind][calls, rows, mismatched rows]
        self.counters = torch.zeros(
            (2, 3), dtype=torch.int64, device=torch.cuda.current_device()
        )
        # Model init runs under a CUDA default-device context; pinned memory must say cpu.
        self.host_counters = torch.zeros(
            (2, 3), dtype=torch.int64, device="cpu", pin_memory=True
        )
        logger.info("K75P proof active")
        threading.Thread(target=self._report, daemon=True).start()

    def check(
        self, *, extend: bool, token_ids: torch.Tensor, stock_logits: torch.Tensor
    ) -> None:
        # The stock pick of the replaced path: torch.argmax after the draft extend,
        # the split argmax of draft_topk1_postprocess in draft decode steps.
        if extend:
            stock_ids = torch.argmax(stock_logits, dim=-1, keepdim=True)
        else:
            _, stock_ids = draft_topk1_argmax_only(stock_logits)
        counters = self.counters[int(extend)]
        counters[0].add_(1)
        counters[1].add_(token_ids.shape[0])
        counters[2].add_(torch.count_nonzero(stock_ids != token_ids))
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            for kind, (calls, rows, mismatched) in zip(
                self.KINDS, self.host_counters.tolist()
            ):
                logger.info(
                    "K75P proof: kind=%s calls=%d rows=%d mismatched=%d",
                    kind,
                    calls,
                    rows,
                    mismatched,
                )
