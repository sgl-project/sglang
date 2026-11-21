"""Token-budget CUDA graph runner for windowed audio encoder towers.

Audio clips are short, so a windowed audio encoder is launch-bound: dozens of
small varlen-attention and MLP kernels per call. This runner captures the
transformer layers (and the output projection) once per token budget and
replays the smallest budget that fits a batch:

* Hidden states are copied into a fixed ``[budget, hidden]`` buffer.
* Window boundaries go into a fixed ``cu_seqlens`` buffer. Unused entries
  repeat the last offset, so padding rows belong to no window and attend to
  nothing; every other op in the layers is per-token.
* ``max_seqlen`` is fixed to the full window length, an upper bound of every
  window the encoder builds.

Batches larger than the biggest budget, or with more windows than a budget
holds, run eagerly.
"""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

from sglang.srt.layers.attention.vision import VisionAttentionMetadata

logger = logging.getLogger(__name__)

# Attention backends allowed inside the graphs. The Triton varlen kernel reads
# cu_seqlens on the device. With FA3 varlen these graphs produced wrong
# outputs under concurrent serving (cause not isolated), so FA3 stays eager.
GRAPH_SAFE_ATTENTION_BACKENDS = ("triton_attn",)
GRAPH_ATTENTION_BACKEND = "triton_attn"

DEFAULT_TOKEN_BUDGETS = (
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
)

# Extra window slots per budget on top of ``budget // window``: every item
# adds at most one partial window.
DEFAULT_MAX_ITEMS = 64


@dataclass
class _CapturedAudioGraph:
    graph: torch.cuda.CUDAGraph
    hidden_states: torch.Tensor
    cu_seqlens: torch.Tensor
    output: torch.Tensor


class AudioEncoderCudaGraphRunner:
    """Replays ``forward_transformer(hidden_states, metadata)`` from graphs.

    ``forward_transformer`` must only launch device work whose shapes depend
    on ``hidden_states.shape`` and the ``cu_seqlens`` buffer length.
    """

    def __init__(
        self,
        forward_transformer: Callable[
            [torch.Tensor, VisionAttentionMetadata], torch.Tensor
        ],
        *,
        hidden_size: int,
        window_size: int,
        dtype: torch.dtype,
        device: torch.device,
        token_budgets: Sequence[int] = DEFAULT_TOKEN_BUDGETS,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.forward_transformer = forward_transformer
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dtype = dtype
        self.device = device
        self.token_budgets = sorted(set(int(b) for b in token_budgets))
        self.max_items = max_items
        self.graphs: Dict[int, _CapturedAudioGraph] = {}
        self.failed_budgets: set[int] = set()
        # A private pool. The decode graphs' shared pool holds memory that is
        # still live between their replays, so an encoder graph captured into
        # it can overwrite another batch's buffers under load.
        self._pool = None
        logger.info(
            "Audio encoder CUDA graph: window=%d budgets=%s",
            window_size,
            self.token_budgets,
        )

    def max_windows(self, budget: int) -> int:
        return budget // self.window_size + self.max_items

    def select_budget(self, num_tokens: int, num_windows: int) -> Optional[int]:
        start = bisect.bisect_left(self.token_budgets, num_tokens)
        for budget in self.token_budgets[start:]:
            if budget in self.failed_budgets:
                continue
            if num_windows <= self.max_windows(budget):
                return budget
        return None

    def _metadata(self, cu_seqlens: torch.Tensor) -> VisionAttentionMetadata:
        return VisionAttentionMetadata(
            cu_seqlens=cu_seqlens,
            seq_lens=cu_seqlens[1:] - cu_seqlens[:-1],
            max_seqlen=self.window_size,
        )

    def _capture(self, budget: int) -> _CapturedAudioGraph:
        hidden_states = torch.zeros(
            budget, self.hidden_size, dtype=self.dtype, device=self.device
        )
        # Full windows over the whole budget for warmup/capture; replays only
        # overwrite the buffer contents.
        num_slots = self.max_windows(budget) + 1
        offsets = list(range(0, budget, self.window_size)) + [budget]
        offsets += [budget] * (num_slots - len(offsets))
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        # Warm up outside the capture (autotuning, lazy JIT, workspaces).
        with torch.no_grad():
            self.forward_transformer(hidden_states, self._metadata(cu_seqlens))
        torch.cuda.synchronize(self.device)

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=self._pool):
            output = self.forward_transformer(hidden_states, self._metadata(cu_seqlens))
        torch.cuda.synchronize(self.device)
        return _CapturedAudioGraph(graph, hidden_states, cu_seqlens, output)

    def capture_all(self) -> None:
        """Capture every budget once, largest first, so serving never stalls
        on a capture after the first audio batch."""
        start = time.perf_counter()
        reserved_before = torch.cuda.memory_reserved(self.device)
        for budget in reversed(self.token_budgets):
            try:
                self.graphs[budget] = self._capture(budget)
            except Exception:
                self.failed_budgets.add(budget)
                logger.exception(
                    "Audio encoder CUDA graph capture failed for budget=%d; "
                    "using eager fallback for it",
                    budget,
                )
        logger.info(
            "Captured %d audio encoder CUDA graphs in %.2f s (reserved +%.1f MiB)",
            len(self.graphs),
            time.perf_counter() - start,
            (torch.cuda.memory_reserved(self.device) - reserved_before) / 2**20,
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: List[int],
        budget: int,
    ) -> Optional[torch.Tensor]:
        """Run one batch through the graph of ``budget``.

        ``cu_seqlens`` are the host-side window offsets (leading zero
        included). Returns ``None`` if that budget failed to capture; the
        caller then runs eagerly.
        """
        if not self.graphs and not self.failed_budgets:
            self.capture_all()
        entry = self.graphs.get(budget)
        if entry is None:
            return None

        num_tokens = hidden_states.shape[0]
        num_slots = entry.cu_seqlens.numel()
        padded = cu_seqlens + [cu_seqlens[-1]] * (num_slots - len(cu_seqlens))
        entry.cu_seqlens.copy_(
            torch.tensor(padded, dtype=torch.int32).pin_memory(), non_blocking=True
        )
        entry.hidden_states[:num_tokens].copy_(hidden_states)
        entry.graph.replay()
        output = entry.output.reshape(-1, entry.output.shape[-1])
        return output[:num_tokens].clone()
