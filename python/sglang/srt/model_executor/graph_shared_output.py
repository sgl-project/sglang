from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_exec,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class GraphSharedOutput:
    """``(max_rows, vocab)`` logits buffer, shared by every cuda-graph runner."""

    _process_shared: GraphSharedOutput | None = None

    def __init__(
        self,
        *,
        device: torch.device,
        max_rows: int,
    ) -> None:
        self.device = torch.device(device)
        self.max_rows = max_rows
        self._logits_buffers: dict[int, torch.Tensor] = {}
        self._compact_logits_buffer = None

    @classmethod
    def create_for_model_runner(
        cls, model_runner: ModelRunner
    ) -> GraphSharedOutput | None:
        cuda_graph_config = get_exec().graph.cuda_graph_config
        if cuda_graph_config is None:
            return None

        max_rows = 0
        decode = cuda_graph_config.decode
        if decode.backend != Backend.DISABLED and decode.bs:
            max_rows = max(max_rows, model_runner.max_decode_logits_rows())

        if max_rows <= 0:
            return None

        device = torch.device(model_runner.device)
        from sglang.srt.speculative.compact_verify.config import sharded_graph_output

        if sharded_graph_output(model_runner):
            # Do not enlarge the draft runner's full-vocab pool to target R.
            return cls(device=device, max_rows=max_rows)
        shared = cls._process_shared
        if (
            shared is not None
            and shared.device == device
            and shared.max_rows >= max_rows
        ):
            return shared
        cls._process_shared = cls(device=device, max_rows=max_rows)
        return cls._process_shared

    def get_logits_buffer(self, vocab_size: int, *, rows: int) -> torch.Tensor:
        assert rows <= self.max_rows, (
            f"shared logits buffer holds {self.max_rows} rows but caller "
            f"needs {rows} (vocab_size={vocab_size})"
        )
        buffer = self._logits_buffers.get(vocab_size)
        if buffer is None:
            buffer = torch.zeros(
                (self.max_rows, vocab_size), dtype=torch.float, device=self.device
            )
            self._logits_buffers[vocab_size] = buffer
        return buffer[:rows]

    def get_compact_logits_buffer(self, *, rows: int) -> torch.Tensor:
        assert rows <= self.max_rows
        if self._compact_logits_buffer is None:
            self._compact_logits_buffer = torch.zeros(
                (self.max_rows, 38720), device=self.device, dtype=torch.bfloat16
            )
        return self._compact_logits_buffer[:rows]
