from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class GraphSharedOutput:
    """``(max_rows, vocab)`` logits buffer, shared by every cuda-graph runner."""

    _process_shared: Optional[GraphSharedOutput] = None

    def __init__(
        self,
        *,
        device: torch.device,
        max_rows: int,
    ) -> None:
        self.device = torch.device(device)
        self.max_rows = max_rows
        self._logits_buffers: Dict[int, torch.Tensor] = {}

    @classmethod
    def create_for_model_runner(
        cls, model_runner: ModelRunner
    ) -> Optional[GraphSharedOutput]:
        cuda_graph_config = model_runner.server_args.cuda_graph_config
        if cuda_graph_config is None:
            return None

        max_rows = 0
        decode = cuda_graph_config.decode
        if decode.backend != Backend.DISABLED and decode.bs:
            max_rows = max(max_rows, model_runner.max_decode_logits_rows())

        if max_rows <= 0:
            return None

        device = torch.device(model_runner.device)
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
