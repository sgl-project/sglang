# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Backend interface for CUDA graph capture/replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ExecutionBackend(ABC):
    """Pure ABC: no state, no defaults. Each implementation owns its
    per-backend state and binds the handles it needs from the
    cuda_graph_runner passed to its __init__.

    Methods:
      - record_session(stream) — context wrapping the runner's outer
        prepare loop; backends bind stream / pool and open per-backend
        capture flags here.
      - record(shape_key, forward_fn, dummies, post_warmup_hook)
        — record the runnable artifact for shape_key; one call per
        shape inside record_session.
      - can_run_graph(forward_batch, shape_key) — can this backend run
        for the given batch at the given shape.
      - run_session() — context wrapping run-time model code;
        backends open the "we are replaying" flag here when they have
        one.
      - run(shape_key, static_forward_batch, **kwargs) — invoke
        the recorded artifact (graph replay; eager runs forward live).
      - cleanup() — release pool and drop recorded artifacts.

    Notes:
      - The outer prepare loop is runner-specific; it lives on the
        runner, not here.
    """

    @abstractmethod
    def record_session(self, stream: torch.cuda.Stream) -> Iterator[None]: ...

    @abstractmethod
    def record(
        self,
        shape_key: ShapeKey,
        forward_fn,
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None: ...

    @abstractmethod
    def can_run_graph(
        self, forward_batch: ForwardBatch, shape_key: ShapeKey
    ) -> bool: ...

    @abstractmethod
    def run_session(self) -> Iterator[None]: ...

    @abstractmethod
    def run(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...
