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

from sglang.srt.model_executor.graph_serialization.loadstore import (
    GraphImportError,
    unsupported_shape,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.graph_serialization.loadstore import (
        GraphCache,
        ShapeArtifact,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BaseCudaGraphBackend(ABC):
    """ABC with no state. Each implementation owns its per-backend state and
    binds the handles it needs from the cuda_graph_runner passed to its
    __init__. The only defaults are the serialization pair, which report
    that the backend does not serialize.

    Methods:
      - capture_session(stream) — context wrapping the runner's outer
        capture loop; backends bind stream / pool and open per-backend
        capture flags here.
      - capture_one(shape_key, forward_fn, capture_inputs, post_warmup_hook)
        — record the replayable artifact for shape_key; one call per
        shape inside capture_session.
      - can_run(forward_batch, shape_key) — can this backend replay
        for the given batch at the given shape.
      - replay_session() — context wrapping replay-time model code;
        backends open the "we are replaying" flag here when they have
        one.
      - replay(shape_key, static_forward_batch, **kwargs) — invoke
        the captured artifact.
      - cleanup() — release pool and drop captured artifacts.
      - export_shape(shape_key, cache) — pointer-free ShapeArtifact of the
        shape recorded by capture_one (design section 6.7). Default:
        unsupported_shape(...) carrying _SERIALIZATION_REASON.
      - import_shape(shape_key, artifact, cache) — install a replayable
        artifact without running forward_fn; all-or-nothing. Default:
        raise GraphImportError.

    Notes:
      - The outer capture loop is runner-specific; it lives on the
        runner, not here.
      - capture_inputs optionally carries capture-time input owners that a
        backend must retain when its graph records their tensor addresses.
    """

    @abstractmethod
    def capture_session(self, stream: torch.cuda.Stream) -> Iterator[None]: ...

    @abstractmethod
    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn,
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None: ...

    @abstractmethod
    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool: ...

    @abstractmethod
    def replay_session(self) -> Iterator[None]: ...

    @abstractmethod
    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any: ...

    @abstractmethod
    def cleanup(self) -> None: ...

    # Backends that do not serialize inherit the two defaults below and only
    # override this reason (design section 9.3).
    _SERIALIZATION_REASON = "backend does not serialize"

    def export_shape(self, shape_key: ShapeKey, cache: GraphCache) -> ShapeArtifact:
        """Pointer-free artifact of the shape recorded by capture_one(shape_key)."""
        return unsupported_shape(
            shape_key, type(self).__name__, self._SERIALIZATION_REASON
        )

    def import_shape(
        self, shape_key: ShapeKey, artifact: ShapeArtifact, cache: GraphCache
    ) -> None:
        """Install a replayable artifact without running forward_fn. All-or-nothing:
        raise GraphImportError leaving no partial state."""
        raise GraphImportError(f"{shape_key}: {self._SERIALIZATION_REASON}")
