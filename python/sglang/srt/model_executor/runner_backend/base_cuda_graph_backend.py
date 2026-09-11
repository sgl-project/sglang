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

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.graph_serialization.format import ShapeArtifact
    from sglang.srt.model_executor.graph_serialization.materializer import (
        GraphLoadContext,
        GraphSaveContext,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BaseCudaGraphBackend(ABC):
    """Pure ABC: no state, no defaults. Each implementation owns its
    per-backend state and binds the handles it needs from the
    cuda_graph_runner passed to its __init__.

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
      - export_shape(shape_key, ctx) — serialize the artifact recorded by
        capture_one(shape_key) into a pointer-free ShapeArtifact
        (design section 6.7).
      - import_shape(shape_key, artifact, ctx) — install a replayable
        artifact without running forward_fn; all-or-nothing (design
        section 6.7).

    Notes:
      - The outer capture loop is runner-specific; it lives on the
        runner, not here.
      - capture_inputs optionally carries capture-time input owners that a
        backend must retain when its graph records their tensor addresses.
      - export_shape / import_shape are reached only through a
        GraphMaterializer other than CaptureOnlyMaterializer, so the
        default server path (``--cuda-graph-cache-mode off``) never calls
        them.
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

    @abstractmethod
    def export_shape(self, shape_key: ShapeKey, ctx: GraphSaveContext) -> ShapeArtifact:
        """Serialize the artifact recorded by capture_one(shape_key).

        Backends that cannot (tc_piecewise, NPU, XPU) return a ShapeArtifact
        whose graphs carry verdict='needs_recapture' (design section 6.7).
        """

    @abstractmethod
    def import_shape(
        self,
        shape_key: ShapeKey,
        artifact: ShapeArtifact,
        ctx: GraphLoadContext,
    ) -> None:
        """Install a replayable artifact WITHOUT running forward_fn.

        All-or-nothing: raise GraphImportError leaving no partial state;
        can_run/replay must then behave exactly as after capture_one
        (design section 6.7).
        """
