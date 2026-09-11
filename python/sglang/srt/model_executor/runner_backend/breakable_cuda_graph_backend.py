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
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.graph_serialization.format import (
    ShapeArtifact,
    ShapeKeyRecord,
)
from sglang.srt.model_executor.graph_serialization.materializer import (
    GraphImportError,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.graph_serialization.format import (
        BreakSiteRecord,
        KernelIdentity,
        OutputSchema,
    )
    from sglang.srt.model_executor.graph_serialization.materializer import (
        GraphLoadContext,
        GraphSaveContext,
    )
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        warmup_out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            warmup_out = forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = BreakableCUDAGraph(self.deduped_cuda_graph)
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        size = shape_key.size
        if self._shared_output_buffer is None:
            self._shared_output_buffer = self._alloc_full_buffer(warmup_out, size)
        with (
            graph_pool_capture_scope(),
            BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=self._pool,
                stream=self._capture_stream,
                barrier_fn=self._tp_group.barrier,
            ),
        ):
            out = captured_fn()
            out_rows = self._output_rows(out, size)
            self._copy_output_to_buffer(out, self._shared_output_buffer, out_rows)

        stored = self._slice_output(self._shared_output_buffer, out_rows)
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        # CUDA graphs retain tensor addresses, not Python tensor lifetimes.
        self._capture_inputs[shape_key] = capture_inputs

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body, clamped to ``cap``.

        A body that shards or prunes its output along dim 0 returns fewer than
        ``cap`` rows; everything else returns exactly ``cap``.
        """
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        with graph_pool_replay_scope():
            self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._pool = None
        self._shared_output_buffer = None

    # -- serialization seam (design sections 6.7, 6.9, 9.2) -----------------

    @staticmethod
    def _segment_raw_graph(segment: Any) -> int:
        """The ``CUgraph`` handle of one ``BreakableCUDAGraph`` segment.

        Segments are ``torch.cuda.CUDAGraph`` objects (``raw_cuda_graph()``)
        or, under ``SGLANG_ENABLE_CUDA_GRAPH_DEDUP``, ``DedupedCudaGraph``
        wrappers that carry the handle as ``raw_graph``.
        """
        raw_cuda_graph = getattr(segment, "raw_cuda_graph", None)
        if callable(raw_cuda_graph):
            return int(raw_cuda_graph())
        raw_graph = getattr(segment, "raw_graph", None)
        if raw_graph is None:
            raise TypeError(
                f"BCG segment {type(segment).__name__} exposes neither "
                "raw_cuda_graph() nor raw_graph"
            )
        return int(raw_graph)

    def export_shape(self, shape_key: ShapeKey, ctx: GraphSaveContext) -> ShapeArtifact:
        """One ``SerializedGraph`` per captured segment, one
        ``BreakSiteRecord`` per break (design section 6.9), the shared output
        and the retained capture inputs (section 9.2). ``KeyError`` for a
        shape ``capture_one`` never recorded.

        A segment's ``raw_cuda_graph()`` exists only on a ``keep_graph=True``
        capture. ``BreakableCUDAGraph`` requests that only under
        ``SGLANG_ENABLE_CUDA_GRAPH_DEDUP`` today; the plain-path ``keep_graph``
        seam for segments (``runner_backend_utils/breakable_cuda_graph``) is
        not wired in this draft, so an export without dedup fails loudly
        there. Design section 6.7 specifies the ``keep_graph`` capture for
        the Full backend; the runners' ``keep_graph`` flag is ignored here.
        """
        graph = self._graphs[shape_key]
        segments = tuple(
            ctx.codec.encode(
                self._segment_raw_graph(segment),
                registry=ctx.registry,
                resolver=ctx.resolver,
                policy=ctx.policy,
                event_roles=ctx.event_roles,
            )
            for segment in graph._segments
        )
        breaks = self._describe_break_sites(graph)
        output, capture_inputs = self._describe_output(
            self._outputs[shape_key], self._capture_inputs.get(shape_key)
        )
        return ShapeArtifact(
            shape_key=ShapeKeyRecord.from_shape_key(shape_key),
            backend="breakable",
            graphs=segments,
            output=output,
            kernels=self._kernel_table(ctx),
            breaks=breaks,
            capture_inputs=capture_inputs,
        )

    def import_shape(
        self,
        shape_key: ShapeKey,
        artifact: ShapeArtifact,
        ctx: GraphLoadContext,
    ) -> None:
        """Materialize every segment (the all-or-nothing preamble, design
        section 6.7), then rebuild the break closures and assemble the
        ``BreakableCUDAGraph``.

        The preamble is implemented: an artifact from another backend, an
        artifact with no segments, or a segment the codec cannot materialize
        raise ``GraphImportError`` with ``_graphs`` / ``_outputs`` untouched.
        ``rebuild_break_fn`` and ``BreakableCUDAGraph.from_loaded(segments,
        break_fns)`` (design section 6.9) live in
        ``runner_backend_utils/breakable_cuda_graph`` and are not in this
        draft, so a successful preamble ends in ``NotImplementedError``.
        """
        try:
            if artifact.backend != "breakable":
                raise ValueError(
                    f"artifact was exported by backend {artifact.backend!r}, "
                    "not 'breakable'"
                )
            if not artifact.graphs:
                raise ValueError("the breakable artifact has no segments")
            loaded_segments = [
                ctx.codec.materialize(
                    graph,
                    kernels=artifact.kernels,
                    reloc=ctx.reloc,
                    resolver=ctx.resolver,
                    events=ctx.events,
                    device_ctx=ctx.device_ctx,
                )
                for graph in artifact.graphs
            ]
        except Exception as exc:
            raise GraphImportError(
                f"BreakableCudaGraphBackend.import_shape({shape_key}): {exc}"
            ) from exc
        raise NotImplementedError(
            "BreakableCudaGraphBackend.import_shape: rebuilding the break "
            "closures (rebuild_break_fn) and assembling BreakableCUDAGraph."
            f"from_loaded over the {len(loaded_segments)} materialized segment(s) "
            "with the shared output buffer is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 6.9"
        )

    @staticmethod
    def _describe_break_sites(graph: BreakableCUDAGraph) -> tuple[BreakSiteRecord, ...]:
        """One ``BreakSiteRecord`` per ``graph._break_fns`` entry from the
        ``BreakSiteInfo`` the ``eager_on_graph`` wrapper attaches: the site key
        by the three-way rule (``op:sglang::<name>`` / ``py:<module>:<qualname>``
        / ``model:<submodule path>:<method>``, fact 22), tensor arguments as
        region references, scalars by value, the bridge output as a
        ``bcg_bridge`` region (design section 6.9)."""
        raise NotImplementedError(
            "BreakableCudaGraphBackend._describe_break_sites: deriving "
            "BreakSiteRecord entries from the eager_on_graph break closures is "
            "not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 6.9"
        )

    @staticmethod
    def _describe_output(
        out: Any, capture_inputs: Any
    ) -> tuple[OutputSchema, tuple[OutputSchema, ...]]:
        """Describe the shared-output slice ``replay`` returns and the DP
        padding tensors retained as ``capture_inputs`` (design sections 6.7
        and 9.2, fact 13)."""
        raise NotImplementedError(
            "BreakableCudaGraphBackend._describe_output: describing the shared "
            "output buffer slice and the retained capture inputs as region "
            "references is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 6.7 and section 9.2"
        )

    @staticmethod
    def _kernel_table(ctx: GraphSaveContext) -> tuple[KernelIdentity, ...]:
        """The identity table ``KernelNode.identity`` indexes into, shared by
        every segment of the shape (design sections 6.5 and 6.6)."""
        raise NotImplementedError(
            "BreakableCudaGraphBackend._kernel_table: collecting the encoder's "
            "kernel identity table for the artifact is not implemented in this "
            "draft; see DESIGN_cuda_graph_serialization.md section 6.5 and "
            "section 6.6"
        )
