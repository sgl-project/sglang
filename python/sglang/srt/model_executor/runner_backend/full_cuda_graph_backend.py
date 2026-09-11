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
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
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


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.

    ``keep_graph`` makes ``capture_one`` build
    ``torch.cuda.CUDAGraph(keep_graph=True)`` and instantiate explicitly, so
    the captured ``CUgraph`` survives and ``export_shape`` can encode
    ``raw_cuda_graph()`` (design section 6.7). ``resolve_decode_backend`` /
    ``resolve_prefill_backend`` set it from ``keeps_raw_graphs(plan)``, i.e.
    exactly when the runner's graph-serialization plan is enabled; the
    default server path constructs ``CUDAGraph()`` exactly as before.
    """

    # Class-level default so an instance built without __init__ (tests) reads
    # the same default the constructor installs.
    _keep_graph: bool = False

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        keep_graph: bool = False,
    ) -> None:
        self._keep_graph = keep_graph
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # When per-bs capture traces are enabled (--enable-profile-cuda-graph +
        # SGLANG_GRAPH_BATCH_CAPTURE), the runner created a scheduled
        # torch profiler (wait=2, active=1) and exposed it as _profiler. We step()
        # past the two warmup runs so only the capture run is recorded, and each
        # batch size produces its own trace via the profiler's on_trace_ready.
        # With --enable-profile-cuda-graph alone the runner leaves _profiler None
        # (its unscheduled profiler records the whole capture in one pass), so no
        # stepping happens here.
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if profiler is not None:
                profiler.step()
            if post_warmup_hook is not None:
                post_warmup_hook()

        # keep_graph=True only when serialization asked for it (design section
        # 6.7); the default path constructs CUDAGraph() with no arguments.
        if self._keep_graph:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
        else:
            graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        with (
            graph_pool_capture_scope(),
            graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream),
        ):
            out = forward_fn()

        if profiler is not None:
            profiler.step()

        if self._keep_graph:
            # keep_graph=True defers instantiation past capture_end; do it now
            # so the first replay pays nothing extra and raw_cuda_graph() stays
            # valid for export_shape (design section 6.7).
            graph.instantiate()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
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
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None

    # -- serialization seam (design sections 6.7, 9.1) ----------------------

    def export_shape(self, shape_key: ShapeKey, ctx: GraphSaveContext) -> ShapeArtifact:
        """One ``SerializedGraph`` plus the output schema for ``shape_key``.

        ``KeyError`` for a shape ``capture_one`` never recorded. The graph
        must expose ``raw_cuda_graph()``: a capture made with
        ``keep_graph=True`` (what the backend resolvers request whenever the
        plan is enabled) or a loaded graph (design section 6.7).
        """
        graph = self._graphs[shape_key]
        raw = graph.raw_cuda_graph()
        serialized = ctx.codec.encode(
            raw,
            registry=ctx.registry,
            resolver=ctx.resolver,
            policy=ctx.policy,
            event_roles=ctx.event_roles,
        )
        output = self._describe_output(self._outputs[shape_key])
        return ShapeArtifact(
            shape_key=ShapeKeyRecord.from_shape_key(shape_key),
            backend="full",
            graphs=(serialized,),
            output=output,
            kernels=self._kernel_table(ctx),
        )

    def import_shape(
        self,
        shape_key: ShapeKey,
        artifact: ShapeArtifact,
        ctx: GraphLoadContext,
    ) -> None:
        """Install ``artifact`` for ``shape_key`` without a forward.

        All-or-nothing (design section 6.7): the graph is materialized and
        the output rebuilt into locals first; ``_graphs`` / ``_outputs`` are
        written only after both succeeded, and any failure is raised as
        ``GraphImportError`` chained to its cause with the tables untouched.
        """
        try:
            if artifact.backend != "full":
                raise ValueError(
                    f"artifact was exported by backend {artifact.backend!r}, not 'full'"
                )
            if len(artifact.graphs) != 1:
                raise ValueError(
                    "the full backend expects exactly one graph per shape, "
                    f"the artifact has {len(artifact.graphs)}"
                )
            loaded = ctx.codec.materialize(
                artifact.graphs[0],
                kernels=artifact.kernels,
                reloc=ctx.reloc,
                resolver=ctx.resolver,
                events=ctx.events,
                device_ctx=ctx.device_ctx,
            )
            output = self._rebuild_output(artifact.output, ctx.reloc)
        except Exception as exc:
            raise GraphImportError(
                f"FullCudaGraphBackend.import_shape({shape_key}): {exc}"
            ) from exc
        self._graphs[shape_key] = loaded
        self._outputs[shape_key] = output

    @staticmethod
    def _describe_output(out: Any) -> OutputSchema:
        """Describe the object ``replay`` returns as an ``OutputSchema``:
        ``LogitsProcessorOutput`` fields, ``PPProxyTensors`` entries and bare
        tensors become region references with explicit shape, stride and
        dtype (design section 6.7, fact 13)."""
        raise NotImplementedError(
            "FullCudaGraphBackend._describe_output: describing the replay "
            "output (LogitsProcessorOutput / PPProxyTensors / tensor views as "
            "region references) is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 6.7"
        )

    @staticmethod
    def _rebuild_output(schema: OutputSchema, reloc: Any) -> Any:
        """Rebuild the replay output from its schema: every tensor is a
        ``tensor_from_pointer`` view at ``reloc.rebase(ref)`` with the saved
        shape, stride and dtype passed explicitly (design section 6.7,
        fact 13)."""
        raise NotImplementedError(
            "FullCudaGraphBackend._rebuild_output: rebuilding the replay output "
            "from tensor_from_pointer views over relocated regions is not "
            "implemented in this draft; see DESIGN_cuda_graph_serialization.md "
            "section 6.7 (fact 13)"
        )

    @staticmethod
    def _kernel_table(ctx: GraphSaveContext) -> tuple[KernelIdentity, ...]:
        """The identity table ``KernelNode.identity`` indexes into. The
        encoder assigns the indices while it walks the nodes (design sections
        6.5 and 6.6); a full build hands the table back through the codec."""
        raise NotImplementedError(
            "FullCudaGraphBackend._kernel_table: collecting the encoder's "
            "kernel identity table for the artifact is not implemented in this "
            "draft; see DESIGN_cuda_graph_serialization.md section 6.5 and "
            "section 6.6"
        )
