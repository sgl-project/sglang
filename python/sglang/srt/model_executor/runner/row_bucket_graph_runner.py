"""Breakable CUDA graphs of a layer range that runs over a varying number of rows,
one graph per row bucket (a multiple of ``row_step``).

Ops of the body that read per-step state (KV slots, attention metadata) must be
eager breaks; they read the step's real row count from
``forward_batch.global_num_token_non_padded_cpu``.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.forward_batch_info import (
    compute_local_num_token_non_padded_cpu,
)
from sglang.srt.model_executor.forward_context import (
    forward_context,
    get_attn_backend,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

Body = Callable[[List[torch.Tensor], "ForwardBatch"], List[Optional[torch.Tensor]]]
# Wraps capture and replay with the body's batch and the bucket's static inputs.
BatchScope = Callable[["ForwardBatch", List[torch.Tensor]], ContextManager]


def _no_scope(batch: ForwardBatch, inputs: List[torch.Tensor]) -> ContextManager:
    return nullcontext()


def _pack_token_counts(local_rows: int, global_rows: int) -> int:
    assert 0 <= local_rows <= global_rows < 2**31, (local_rows, global_rows)
    return (global_rows << 32) | local_rows


class _Graph:
    __slots__ = (
        "graph",
        "num_token_non_padded",
        "global_num_token_non_padded",
        "packed_token_counts",
        "outputs",
        "batch",
    )

    def __init__(self, graph, num_token_non_padded, outputs, batch):
        self.graph = graph
        # MoE top-k masks rows at or past this count; captured by address.
        self.num_token_non_padded = num_token_non_padded
        self.global_num_token_non_padded = None
        self.packed_token_counts = None
        self.outputs = outputs
        # Kernels write buffers cached on the capture batch and attention metadata.
        self.batch = batch


class RowBucketGraphRunner:
    def __init__(self, *, row_step: int, max_rows: int, label: str) -> None:
        self.row_step = row_step
        self.max_rows = max_rows - max_rows % row_step
        self.label = label
        self._graphs: dict[tuple, _Graph] = {}
        self._static: dict[tuple, List[torch.Tensor]] = {}
        self._warm = False
        self.capture_seconds = 0.0
        self.capture_bytes = 0

    def ready(self, num_tokens: int) -> bool:
        """Whether graphs may run from this step on; every extend forward reports
        its token count here. Kernels that size cached workspaces lazily by row
        count reallocate them on a bigger step, which would leave earlier captures
        reading freed memory; a full prefill chunk has grown them all."""
        if not self._warm:
            chunk = get_schedule().chunked_prefill_size
            self._warm = num_tokens >= (chunk if chunk and chunk > 0 else 8192)
        return self._warm

    @property
    def warm(self) -> bool:
        return self._warm

    def bucket_rows(self, num_rows: int) -> Optional[int]:
        if num_rows == 0 or num_rows > self.max_rows:
            return None
        return -(-num_rows // self.row_step) * self.row_step

    def run(
        self,
        *,
        key: tuple,
        live: List[torch.Tensor],
        forward_batch: ForwardBatch,
        body: Body,
        context: Callable[[ForwardBatch], ContextManager],
        batch_scope: BatchScope = _no_scope,
        bucket_from: Optional[int] = None,
        out_rows: Optional[int] = None,
        global_num_rows: Optional[int] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Run ``body`` on the ``live`` rows from their bucket's graph, capturing
        it on first use. ``key`` names what besides row count and input shapes
        changes the recording; ``context`` and ``batch_scope`` wrap capture and
        replay with the breaks' and the body's batch; ``bucket_from`` picks the
        bucket from another row count. Returns views valid until the next replay,
        cut to ``out_rows`` rows. ``global_num_rows`` opts into a separate GLOBAL
        device count and localizes the LOCAL count for the bucket's TP shards."""
        num_rows = live[0].shape[0]
        out_rows = num_rows if out_rows is None else out_rows
        rows = self.bucket_rows(num_rows if bucket_from is None else bucket_from)
        assert rows is not None, (num_rows, bucket_from)
        if global_num_rows is not None:
            assert 0 <= global_num_rows <= rows, (global_num_rows, rows)
        # Inputs of one shape share their static buffers across keys.
        input_key = tuple((t.dtype, t.shape[1:]) for t in live)
        static = self._static_inputs(input_key, live)
        for buf, t in zip(static, live):
            buf[:num_rows].copy_(t)
        count_key = (
            forward_batch.num_token_non_padded is not None,
            global_num_rows is not None,
            forward_batch.attn_tp_sequence_sharded,
        )
        graph_key = (key, input_key, count_key)
        # The eager breaks read the step's real row count off this view.
        step_batch = copy.copy(forward_batch)
        step_batch.global_num_token_non_padded_cpu = num_rows
        with (
            (
                forward_context(get_forward_context(), forward_batch=step_batch)
                if has_forward_context()
                else nullcontext()
            ),
            context(step_batch),
        ):
            if (rows, graph_key) not in self._graphs:
                # Largest first, so smaller buckets reuse the pool memory the
                # larger ones took; a bucket below this step's rows cannot be
                # captured here (the breaks write every live row).
                for r in range(self.max_rows, rows - 1, -self.row_step):
                    if (r, graph_key) not in self._graphs:
                        self._graphs[(r, graph_key)] = self._capture(
                            r,
                            num_rows,
                            static,
                            forward_batch,
                            body,
                            batch_scope,
                            global_num_rows=global_num_rows,
                        )
            graph = self._graphs[(rows, graph_key)]
            if graph.packed_token_counts is not None:
                local_rows = compute_local_num_token_non_padded_cpu(
                    global_num_token_non_padded=global_num_rows,
                    num_tokens_per_dp=rows,
                    sharded=step_batch.attn_tp_sequence_sharded,
                )
                graph.packed_token_counts.fill_(
                    _pack_token_counts(local_rows, global_num_rows)
                )
                step_batch.global_num_token_non_padded = (
                    graph.global_num_token_non_padded
                )
                step_batch.num_token_non_padded = graph.num_token_non_padded
            elif graph.num_token_non_padded is not None:
                graph.num_token_non_padded.fill_(num_rows)
                step_batch.num_token_non_padded = graph.num_token_non_padded
            with batch_scope(step_batch, [buf[:rows] for buf in static]):
                with graph_pool_replay_scope():
                    graph.graph.replay()
        return [None if t is None else t[:out_rows] for t in graph.outputs]

    def _static_inputs(self, input_key, live) -> List[torch.Tensor]:
        static = self._static.get(input_key)
        if static is None:
            static = self._static[input_key] = [
                t.new_zeros((self.max_rows, *t.shape[1:])) for t in live
            ]
        return static

    def _capture(
        self,
        rows: int,
        num_rows: int,
        static: List[torch.Tensor],
        forward_batch: ForwardBatch,
        body: Body,
        batch_scope: BatchScope,
        *,
        global_num_rows: Optional[int] = None,
    ) -> _Graph:
        start = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        inputs = [buf[:rows] for buf in static]
        # The layers read the step's batch; the copy's count tensor is the graph's.
        # Host-side row counts read during capture cover the whole bucket.
        capture_batch = copy.copy(forward_batch)
        capture_batch.global_num_token_non_padded_cpu = rows
        packed_token_counts = None
        global_num_token_non_padded = None
        num_token_non_padded = None
        if global_num_rows is not None:
            assert 0 <= global_num_rows <= rows, (global_num_rows, rows)
            for count in (
                forward_batch.num_token_non_padded,
                forward_batch.global_num_token_non_padded,
            ):
                assert count is not None and count.dtype == torch.int32
                assert count.shape in ((), (1,)), count.shape
            assert (
                forward_batch.num_token_non_padded.device
                == forward_batch.global_num_token_non_padded.device
            )
            local_rows = compute_local_num_token_non_padded_cpu(
                global_num_token_non_padded=global_num_rows,
                num_tokens_per_dp=rows,
                sharded=capture_batch.attn_tp_sequence_sharded,
            )
            packed_token_counts = torch.full(
                (1,),
                _pack_token_counts(local_rows, global_num_rows),
                dtype=torch.int64,
                device=forward_batch.global_num_token_non_padded.device,
            )
            # CUDA is little-endian: LOCAL is the low32 word, GLOBAL the high32.
            token_counts = packed_token_counts.view(torch.int32)
            num_token_non_padded = token_counts[0].view_as(
                forward_batch.num_token_non_padded
            )
            global_num_token_non_padded = token_counts[1].view_as(
                forward_batch.global_num_token_non_padded
            )
            capture_batch.global_num_token_non_padded = global_num_token_non_padded
            capture_batch.num_token_non_padded = num_token_non_padded
        elif forward_batch.num_token_non_padded is not None:
            num_token_non_padded = torch.full_like(
                forward_batch.num_token_non_padded, num_rows
            )
            capture_batch.num_token_non_padded = num_token_non_padded
        with (
            (
                forward_context(get_forward_context(), forward_batch=capture_batch)
                if has_forward_context()
                else nullcontext()
            ),
            batch_scope(capture_batch, inputs),
        ):
            graph = self._record(
                lambda: body(inputs, capture_batch),
                rows,
                num_token_non_padded,
                capture_batch,
                start,
                free_before,
            )
            graph.global_num_token_non_padded = global_num_token_non_padded
            graph.packed_token_counts = packed_token_counts
            # The graph reads device tensors the scope put on the batch.
            graph.batch += (dict(vars(capture_batch)),)
        return graph

    def _record(
        self,
        body,
        rows,
        num_token_non_padded,
        capture_batch,
        start,
        free_before,
    ) -> _Graph:
        pool = get_or_create_global_graph_memory_pool(torch.cuda)
        set_graph_pool_id(pool)
        tp_group = get_parallel().tp_group
        graph = BreakableCUDAGraph()
        with graph_capture(stream=get_or_create_global_graph_capture_stream()) as ctx:
            # The warmups run eagerly; their KV writes repeat the replay's.
            for _ in range(2):
                torch.cuda.synchronize()
                tp_group.barrier()
                body()
            with (
                graph_pool_capture_scope(),
                BreakableCUDAGraphCapture(
                    cuda_graph=graph,
                    pool=pool,
                    stream=ctx.stream,
                    barrier_fn=tp_group.barrier,
                ),
            ):
                outputs = body()
        torch.cuda.current_stream().wait_stream(ctx.stream)
        torch.cuda.synchronize()
        recorded = _Graph(
            graph,
            num_token_non_padded,
            outputs,
            (capture_batch, get_attn_backend().forward_metadata),
        )
        seconds = time.perf_counter() - start
        used = free_before - torch.cuda.mem_get_info()[0]
        self.capture_seconds += seconds
        self.capture_bytes += used
        if get_parallel().tp_rank == 0:
            logger.info(
                "%s: captured %d rows (%d segments) in %.2f s, %.3f GiB; "
                "%d graphs, %.2f s, %.3f GiB",
                self.label,
                rows,
                len(graph._segments),
                seconds,
                used / 2**30,
                len(self._graphs) + 1,
                self.capture_seconds,
                self.capture_bytes / 2**30,
            )
        return recorded
