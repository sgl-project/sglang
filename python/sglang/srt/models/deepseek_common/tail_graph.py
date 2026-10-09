from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Optional, TypeAlias

import msgspec
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TailInputs(msgspec.Struct, frozen=True):
    residual: torch.Tensor
    pre: torch.Tensor
    positions: torch.Tensor
    input_ids: torch.Tensor
    input_ids_global: torch.Tensor
    hash_ids: Optional[torch.Tensor]
    swa_out_cache_loc: torch.Tensor

    def with_capacity(self, rows: int) -> TailInputs:
        return TailInputs(
            *(
                None if tensor is None else _pad_rows(tensor, rows)
                for tensor in msgspec.structs.astuple(self)
            )
        )


TailOutput: TypeAlias = tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]


class TailGraph(msgspec.Struct, frozen=True):
    graph: BreakableCUDAGraph
    inputs: TailInputs
    outputs: TailOutput
    capture_batch: Optional[ForwardBatch] = None
    q_pad_buffer: Optional[torch.Tensor] = None

    def replay(self, inputs: TailInputs) -> TailOutput:
        for dst, src in zip(
            msgspec.structs.astuple(self.inputs), msgspec.structs.astuple(inputs)
        ):
            assert (dst is None) == (src is None)
            if dst is not None:
                dst[: src.shape[0]].copy_(src)
                dst[src.shape[0] :].zero_()
        with graph_pool_replay_scope():
            self.graph.replay()
        rows = inputs.residual.shape[0]
        output, aux = self.outputs
        return tuple(t[:rows] for t in output), [t[:rows] for t in aux]


def tail_graph_sizes(max_rows: int) -> tuple[int, ...]:
    sizes = []
    rows = 1
    while rows < max_rows:
        sizes.append(rows)
        if rows >= 64 and rows * 3 // 2 < max_rows:
            sizes.append(rows * 3 // 2)
        rows *= 2
    return (*sizes, max_rows)


def select_tail_graph(graphs: dict[int, TailGraph], rows: int) -> Optional[TailGraph]:
    if rows <= 0:
        return None
    capacity = min((size for size in graphs if size >= rows), default=None)
    return None if capacity is None else graphs[capacity]


@torch.no_grad()
def capture_tail_graph(
    *,
    inputs: TailInputs,
    forward: Callable[[TailInputs], TailOutput],
    barrier: Callable[[], None],
    stream: torch.cuda.Stream,
) -> TailGraph:
    caller_stream = torch.cuda.current_stream()
    stream.wait_stream(caller_stream)
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.cuda.synchronize()
            barrier()
            forward(inputs)
        graph = BreakableCUDAGraph()
        with (
            graph_pool_capture_scope(),
            BreakableCUDAGraphCapture(
                cuda_graph=graph,
                pool=get_or_create_global_graph_memory_pool(torch.cuda),
                stream=stream,
                barrier_fn=barrier,
            ),
        ):
            outputs = forward(inputs)
    caller_stream.wait_stream(stream)
    return TailGraph(graph=graph, inputs=inputs, outputs=outputs)


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    assert tensor.shape[0] <= rows
    padded = tensor.new_zeros((rows, *tensor.shape[1:]))
    padded[: tensor.shape[0]].copy_(tensor)
    return padded


def make_capture_batch(
    prototype: ForwardBatch, *, rows: int, window: int
) -> ForwardBatch:
    batch = copy.copy(prototype)
    lengths = [min(window, rows - start) for start in range(0, rows, window)]
    device = prototype.input_ids.device
    batch.batch_size = len(lengths)
    batch.input_ids = prototype.input_ids.new_zeros(rows)
    batch.input_embeds = None
    batch.positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    batch.req_pool_indices = torch.arange(len(lengths), device=device)
    batch.seq_lens_cpu = torch.tensor(lengths)
    batch.seq_lens = batch.seq_lens_cpu.to(device)
    batch.orig_seq_lens = batch.seq_lens
    batch.seq_lens_sum = rows
    batch.out_cache_loc = prototype.out_cache_loc.new_zeros(rows)
    batch.extend_num_tokens = rows
    batch.extend_seq_lens_cpu = lengths
    batch.extend_seq_lens = batch.seq_lens
    batch.extend_prefix_lens_cpu = [0] * len(lengths)
    batch.extend_prefix_lens = torch.zeros_like(batch.seq_lens)
    batch.extend_start_loc = batch.seq_lens.cumsum(0) - batch.seq_lens
    batch.extend_logprob_start_lens_cpu = lengths
    batch.global_num_token_non_padded_cpu = rows
    batch.num_token_non_padded = (
        None
        if prototype.num_token_non_padded is None
        else torch.full_like(prototype.num_token_non_padded, rows)
    )
    return batch
