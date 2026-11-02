"""DSV4 NPU sources that run as eager breaks inside the breakable prefill graph.

The c4 indexer and the compressors consume host-built metadata (state block
tables sized from host lengths, lightning-indexer and compressor metadata), so
a captured segment cannot hold them: replay rebuilds that metadata for the live
batch. Like the sparse attention break in models/deepseek_v4.py, they run
between segments and read the replay batch and its metadata at call time.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
        C4IndexerAscendBackendMixin,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def in_prefill_graph(forward_batch: ForwardBatch) -> bool:
    # Same predicate as the attention break in models/deepseek_v4.py.
    return forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph()


@eager_on_graph
def bcg_dsv4_c4_indexer(
    *,
    backend: C4IndexerAscendBackendMixin,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    c4_indexer,
    skip_compressor: bool,
) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    # The static batch keeps bucket-sized positions; the indexer RoPE reads one
    # per live row.
    live_batch = copy.copy(forward_batch)
    live_batch.positions = forward_batch.positions[:num_tokens]
    backend.forward_c4_indexer_eager(
        x=x[:num_tokens],
        q_lora=q_lora[:num_tokens],
        forward_batch=live_batch,
        c4_indexer=c4_indexer,
        skip_compressor=skip_compressor,
    )


@eager_on_graph
def bcg_dsv4_compressor(*, x: torch.Tensor, compressor) -> None:
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_tokens = forward_batch.global_num_token_non_padded_cpu
    if num_tokens == 0:
        return
    compressor(x[:num_tokens], forward_batch)
