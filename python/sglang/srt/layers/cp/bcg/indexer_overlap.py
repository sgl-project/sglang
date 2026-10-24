"""Stable BCG buffers for the shared CP KV/indexer overlap region."""

from sglang.srt.layers.cp.bcg.interleave_bcg import cp_interleave_indexer_rows
from sglang.srt.layers.cp.indexer_overlap import run_indexer_with_kv_gather
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _indexer_with_kv(
    indexer, x, q_lora, positions, output, layer_id, kv_local, kv_output
):
    # Resolve counts, prefix-cache state and destinations on every replay.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    n, logical = cp_interleave_indexer_rows(forward_batch)
    if (
        not 0
        <= logical
        <= n
        <= min(x.shape[0], q_lora.shape[0], positions.shape[0], kv_local.shape[0])
    ):
        raise ValueError("Invalid CP indexer input token counts")
    topk, kv = run_indexer_with_kv_gather(
        indexer, x[:n], q_lora[:n], positions[:n], forward_batch, layer_id, kv_local[:n]
    )
    kv_rows = sum(forward_batch.extend_seq_lens_cpu)
    if kv.shape[0] != kv_rows or kv_rows > kv_output.shape[0]:
        raise ValueError("CP KV gather returned an unexpected token count")
    if (
        topk is None
        or topk.ndim != 2
        or not logical <= topk.shape[0] <= n
        or topk.shape[1] != output.shape[1]
    ):
        raise ValueError("CP indexer returned an unexpected top-k shape")
    kv_output[:kv_rows].copy_(kv)
    kv_output[kv_rows:].zero_()
    output[:logical].copy_(topk[:logical])
    output[logical:].fill_(-1)


def _capture_stub(indexer, x, q_lora, positions, output, layer_id, kv_local, kv_output):
    output.fill_(-1)
    kv_output.zero_()


bcg_indexer_with_kv = eager_on_graph(True, capture_stub=_capture_stub)(_indexer_with_kv)
