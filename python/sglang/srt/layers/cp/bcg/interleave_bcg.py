"""Interleave BCG adapter for GLM-5.3 Flash text prefills.

Keep full-sequence KPool positions separate from the local DSA positions.
Graph row capacity is fixed while live token counts and prefix state can change.
"""

import torch

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.layers.cp.bcg.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.interleave import run_indexer_with_kv_gather
from sglang.srt.layers.cp.utils import cp_interleave_input_ids
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_parallel


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


class InterleaveCPBCGInput(PrefillCPBCGInput):
    @classmethod
    def create(cls, runner):
        result = super().create(runner)
        # Each local shard is CP-aligned before rank-major MoE all-gather.
        alignment = get_parallel().attn_cp_size ** 2
        capacity = (runner.max_num_tokens + alignment - 1) // alignment * alignment
        result.input_ids_global = torch.zeros(
            capacity, dtype=torch.int64, device=runner.device
        )
        # Graphs must read the live count from the same address on each replay.
        result.moe_local_token_count = torch.zeros(
            (), dtype=torch.int32, device=runner.device
        )
        return result

    def allows_replay(self, batch_size, num_tokens, prefix_lens, contains_mm_inputs):
        return batch_size > 0 and not contains_mm_inputs and prefix_lens is not None

    @staticmethod
    def supports(server_args):
        cfg = resolving_view(server_args)
        return (
            cfg.dp_size == 1
            and cfg.ep_size == 1
            and resolved_view(server_args).attn_cp_size > 1
            and "Glm5NextForConditionalGeneration"
            in model_config_of(server_args).hf_config.architectures
        )

    def _max_local_tokens(self, extend_seq_lens):
        size = get_parallel().attn_cp_size
        return (sum(int(length) for length in extend_seq_lens) + size - 1) // size

    def model_positions(self, forward_batch):
        return self.positions[: self.live_local_tokens]

    def prepare(self, runner, forward_batch, *, static_num_tokens, capture):
        global_positions = forward_batch.positions
        out_cache_loc = forward_batch.out_cache_loc
        super().prepare(
            runner, forward_batch, static_num_tokens=static_num_tokens, capture=capture
        )
        metadata = forward_batch.attn_cp_metadata
        # KDA projections/reorders retain the captured shape. Eager KPool
        # compression, KV writes and the output gather use the live lengths
        # produced by prepare_cp_forward, never the bucket's padded tail.
        metadata.graph_num_tokens = static_num_tokens
        self.moe_local_token_count.fill_(
            metadata.per_rank_logical_token[get_parallel().attn_cp_rank]
        )
        metadata.moe_local_token_count = self.moe_local_token_count
        forward_batch.out_cache_loc = out_cache_loc
        # KPool compresses consecutive global tokens; only the model argument
        # (DSA RoPE) is local. Do not replace ForwardBatch's global positions.
        forward_batch.positions = global_positions
        ids = cp_interleave_input_ids(
            forward_batch.input_ids[: forward_batch.extend_num_tokens], forward_batch
        )
        self.input_ids_global[: ids.shape[0]].copy_(ids)
        forward_batch.input_ids_global = self.input_ids_global[: ids.shape[0]]


def cp_interleave_indexer_rows(forward_batch):
    metadata = forward_batch.attn_cp_metadata
    rank = get_parallel().attn_cp_rank
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    return metadata.per_rank_actual_token[rank], logical[rank]
