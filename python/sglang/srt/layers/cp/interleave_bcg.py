"""Experimental, exact-shape interleave BCG adapter for GLM-5.3 Flash.

Keep full-sequence KPool positions separate from the local DSA positions.
Only single-request, prefix-free, text batches with an exact captured token
count replay. Other batches retain the existing eager CP path, including MM.
"""

import torch

from sglang.srt.arg_groups.overrides import (
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.layers.cp.bcg import PrefillCPBCGInput
from sglang.srt.layers.cp.utils import cp_interleave_input_ids
from sglang.srt.runtime_context import get_parallel


def supports_interleave_bcg(server_args):
    cfg = resolving_view(server_args)
    resolved = resolved_view(server_args)
    return (
        cfg.enable_prefill_cp
        and cfg.pp_size == 1
        and cfg.dp_size == 1
        and cfg.ep_size == 1
        and resolved.attn_cp_size > 1
        and resolved.attn_cp_size == cfg.tp_size
        and "Glm5NextForConditionalGeneration"
        in model_config_of(server_args).hf_config.architectures
    )


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
        return result

    def allows_replay(self, batch_size, num_tokens, prefix_lens, contains_mm_inputs):
        return (
            batch_size == 1
            and num_tokens in self.bucket_local_tokens
            and not contains_mm_inputs
            and prefix_lens is not None
            and not any(prefix_lens)
        )

    def required_local_tokens(self, extend_seq_lens):
        if extend_seq_lens is None or len(extend_seq_lens) != 1:
            return None
        size = get_parallel().attn_cp_size
        rows = (int(extend_seq_lens[0]) + size - 1) // size
        return (rows + size - 1) // size * size

    def model_positions(self, forward_batch):
        return self.positions[: self.live_local_tokens]

    def prepare(self, runner, forward_batch, *, static_num_tokens, capture):
        global_positions = forward_batch.positions
        super().prepare(
            runner, forward_batch, static_num_tokens=static_num_tokens, capture=capture
        )
        # KPool compresses consecutive global tokens; only the model argument
        # (DSA RoPE) is local. Do not replace ForwardBatch's global positions.
        forward_batch.positions = global_positions
        ids = cp_interleave_input_ids(forward_batch.input_ids, forward_batch)
        self.input_ids_global[: ids.shape[0]].copy_(ids)
        forward_batch.input_ids_global = self.input_ids_global[: ids.shape[0]]


def cp_interleave_indexer_rows(forward_batch):
    metadata = forward_batch.attn_cp_metadata
    rank = get_parallel().attn_cp_rank
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    return metadata.per_rank_actual_token[rank], logical[rank]
