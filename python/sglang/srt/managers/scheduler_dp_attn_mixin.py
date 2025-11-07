from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size,
    attn_tp_size: int,
    tp_group,
    get_idle_batch,
    disable_cuda_graph: bool,
    spec_algorithm,
    speculative_num_draft_tokens,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None:
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        if local_batch.return_logprob:
            num_tokens_for_logprob = sum(
                # We should have at least 1 token for sample in every case.
                max(extend_len - logprob_start_len, 1)
                for logprob_start_len, extend_len in zip(
                    local_batch.extend_logprob_start_lens,
                    local_batch.extend_lens,
                )
            )
        else:
            # When return_logprob = False, only need last token per request
            num_tokens_for_logprob = local_batch.batch_size()

    if local_batch is None or local_batch.forward_mode.is_decode_or_idle():
        can_cuda_graph = 1
    else:
        can_cuda_graph = 0

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False

    tbo_preparer = TboDPAttentionPreparer()
    if len(offload_tags) == 0 and disable_overlap_schedule:
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_info = torch.tensor(
        [
            num_tokens,
            can_cuda_graph,
            num_tokens_for_logprob,
            is_extend_in_batch,
            *tbo_preparer.prepare_all_gather(
                local_batch,
            ),
        ],
        dtype=torch.int64,
        device=device,
    )
    global_info = torch.empty(
        (dp_size, attn_tp_size, 6),
        dtype=torch.int64,
        device=device,
    )
    torch.distributed.all_gather_into_tensor(
        global_info.flatten(),
        local_info,
        group=group,
    )
    global_num_tokens = global_info[:, 0, 0].tolist()
    can_cuda_graph = min(global_info[:, 0, 1].tolist())
    global_num_tokens_for_logprob = global_info[:, 0, 2].tolist()
    is_extend_in_batch = global_info[:, 0, 3].tolist()

    tbo_split_seq_index, global_forward_mode = tbo_preparer.compute_output(
        global_info[:, :, 4:6]
    )

    if local_batch is None and max(global_num_tokens) > 0:
        local_batch = get_idle_batch()

    if local_batch is not None:
        # TODO: handle the case when moe_dense_tp_size != 1
        if not require_mlp_tp_gather:
            local_batch.global_num_tokens = [num_tokens]
            local_batch.global_num_tokens_for_logprob = [num_tokens_for_logprob]
        else:
            local_batch.global_num_tokens = global_num_tokens
            local_batch.global_num_tokens_for_logprob = global_num_tokens_for_logprob
        local_batch.is_extend_in_batch = any(is_extend_in_batch)
        local_batch.tbo_split_seq_index = tbo_split_seq_index
        local_batch.global_forward_mode = global_forward_mode

        # Check forward mode for cuda graph
        if not disable_cuda_graph:
            local_batch.can_run_dp_cuda_graph = can_cuda_graph

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            spec_algorithm=self.spec_algorithm,
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
