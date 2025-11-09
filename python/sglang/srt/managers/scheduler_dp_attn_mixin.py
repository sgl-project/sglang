from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size, 6),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        self.global_num_tokens = tp0_info[:, 0].tolist()
        self.global_num_tokens_for_logprob = tp0_info[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
):
    local_batch = batch

    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        local_batch.global_num_tokens = [mlp_sync_info.num_tokens]
        local_batch.global_num_tokens_for_logprob = [
            mlp_sync_info.num_tokens_for_logprob
        ]
    else:
        local_batch.global_num_tokens = mlp_sync_info.global_num_tokens
        local_batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    local_batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
    local_batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
    local_batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    local_batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt_extend():
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

    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt_extend()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False

    tbo_preparer = TboDPAttentionPreparer()
    if len(offload_tags) == 0 and disable_overlap_schedule:
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )
    mlp_sync_info.all_gather(device=device, group=group)

    mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
        tbo_preparer.compute_output(
            mlp_sync_info.tp0_info[:, 4:6],
        )
    )

    need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch and local_batch is None:
        local_batch = get_idle_batch()

    batch_to_gather = local_batch
    if need_idle_batch and local_batch.forward_mode.is_prebuilt_extend():
        # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
        local_batch.inner_idle_batch = get_idle_batch()
        batch_to_gather = local_batch.inner_idle_batch

    if local_batch is not None:
        _update_gather_batch(batch_to_gather, mlp_sync_info, require_mlp_tp_gather)

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
