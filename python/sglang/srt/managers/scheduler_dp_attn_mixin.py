from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()

# int32 suffices for all 6 packed fields (token counts, bools, forward-mode enum)
# and halves the all_gather wire size vs int64.
_INFO_DTYPE = torch.int32


@dataclass
class _MLPSyncBuffers:
    """Persistent pre-allocated tensors reused across steps.

    Avoids per-step CUDA/CPU malloc for the three tensors that are identical
    in shape and device across every call to prepare_mlp_sync_batch_raw.
    One instance per process (module-level singleton, see _get_or_init_mlp_sync_buffers).
    """

    local_tensor: torch.Tensor  # shape (6,) – filled in-place each step
    global_tensor: torch.Tensor  # shape (dp_size, tp_size*cp_size, 6)
    global_tensor_flat: torch.Tensor  # contiguous view of global_tensor for all_gather
    fallback_tensor: torch.Tensor  # shape (6,) – constant fallback for inactive ranks
    has_inactive_ranks: bool  # pre-computed; False in homogeneous deployments


_mlp_sync_buffers: Optional[_MLPSyncBuffers] = None
# Cached across steps; TboDPAttentionPreparer carries only within-step state
# (set by prepare_all_gather, consumed by compute_output) so reuse is safe.
_tbo_preparer: Optional[TboDPAttentionPreparer] = None


def _get_tbo_preparer() -> TboDPAttentionPreparer:
    global _tbo_preparer
    if _tbo_preparer is None:
        _tbo_preparer = TboDPAttentionPreparer()
    return _tbo_preparer


def _get_or_init_mlp_sync_buffers(
    dp_size: int,
    tp_size: int,
    cp_size: int,
    device,
    dtype: torch.dtype,
) -> _MLPSyncBuffers:
    global _mlp_sync_buffers
    if _mlp_sync_buffers is None:
        tp_group = get_tp_group()
        active_ranks = (
            tp_group.active_ranks_cpu if device == "cpu" else tp_group.active_ranks
        )
        has_inactive = bool((active_ranks == 0).any().item())

        fallback_values = [
            0,  # num_tokens
            0,  # num_tokens_for_logprob
            1,  # can_cuda_graph
            0,  # is_extend_in_batch
            1,  # local_can_run_tbo
            ForwardMode.IDLE.value,  # local_forward_mode
        ]

        # Pin CPU tensors so that any future async H2D copy (e.g. to GPU buffer)
        # can use DMA without an extra staging bounce.
        use_pin = device == "cpu" and torch.cuda.is_available()
        global_tensor = torch.empty(
            (dp_size, tp_size * cp_size, 6),
            dtype=dtype,
            device=device,
            pin_memory=use_pin,
        )
        _mlp_sync_buffers = _MLPSyncBuffers(
            local_tensor=torch.zeros(6, dtype=dtype, device=device),
            global_tensor=global_tensor,
            global_tensor_flat=global_tensor.flatten(),
            fallback_tensor=torch.tensor(fallback_values, dtype=dtype, device=device),
            has_inactive_ranks=has_inactive,
        )
    return _mlp_sync_buffers


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info: torch.Tensor = None
    cpu_tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def all_gather(
        self,
        device,
        group: torch.distributed.ProcessGroup,
        buffers: _MLPSyncBuffers,
    ):
        buf = buffers.local_tensor
        buf[0] = self.num_tokens
        buf[1] = self.num_tokens_for_logprob
        buf[2] = int(self.can_cuda_graph)
        buf[3] = int(self.is_extend_in_batch)
        buf[4] = int(self.local_can_run_tbo)
        buf[5] = self.local_forward_mode

        torch.distributed.all_gather_into_tensor(
            buffers.global_tensor_flat,
            buffers.local_tensor,
            group=group,
        )

        # Fast path: skip inactive-rank fixup when all ranks are active (common case).
        if buffers.has_inactive_ranks:
            tp_group = get_tp_group()
            tp_active_ranks = (
                tp_group.active_ranks_cpu if device == "cpu" else tp_group.active_ranks
            )
            tp_info = buffers.global_tensor.view(
                self.dp_size * self.tp_size * self.cp_size, 6
            )
            tp_info[tp_active_ranks == 0] = buffers.fallback_tensor

        tp0_info = buffers.global_tensor[:, 0, :]
        self.tp0_info = tp0_info

        cpu_tp0_info = tp0_info if device == "cpu" else tp0_info.cpu()
        self.cpu_tp0_info = cpu_tp0_info
        self.global_num_tokens = cpu_tp0_info[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_tp0_info[:, 1].tolist()
        self.can_cuda_graph = bool(cpu_tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(cpu_tp0_info[:, 3].max().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                cpu_tp0_info[:, 5].tolist()
            )


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    if len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    tbo_preparer = _get_tbo_preparer()
    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    if not skip_all_gather:
        buffers = _get_or_init_mlp_sync_buffers(
            dp_size, attn_tp_size, attn_cp_size, device, _INFO_DTYPE
        )
        mlp_sync_info.all_gather(device=device, group=group, buffers=buffers)

        # cpu_tp0_info[:, 4:6] is already on CPU – no extra D2H in compute_output.
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.cpu_tp0_info[:, 4:6],
            )
        )

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def maybe_prepare_mlp_sync_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

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
