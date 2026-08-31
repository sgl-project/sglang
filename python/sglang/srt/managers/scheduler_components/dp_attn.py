from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import get_cp_strategy
from sglang.srt.layers.dp_attention import world_dp_gather_enabled
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.kv_cache_builder import uses_ssm_state
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import PrefillCudaGraphRunner
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


def _resolve_elastic_world_dp_size(
    dp_size: int,
    *,
    group: torch.distributed.ProcessGroup,
    local_num_tokens: int,
    local_forward_mode: int,
) -> int:
    if not world_dp_gather_enabled():
        return dp_size

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.layers.dp_attention import get_attention_dp_size

    live_dp_size = get_attention_dp_size()
    effective_ep_size = ElasticEPStateManager.get_effective_ep_size()
    world_size = torch.distributed.get_world_size(group)

    if live_dp_size != effective_ep_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size is out of sync: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} effective_ep_size={effective_ep_size} "
            f"world_size={world_size} server_args_dp_size={dp_size} "
            f"local_num_tokens={local_num_tokens} "
            f"local_forward_mode={local_forward_mode}"
        )
    if live_dp_size > world_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size exceeds WORLD size: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} world_size={world_size} "
            f"effective_ep_size={effective_ep_size}"
        )

    return live_dp_size


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_run_decode_cuda_graph: bool
    can_run_prefill_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info_cpu: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_run_decode_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_prefill_cuda_graph),
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_run_decode_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_prefill_cuda_graph
            ],
            device=device,
            dtype=dtype,
        )

    def finalize_local(self):
        """Populate gather-derived metadata from the sole attention-DP rank."""
        self.tp0_info_cpu = self._get_local_tensor(device="cpu").view(1, -1)
        self.global_num_tokens = [self.num_tokens]
        self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob]
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                self.tp0_info_cpu[:, 5].tolist()
            )

    def all_gather(
        self,
        device,
        group: torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        info_width = local_info_tensor.numel()
        # Inactive max_world_size slots must decode as IDLE. repeat() (not
        # expand().contiguous()) so the buffer never aliases fallback_tensor:
        # at world size 1 the expanded view is already contiguous, contiguous()
        # is a no-op, and the masked fallback writes below would then read and
        # write the same storage.
        global_info_tensor = fallback_tensor.repeat(
            self.dp_size, self.tp_size * self.cp_size, 1
        )

        if use_all_reduce:
            # Admission can expose different WORLD sizes; use fixed global slots.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, info_width)
            rank = torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor
            torch.distributed.all_reduce(
                global_info_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        else:
            torch.distributed.all_gather_into_tensor(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        tp_info = global_info_tensor.view(
            self.dp_size * self.tp_size * self.cp_size, info_width
        )
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = torch.ones(
                num_ranks_in_tp_info,
                dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

        # One D2H for every field: each `.item()` / `.tolist()` on a device
        # tensor is its own stream sync. Copy the whole tensor, not the
        # `[:, 0, :]` slice -- that slice is non-contiguous once
        # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
        tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
        self.tp0_info_cpu = tp0_info_cpu
        self.global_num_tokens = tp0_info_cpu[:, 0].tolist()
        self.global_num_tokens_for_logprob = tp0_info_cpu[:, 1].tolist()
        self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())
        self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())
        self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                tp0_info_cpu[:, 5].tolist()
            )


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_global_metadata=False,
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
    if not skip_global_metadata:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_decode_cuda_graph = mlp_sync_info.can_run_decode_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = mlp_sync_info.can_run_prefill_cuda_graph


def should_skip_scheduler_all_gather(dp_size: int) -> bool:
    """Return whether scheduler metadata is already local and rank-invariant.

    With one attention-DP rank there is no cross-DP state to reconcile.  The
    TP schedulers consume the same broadcast request stream, so gathering the
    identical batch mode, graph eligibility, and token counts only adds a
    device collective plus host synchronization.  Preserve the environment
    override for deployments that explicitly guarantee this invariant beyond
    DP1.
    """

    return dp_size == 1 or envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()


def _local_decode_cuda_graph_vote(
    *,
    local_batch: Optional[ScheduleBatch],
    disable_cuda_graph: bool,
) -> bool:
    """This rank's vote for the decode graph (min-reduced across dp ranks)."""
    if disable_cuda_graph:
        return False
    return (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    )


def _local_prefill_cuda_graph_vote(
    *,
    local_batch: Optional[ScheduleBatch],
    prefill_graph_runner,
    coordinated_prefill: bool,
    breakable_prefill: bool,
    spec_algorithm: SpeculativeAlgorithm,
    model_config,
) -> bool:
    """This rank's vote for the prefill graph (min-reduced across dp
    ranks). Extend/mixed batches vote their own replayability; a decode
    batch eligible for the decode->extend conversion votes as its 1-token-
    extend view, so the vote and the post-sync conversion always agree."""
    if local_batch is None or local_batch.forward_mode.is_idle():
        return True
    if not coordinated_prefill:
        return False

    mode = local_batch.forward_mode
    if mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
        num_tokens = local_batch.extend_num_tokens
        input_embeds = local_batch.input_embeds
        replace_embeds = local_batch.replace_embeds
        prefix_lens = local_batch.prefix_lens
        return_logprob = local_batch.return_logprob
    elif (
        mode.is_decode()
        # Conversion replays the breakable graphs only; full's fixed
        # request-slot geometry does not cover converted decode tails.
        and breakable_prefill
        # decode->extend conversion eligibility; needs the captured-graph
        # prefill runner, not the eager fallback.
        and isinstance(prefill_graph_runner, PrefillCudaGraphRunner)
        and spec_algorithm.is_none()
        and not local_batch.return_logprob
        # Grammar FSMs advance through the decode result path only.
        and not local_batch.has_grammar
        # Small-bucket BCG replays amplify the a2a EP logits drift (#30898)
        # into an accuracy loss.
        and get_moe_a2a_backend().is_none()
        # The converted view lacks prepare_for_extend's mamba-track fills.
        and not uses_ssm_state(model_config)
        # HiSparse decode has its own batch lifecycle and host-offloaded KV.
        and not get_memory().enable_hisparse
        and not get_exec().overlap.enable_two_batch_overlap
        and get_cp_strategy() is None
    ):
        num_tokens = local_batch.batch_size()
        input_embeds = None
        replace_embeds = None
        prefix_lens = None
        return_logprob = False
    else:
        return False

    if prefill_graph_runner is None:
        return True
    return prefill_graph_runner.can_replay_locally(
        batch_size=local_batch.batch_size(),
        num_tokens=num_tokens,
        input_embeds=input_embeds,
        replace_embeds=replace_embeds,
        prefix_lens=prefix_lens,
        is_target_verify=mode.is_target_verify(),
        capture_hidden_mode=None,
        return_logprob=return_logprob,
        lora_ineligible=prefill_graph_runner.enable_lora,
    )


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    model_runner: ModelRunner,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    dwdp: bool = False,
):
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
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

    can_run_decode_cuda_graph = _local_decode_cuda_graph_vote(
        local_batch=local_batch, disable_cuda_graph=disable_cuda_graph
    )
    breakable_prefill = check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    coordinated_prefill = breakable_prefill or check_cuda_graph_backend(
        Phase.PREFILL, Backend.FULL
    )
    prefill_graph_runner = (
        model_runner.prefill_cuda_graph_runner if coordinated_prefill else None
    )
    can_run_prefill_cuda_graph = _local_prefill_cuda_graph_vote(
        local_batch=local_batch,
        prefill_graph_runner=prefill_graph_runner,
        coordinated_prefill=coordinated_prefill,
        breakable_prefill=breakable_prefill,
        spec_algorithm=model_runner.spec_algorithm,
        model_config=model_runner.model_config,
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    use_world_group = world_dp_gather_enabled()
    if use_world_group:
        from sglang.srt.distributed.parallel_state import get_world_group

        world = get_world_group()
        group = torch.distributed.group.WORLD
        device = world.device
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if use_world_group:
        dp_size = _resolve_elastic_world_dp_size(
            dp_size,
            group=group,
            local_num_tokens=num_tokens,
            local_forward_mode=local_forward_mode,
        )
    skip_all_gather = should_skip_scheduler_all_gather(dp_size)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_run_decode_cuda_graph=can_run_decode_cuda_graph,
        can_run_prefill_cuda_graph=can_run_prefill_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    if dp_size == 1:
        mlp_sync_info.finalize_local()
    elif not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=use_world_group,
        )

    metadata_ready = mlp_sync_info.tp0_info_cpu is not None
    if metadata_ready:
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info_cpu[:, 4:6],
            )
        )

    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1 (and always under DWDP: ranks run independently)
        need_idle_batch = not dwdp and dp_size > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather,
            mlp_sync_info,
            require_mlp_tp_gather,
            skip_global_metadata=not metadata_ready,
        )

    # Set on `local_batch`, not `batch_to_gather`: for PREBUILT batches the
    # scheduler's `last_batch` is the prebuilt batch, not its inner idle batch.
    if local_batch is not None and metadata_ready:
        local_batch.recv_skipper_forward_mode = (
            SchedulerRecvSkipper.derive_forward_mode(
                mlp_sync_info.tp0_info_cpu[:, 5].tolist()
            )
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    model_runner: ModelRunner
    tp_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    offload_tags: set[str]
    ps: ParallelState
    model_config: ModelConfig
    enable_overlap: bool
    spec_algorithm: SpeculativeAlgorithm
    get_require_mlp_sync: Callable[[], bool]

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            model_runner=self.model_runner,
            dp_size=get_parallel().dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(),
            disable_overlap_schedule=get_schedule().disable_overlap_schedule,
            offload_tags=self.offload_tags,
            dwdp=get_parallel().dwdp_size > 1,
        )

    def maybe_prepare_mlp_sync_batch(
        self,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.get_require_mlp_sync() for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.get_require_mlp_sync():
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def maybe_convert_decode_to_extend(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """After the mlp-sync gather: convert an eligible decode batch to the
        extend view when a peer rank runs extend this step, so the step stays
        mode-homogeneous and every rank replays the extend graphs instead of
        all falling to eager."""
        if batch is None or not batch.forward_mode.is_decode():
            return batch
        # Global triggers from the gather. This rank's own eligibility (spec/
        # TBO/CP/logprob/replayability) is folded into the min-reduced vote:
        # if it failed, can_run_dp_prefill_cuda_graph is already False.
        if not batch.is_extend_in_batch:
            return batch
        if not batch.can_run_dp_prefill_cuda_graph:
            # The step is eager everywhere; eager decode beats eager mixed.
            return batch
        global_tokens = batch.global_num_tokens
        if (
            global_tokens is not None
            and len(global_tokens) > 1
            and min(global_tokens) == 0
        ):
            # An idle rank makes the prefill runner reject replay for every
            # rank (_has_inactive_dp_rank); converting would only trade eager
            # decode for eager mixed.
            return batch
        batch.convert_decode_to_extend()
        return batch

    def get_idle_batch(self) -> ScheduleBatch:
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
