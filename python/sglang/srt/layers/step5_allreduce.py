"""Step5 communication with deferred MoE finalization."""

from functools import partial
from typing import Optional

import torch
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    in_the_same_node_as,
)
from sglang.srt.layers.flashinfer_comm_fusion import (
    FlashInferWorkspaceManager,
    resolve_flashinfer_allreduce_fusion_backend,
)
from sglang.srt.layers.layer_boundary import (
    DeferredFinalize,
    FfnInputFusion,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    FUSE_ALLREDUCE_MAX_BATCH_SIZE,
    flashinfer_ar_fusion_applies as apply_flashinfer_allreduce_fusion,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmDeferredFinalizeOutput,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_exec, get_parallel, max_decode_batch_tokens


@cache_once
def _get_comm() -> Optional[CustomAllReduceV2]:
    parallel = get_parallel()
    backend = resolve_flashinfer_allreduce_fusion_backend()
    topology = (parallel.tp_size, parallel.attn_tp_size, parallel.moe_ep_size)
    a2a_backend = get_moe_a2a_backend()
    supported = (
        (((4, 4, 4), "trtllm"),)
        if a2a_backend.is_megamoe()
        else (((8, 8, 8), "mnnvl"), ((4, 4, 1), "trtllm"))
    )
    if (
        torch.version.hip is not None
        or torch.cuda.get_device_capability() != (10, 3)
        or (topology, backend) not in supported
        or parallel.attn_dp_size != 1
        or parallel.pp_size != 1
        or get_exec().comm.enable_symm_mem
        or not (a2a_backend.is_none() or a2a_backend.is_megamoe())
    ):
        return None
    group = get_tp_group()
    moe_group = get_moe_ep_group() if parallel.moe_ep_size > 1 else get_moe_tp_group()
    if (
        not isinstance(group.ca_comm, CustomAllReduceV2)
        or group.ca_comm.disabled
        or not group.ca_comm.has_multicast
        or group.ranks != get_attn_tp_group().ranks
        or group.ranks != moe_group.ranks
        or not all(in_the_same_node_as(group.cpu_group, source_rank=0))
    ):
        return None
    # Keep phase counters separate from ordinary all-reduce. Only the small
    # decode/verify path uses this fusion; larger batches retain their backend.
    max_rows = 32 if parallel.tp_size == 4 else 4
    comm = CustomAllReduceV2(
        group.cpu_group,
        torch.device("cuda", torch.cuda.current_device()),
        max_push_size=max_rows * 4096 * 2,
        max_push_blocks=max_rows,
        max_pull_size=0,
    )
    if comm.disabled or not comm.has_multicast:
        return None
    all_reduce_fusion.register_comm(comm.obj)
    return comm


def _verify_workspace_rows() -> int:
    return min(max_decode_batch_tokens(), FUSE_ALLREDUCE_MAX_BATCH_SIZE)


@cache_once
def _get_verify_workspace():
    comm = _get_comm()
    if comm is None or comm.world_size != 4:
        return None

    import torch.distributed as dist

    group = get_tp_group()
    # The package version alone does not identify the RMSNorm arithmetic fix.
    try:
        from flashinfer.jit.comm import gen_trtllm_mnnvl_comm_module

        module = gen_trtllm_mnnvl_comm_module().build_and_load()
        version = getattr(module, "step5_tp4_rmsnorm_order_version", None)
        supported = version is not None and version() == 2
    except (ImportError, AttributeError, RuntimeError):
        supported = False
    ready = torch.tensor(int(supported), dtype=torch.int32, device="cpu")
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if not ready.item():
        return None

    manager = FlashInferWorkspaceManager()
    manager.initialize(
        world_size=4,
        rank=group.rank_in_group,
        max_token_num=_verify_workspace_rows(),
        hidden_dim=4096,
        backend="mnnvl",
        dtype=torch.bfloat16,
        use_oneshot=True,
        device_group=group.device_group,
        cpu_group=group.cpu_group,
    )
    ready.fill_(int(manager.initialized))
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if not ready.item():
        manager.cleanup()
        raise RuntimeError("Step5 verify communication workspace initialization failed")
    return manager


class Step5DeferredMoE(DeferredFinalize, frozen=True):
    routed: FlashInferTrtllmDeferredFinalizeOutput
    shared: torch.Tensor

    def complete(self):
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        return tensor_model_parallel_all_reduce(
            finalize_flashinfer_trtllm_deferred_output(
                self.routed, self.shared, output_dtype=torch.bfloat16, round_routed=True
            )
        )


class Step5BoundaryFusions:
    def __init__(self, input_layernorm):
        weight = input_layernorm.weight
        self.allreduce_comm = (
            _get_comm()
            if weight.is_cuda
            and weight.dtype == torch.bfloat16
            and weight.numel() == 4096
            else None
        )
        self.max_fused_rows = (
            32
            if self.allreduce_comm is not None and self.allreduce_comm.world_size == 4
            else 4
        )
        self.attn_allreduce_indices = self.attn_allreduce_weights = None
        if (
            self.allreduce_comm is not None
            and self.allreduce_comm.world_size == 4
            and not get_exec().deterministic.enable_deterministic_inference
            and not get_exec().comm.enable_quant_communications
        ):
            # Identity routing reuses the same fused push/residual/norm kernel.
            # Allocate once before capture and share the communicator's counters.
            self.attn_allreduce_indices = torch.arange(
                self.max_fused_rows, device=weight.device, dtype=torch.int32
            )
            self.attn_allreduce_weights = torch.ones(
                (self.max_fused_rows, 1), device=weight.device, dtype=torch.float32
            )
        self.verify_workspace = (
            _get_verify_workspace()
            if self.attn_allreduce_indices is not None
            and _verify_workspace_rows() > self.max_fused_rows
            else None
        )
        self.verify_workspace_rows = (
            self.verify_workspace.max_token_num
            if self.verify_workspace is not None
            else 0
        )

    def _can_use_verify_workspace(self, hidden_states, residual):
        return (
            self.verify_workspace is not None
            and not get_attn_tp_context().input_scattered
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.shape[1] == 4096
            and self.max_fused_rows
            < hidden_states.shape[0]
            <= self.verify_workspace_rows
            and hidden_states.is_contiguous()
            and residual is not None
            and residual.shape == hidden_states.shape
            and residual.dtype == hidden_states.dtype
            and residual.is_contiguous()
            and apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        )

    def _verify_allreduce_norm(self, hidden_states, residual, layernorm):
        from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

        norm_out = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(residual)
        allreduce_fusion(
            input=hidden_states,
            workspace=self.verify_workspace.workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            norm_out=norm_out,
            residual_out=residual_out,
            residual_in=residual,
            rms_gamma=layernorm.gemma_weight,
            rms_eps=layernorm.variance_epsilon,
            launch_with_pdl=True,
            trigger_completion_at_end=False,
        )
        return norm_out, residual_out

    def _ffn_input(self, plan, hidden_states, residual, forward_batch):
        if self._can_use_verify_workspace(hidden_states, residual):
            return self._verify_allreduce_norm(hidden_states, residual, plan.norm)
        if (
            self.attn_allreduce_indices is not None
            and not get_attn_tp_context().input_scattered
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and 0 < hidden_states.shape[0] <= self.max_fused_rows
            and hidden_states.shape[1] == 4096
            and apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
            and hidden_states.is_contiguous()
            and residual is not None
            and residual.shape == hidden_states.shape
            and residual.dtype == hidden_states.dtype
            and residual.is_contiguous()
        ):
            rows = hidden_states.shape[0]
            return all_reduce_fusion.moe_finalize_all_reduce_residual_rmsnorm(
                hidden_states,
                self.attn_allreduce_indices[:rows],
                self.attn_allreduce_weights[:rows],
                None,
                residual,
                plan.norm.gemma_weight,
                plan.norm.variance_epsilon,
                world_size=4,
                cluster_size=4,
            )
        return None

    def can_fuse_moe(
        self, shared: torch.Tensor, routed: FlashInferTrtllmDeferredFinalizeOutput
    ) -> bool:
        return (
            self.allreduce_comm is not None
            and shared.dtype == torch.bfloat16
            and shared.ndim == 2
            and 0 < shared.shape[0] <= self.max_fused_rows
            and shared.shape[1] == 4096
            and shared.is_contiguous()
            and routed.gemm2_out.dtype == torch.bfloat16
            and routed.expert_weights.dtype == torch.float32
            and routed.expert_weights.shape == (shared.shape[0], routed.top_k)
        )

    def defer_combined_moe(
        self, routed: torch.Tensor, shared: torch.Tensor
    ) -> Optional[Step5DeferredMoE]:
        if self.attn_allreduce_indices is None or routed.shape != shared.shape:
            return None
        rows = routed.shape[0]
        if not 0 < rows <= self.max_fused_rows:
            return None
        deferred = FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=routed,
            expert_weights=self.attn_allreduce_weights[:rows],
            expanded_idx_to_permuted_idx=self.attn_allreduce_indices[:rows],
            top_k=1,
        )
        if not (routed.is_contiguous() and self.can_fuse_moe(shared, deferred)):
            return None
        return Step5DeferredMoE(deferred, shared)

    def attn_input_fusions(self, plan):
        return (partial(self._attn_input, plan),)

    def ffn_input_fusions(self, plan):
        if (
            self.allreduce_comm is not None
            and TokenAxis.ATTN_TP not in plan.incoming_residual_rows.sharded
            and get_parallel().attn_tp_size == get_parallel().tp_size
        ):
            return (
                FfnInputFusion(
                    completes=SumGroup.ATTN_TP, run=partial(self._ffn_input, plan)
                ),
            )
        return ()

    def can_defer_all_reduce(self, plan, forward_batch):
        return (
            self.allreduce_comm is not None
            and not get_attn_tp_context().input_scattered
        )

    def can_defer_finalize(self, plan, forward_batch):
        return (
            not plan.terminal
            and self.can_defer_all_reduce(plan, forward_batch)
            and 0 < forward_batch.input_ids.shape[0] <= self.max_fused_rows
        )

    def _attn_input(self, plan, owed, residual, forward_batch, post_residual_addition):
        if post_residual_addition is not None or residual is None:
            return None
        if isinstance(owed, Step5DeferredMoE):
            if not self.can_fuse_moe(owed.shared, owed.routed):
                return None
            routed = owed.routed
            return all_reduce_fusion.moe_finalize_all_reduce_residual_rmsnorm(
                routed.gemm2_out,
                routed.expanded_idx_to_permuted_idx,
                routed.expert_weights,
                owed.shared,
                residual,
                plan.norm.gemma_weight,
                plan.norm.variance_epsilon,
                world_size=self.allreduce_comm.world_size,
                cluster_size=4,
            )
        if (
            isinstance(owed, UnreducedOutput)
            and owed.group is get_tp_group()
            and self._can_use_verify_workspace(owed.partial, residual)
        ):
            return self._verify_allreduce_norm(owed.partial, residual, plan.norm)
        return None
