"""Cake (FlashInfer) backends for the ``communication`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels`, which import FlashInfer only
when a kernel is actually called. All entries are multi-rank: the process
group, symmetric / IPC workspaces and cross-rank launch ordering are owned by
``sglang.srt.layers.communication``; callers gate on the adapters'
``supports_*`` checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

_BLACKWELL_DC = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})
_TP = "sglang.kernels.cake_kernels.communication:"
_A2A = "sglang.kernels.cake_kernels.communication_moe_a2a:"
_KIMI = "sglang.kernels.cake_kernels.communication_kimi_k3_tp12_tail:"


def _register(
    name: str,
    target: str,
    *,
    description: str,
    dtypes: tuple[str, ...] = (),
    in_place: bool = False,
    signature: str,
) -> None:
    register_kernel(
        KernelSpec(
            op=f"communication.{name}",
            backend=KernelBackend.FLASHINFER,
            target=target,
            capabilities=_BLACKWELL_DC,
            format_signature=FormatSignature(
                supported_dtypes=dtypes,
                in_place=in_place,
                description=signature,
            ),
            description=description,
        )
    )


# --- all-gather matmul (NVSHMEM symmetric memory, world size 2/4/8) ----------
_register(
    "all_gather_matmul",
    _TP + "all_gather_matmul",
    dtypes=("bfloat16", "float16"),
    signature=(
        "inp [M,8192] (M%128==0), w [8192,2048] -> out [M*world_size,2048]; "
        "NCCL group, world_size in {2,4,8}, NVSHMEM symmetric memory"
    ),
    description="Cake push-wait all-gather matmul distributed by FlashInfer.",
)
_register(
    "prepare_all_gather_matmul",
    _TP + "prepare_all_gather_matmul",
    dtypes=("bfloat16",),
    signature=(
        "packed-QKV launcher: (world_size,N) in {(8,1280) sm_100a/103a, "
        "(4,2560) sm_103a}; returns callable(inp) -> out"
    ),
    description=(
        "Cake prepared BF16 packed-QKV all-gather matmul launcher distributed "
        "by FlashInfer."
    ),
)

# --- fused residual + 2-track RMSNorm + 8-peer BF16 combine (CUDA IPC) --------
_register(
    "fused_norm_combine",
    _TP + "fused_norm_combine",
    dtypes=("bfloat16",),
    in_place=True,
    signature=(
        "x/residual [T,2,2560], weight [2,2560] -> norm_out/residual_out "
        "[T,2,2560], collective_out [T,2560]; world_size 8, T <= max_tokens"
    ),
    description=(
        "Cake fused residual-add + two-track RMSNorm + eight-peer BF16 all-reduce "
        "distributed by FlashInfer."
    ),
)
_register(
    "fused_norm_combine_create_workspace",
    _TP + "fused_norm_combine_create_workspace",
    signature="collective; rank, world_size=8, max_tokens, hidden=2560, group, device",
    description="Cake fused norm-combine CUDA-IPC workspace factory (FlashInfer).",
)
_register(
    "fused_norm_combine_destroy_workspace",
    _TP + "fused_norm_combine_destroy_workspace",
    signature="collective; destroy after the last launch completed",
    description="Cake fused norm-combine CUDA-IPC workspace teardown (FlashInfer).",
)

# --- MoE all-reduce union / finalize + all-reduce (TRT-LLM IPC workspace) -----
_register(
    "trtllm_moe_allreduce_fusion",
    _TP + "trtllm_moe_allreduce_fusion",
    dtypes=("bfloat16", "float16"),
    in_place=True,
    signature=(
        "expert-scaled reduction + token input + one-shot Lamport all-reduce + "
        "residual + RMSNorm; hidden 7168, world_size in {2,4,8}, no quant"
    ),
    description=(
        "Cake MoE reduction + all-reduce + residual + RMSNorm union distributed "
        "by FlashInfer."
    ),
)
_register(
    "trtllm_moe_finalize_allreduce_fusion",
    _TP + "trtllm_moe_finalize_allreduce_fusion",
    dtypes=("bfloat16", "float16"),
    in_place=True,
    signature=(
        "top-k finalize of permuted expert output + all-reduce + residual + "
        "RMSNorm [+ NVFP4 quant]; hidden 7168, world_size in {2,4,8}"
    ),
    description=(
        "Cake MoE finalize + all-reduce + residual + RMSNorm distributed by FlashInfer."
    ),
)

# --- MoE EP throughput all-to-all (functional API, backend="cake") ------------
_register(
    "moe_a2a_get_workspace_size_per_rank",
    _A2A + "moe_a2a_get_workspace_size_per_rank",
    signature="host-only; bytes of one rank's all-to-all workspace slice",
    description="Cake MoE all-to-all per-rank workspace sizing (FlashInfer).",
)
_register(
    "moe_a2a_initialize",
    _A2A + "moe_a2a_initialize",
    in_place=True,
    signature="workspace [ep_size, bytes] uint8 -> metainfo tensor",
    description="Cake MoE all-to-all workspace initialisation (FlashInfer).",
)
_register(
    "moe_a2a_dispatch",
    _A2A + "moe_a2a_dispatch",
    in_place=True,
    signature=(
        "token_selected_experts [T,top_k] int32 + payload list -> "
        "(recv payloads [ep_size,max_tokens,...], combine offset, eplb stats)"
    ),
    description="Cake MoE all-to-all dispatch distributed by FlashInfer.",
)
_register(
    "moe_a2a_combine",
    _A2A + "moe_a2a_combine",
    in_place=True,
    signature="payload [ep_size,max_tokens,H] -> combined [local_tokens,H]",
    description="Cake MoE all-to-all combine distributed by FlashInfer.",
)
_register(
    "moe_a2a_sanitize_expert_ids",
    _A2A + "moe_a2a_sanitize_expert_ids",
    in_place=True,
    signature="expert_ids slots without routed tokens := invalid_expert_id",
    description="Cake MoE all-to-all expert-id sanitiser distributed by FlashInfer.",
)

# --- moe_ep split-comm backend (CakeAlltoAll) ----------------------------------
_register(
    "moe_ep_alltoall",
    _A2A + "moe_ep_alltoall",
    signature=(
        "collective factory; BootstrapConfig + MoEEpCommParams [+ "
        "CakeAlltoAllConfig] -> CakeAlltoAll (dispatch/combine/destroy)"
    ),
    description=(
        "Cake NVLink one-sided MoE EP all-to-all backend (flashinfer.moe_ep "
        "CakeAlltoAll)."
    ),
)

# --- Kimi-K3 TP12 fused LatentMoE tail (MNNVL, exactly 12 ranks) --------------
_register(
    "create_kimi_k3_tp12_tail_workspace",
    _KIMI + "create_kimi_k3_tp12_tail_workspace",
    signature="collective; rank in [0,12), max_tokens, group, device, comm_backend",
    description="Cake Kimi-K3 TP12 tail MNNVL workspace factory (FlashInfer).",
)
_register(
    "prepare_kimi_k3_tp12_tail",
    _KIMI + "prepare_kimi_k3_tp12_tail",
    dtypes=("bfloat16",),
    signature=(
        "routed [M,3584], shared [M,7168], norm_w [3584], up_w [7168,3584], "
        "out [M,7168] -> replayable runner"
    ),
    description="Cake Kimi-K3 TP12 tail prepared runner distributed by FlashInfer.",
)
_register(
    "kimi_k3_tp12_tail",
    _KIMI + "kimi_k3_tp12_tail",
    dtypes=("bfloat16",),
    in_place=True,
    signature=(
        "out = BF16(KimiRMSNorm(sum_r routed_r) @ up_w.T + sum_r shared_r); "
        "12 ranks, out identical on every rank"
    ),
    description="Cake Kimi-K3 TP12 fused LatentMoE tail distributed by FlashInfer.",
)


def _k(name: str) -> Callable[..., Any]:
    return get_kernel(f"communication.{name}", KernelBackend.FLASHINFER)


def cake_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on ``supports_all_gather_matmul``."""
    return _k("all_gather_matmul")(inp, w, group, verbose=verbose)


def cake_prepare_all_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Explicit Cake entry point; callers gate on ``supports_prepare_all_gather_matmul``."""
    return _k("prepare_all_gather_matmul")(inp, w, group, verbose=verbose)


def cake_fused_norm_combine(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    collective_out: torch.Tensor,
    workspace: Any,
    epsilon: float,
) -> None:
    """Explicit Cake entry point; callers gate on ``supports_fused_norm_combine``."""
    return _k("fused_norm_combine")(
        x,
        residual,
        weight,
        norm_out=norm_out,
        residual_out=residual_out,
        collective_out=collective_out,
        workspace=workspace,
        epsilon=epsilon,
    )


def cake_fused_norm_combine_create_workspace(
    *,
    rank: int,
    world_size: int,
    max_tokens: int,
    hidden: int = 2560,
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> Any:
    """Collective workspace factory (every rank, same order, outside capture)."""
    return _k("fused_norm_combine_create_workspace")(
        rank=rank,
        world_size=world_size,
        max_tokens=max_tokens,
        hidden=hidden,
        group=group,
        device=device,
    )


def cake_fused_norm_combine_destroy_workspace(workspace: Any) -> None:
    """Collective workspace teardown after the last launch completed."""
    return _k("fused_norm_combine_destroy_workspace")(workspace)


def cake_trtllm_moe_allreduce_fusion(
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    layout_code: Optional[Any],
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    weight_bias: Optional[float] = None,
) -> None:
    """Explicit Cake entry point; callers gate on ``supports_trtllm_moe_allreduce_fusion``."""
    return _k("trtllm_moe_allreduce_fusion")(
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        launch_with_pdl,
        residual_in,
        rms_gamma,
        rms_eps,
        scale_factor,
        moe_reduction_device_num_experts,
        moe_reduction_scale_input,
        moe_reduction_active_experts_token_input,
        moe_reduction_token_input,
        layout_code,
        moe_allreduce_out,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
        weight_bias,
    )


def cake_trtllm_moe_finalize_allreduce_fusion(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: Optional[torch.Tensor],
    expert_scale_factor: Optional[torch.Tensor],
    routed_scaling_factor: Optional[float],
    weight_bias: Optional[float] = None,
) -> None:
    """Explicit Cake entry point; callers gate on ``supports_trtllm_moe_finalize_allreduce_fusion``."""
    return _k("trtllm_moe_finalize_allreduce_fusion")(
        allreduce_in,
        residual_in,
        norm_weight,
        expanded_idx_to_permuted_idx,
        norm_out,
        residual_out,
        quant_out,
        scale_out,
        workspace_ptrs,
        launch_with_pdl,
        world_rank,
        world_size,
        eps,
        shared_expert_output,
        expert_scale_factor,
        routed_scaling_factor,
        weight_bias,
    )


def cake_moe_a2a_get_workspace_size_per_rank(
    ep_size: int,
    max_num_tokens: int,
    total_dispatch_payload_size_per_token: int,
    combine_payload_size_per_token: int,
    eplb_stats_num_experts: int = 0,
) -> int:
    """Per-rank all-to-all workspace bytes for the Cake backend."""
    return _k("moe_a2a_get_workspace_size_per_rank")(
        ep_size,
        max_num_tokens,
        total_dispatch_payload_size_per_token,
        combine_payload_size_per_token,
        eplb_stats_num_experts,
    )


def cake_moe_a2a_initialize(
    workspace: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    max_num_tokens: int,
    eplb_stats_num_experts: int = 0,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on ``supports_moe_a2a``."""
    return _k("moe_a2a_initialize")(
        workspace, ep_rank, ep_size, max_num_tokens, eplb_stats_num_experts
    )


def cake_moe_a2a_dispatch(
    token_selected_experts: torch.Tensor,
    input_payloads: list[torch.Tensor],
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    num_experts: int,
    enable_pdl: Optional[bool] = None,
    eplb_local_stats: Optional[torch.Tensor] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
    *,
    recv_view_cache: Optional[dict] = None,
) -> tuple[list[torch.Tensor], int, Optional[torch.Tensor]]:
    """Explicit Cake entry point; callers gate on ``supports_moe_a2a``."""
    return _k("moe_a2a_dispatch")(
        token_selected_experts,
        input_payloads,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        num_experts,
        enable_pdl,
        eplb_local_stats,
        enable_rank_mask,
        active_rank_mask,
        recv_view_cache=recv_view_cache,
    )


def cake_moe_a2a_combine(
    payload: torch.Tensor,
    local_num_tokens: int,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    runtime_max_tokens_per_rank: int,
    ep_rank: int,
    ep_size: int,
    top_k: int,
    combine_payload_offset: int,
    payload_in_workspace: bool = False,
    output_dtype: Optional[torch.dtype] = None,
    output_scales: Optional[torch.Tensor] = None,
    output_scalar_scale: float = 1.0,
    sf_layout: Any = None,
    output: Optional[torch.Tensor] = None,
    *,
    use_low_precision: bool = False,
    enable_pdl: Optional[bool] = None,
    enable_rank_mask: bool = False,
    active_rank_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on ``supports_moe_a2a``."""
    return _k("moe_a2a_combine")(
        payload,
        local_num_tokens,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        ep_size,
        top_k,
        combine_payload_offset,
        payload_in_workspace,
        output_dtype,
        output_scales,
        output_scalar_scale,
        sf_layout,
        output,
        use_low_precision=use_low_precision,
        enable_pdl=enable_pdl,
        enable_rank_mask=enable_rank_mask,
        active_rank_mask=active_rank_mask,
    )


def cake_moe_a2a_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    workspace: torch.Tensor,
    metainfo: torch.Tensor,
    ep_rank: int,
    invalid_expert_id: int,
    enable_pdl: Optional[bool] = None,
) -> Any:
    """Explicit Cake entry point; callers gate on ``supports_moe_a2a``."""
    return _k("moe_a2a_sanitize_expert_ids")(
        expert_ids, workspace, metainfo, ep_rank, invalid_expert_id, enable_pdl
    )


def cake_moe_ep_alltoall(bootstrap: Any, params: Any, config: Any = None) -> Any:
    """Collective ``CakeAlltoAll`` factory; callers gate on ``supports_moe_ep_alltoall``."""
    return _k("moe_ep_alltoall")(bootstrap, params, config)


def cake_create_kimi_k3_tp12_tail_workspace(
    *,
    rank: int,
    max_tokens: int = 4096,
    group: Any = None,
    device: Optional[torch.device] = None,
    comm_backend: Any = None,
) -> Any:
    """Collective twelve-rank MNNVL workspace factory."""
    return _k("create_kimi_k3_tp12_tail_workspace")(
        rank=rank,
        max_tokens=max_tokens,
        group=group,
        device=device,
        comm_backend=comm_backend,
    )


def cake_prepare_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
) -> Any:
    """Explicit Cake entry point; callers gate on ``supports_kimi_k3_tp12_tail``."""
    return _k("prepare_kimi_k3_tp12_tail")(
        routed_partial, shared_partial, norm_weight, up_weight, out, workspace=workspace
    )


def cake_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: Any,
) -> torch.Tensor:
    """Explicit Cake entry point; callers gate on ``supports_kimi_k3_tp12_tail``."""
    return _k("kimi_k3_tp12_tail")(
        routed_partial, shared_partial, norm_weight, up_weight, out, workspace=workspace
    )


__all__ = [
    "cake_all_gather_matmul",
    "cake_create_kimi_k3_tp12_tail_workspace",
    "cake_fused_norm_combine",
    "cake_fused_norm_combine_create_workspace",
    "cake_fused_norm_combine_destroy_workspace",
    "cake_kimi_k3_tp12_tail",
    "cake_moe_a2a_combine",
    "cake_moe_a2a_dispatch",
    "cake_moe_a2a_get_workspace_size_per_rank",
    "cake_moe_a2a_initialize",
    "cake_moe_a2a_sanitize_expert_ids",
    "cake_moe_ep_alltoall",
    "cake_prepare_all_gather_matmul",
    "cake_prepare_kimi_k3_tp12_tail",
    "cake_trtllm_moe_allreduce_fusion",
    "cake_trtllm_moe_finalize_allreduce_fusion",
]
