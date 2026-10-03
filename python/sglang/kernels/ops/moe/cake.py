"""Cake (FlashInfer) backends for the ``moe`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels` (``moe_*`` modules), which
import FlashInfer only when a kernel is actually called. Callers gate every
entry on the adapter's ``supports_*`` admission check.

Multi-rank entries (MXFP8 MegaMoE EP16, SM90 push-cake megakernel) forward the
prepare / session factory and the config / weight preprocessors only; process
groups and ``MoEEpLayer`` bootstrap stay in ``sglang.srt``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

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

_ADAPTERS = "sglang.kernels.cake_kernels."

_SM100_SM103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})
_SM100_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 0))})
_SM103_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3))})
_SM90_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 0))})
_SM90_TO_SM103 = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(10, 3))})
# sm_100a / sm_103a / sm_107a; the adapter's supports_* checks exact membership.
_SM100_TO_SM107 = frozenset(
    {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 7))}
)

# (op name, adapter module, capabilities, supported dtypes, in_place, format, description)
_REGISTRATIONS: Tuple[Tuple[str, str, Any, Tuple[str, ...], bool, str, str], ...] = (
    # --- Kimi-K3 SiTU fused MoE (TP8-local NVFP4 group-16) -------------------
    (
        "kimi_k3_situ_fused_moe_workspace_size",
        "moe_kimi_k3_situ",
        _SM100_SM103,
        (),
        False,
        "host metadata: SiTU workspace bytes for max_num_tokens (H=3584,I=384,E=896,top_k=16)",
        "Cake Kimi-K3 SiTU fused MoE workspace size (FlashInfer cutlass_fused_moe_workspace_size(backend='cake')).",
    ),
    (
        "kimi_k3_situ_fused_moe_prepare_workspace",
        "moe_kimi_k3_situ",
        _SM100_SM103,
        ("uint8",),
        True,
        "prepare 1-D uint8 workspace for one token count outside CUDA-graph capture",
        "Cake Kimi-K3 SiTU fused MoE workspace preparation (FlashInfer cake_fused_moe_prepare_workspace).",
    ),
    (
        "kimi_k3_situ_fused_moe",
        "moe_kimi_k3_situ",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "BF16 x [T,3584] + int32 ids/bf16 scales [T,16] + packed NVFP4 group-16 weights -> BF16 output [T,3584]",
        "Cake Kimi-K3 SiTU fused MoE complete call (FlashInfer cutlass_fused_moe(backend='cake')).",
    ),
    # --- NVFP4 warp-decode runner ---------------------------------------------
    (
        "warp_decode_config",
        "moe_warp_decode",
        _SM100_SM103,
        (),
        False,
        "CakeWarpDecodeConfig(backend='cake') for MoEConfig.backend",
        "Cake NVFP4 warp-decode MoE backend config (FlashInfer CakeWarpDecodeConfig).",
    ),
    (
        "warp_decode_runner",
        "moe_warp_decode",
        _SM100_SM103,
        (),
        False,
        "CakeWarpDecodeRunner(config, device); 1..32 tokens, NVFP4xNVFP4, calibrated geometries",
        "Cake NVFP4 warp-decode MoE runner (FlashInfer CakeWarpDecodeRunner).",
    ),
    (
        "warp_decode_prepare_weights",
        "moe_warp_decode",
        _SM100_SM103,
        ("bfloat16",),
        False,
        "BF16 w1/w2 -> TRTLLM NVFP4 physical weight view shared with TrtllmFp4Config",
        "Cake warp-decode weight preparation (FlashInfer CakeWarpDecodeConfig.prepare_weights).",
    ),
    (
        "warp_decode_prepare_activations",
        "moe_warp_decode",
        _SM100_SM103,
        ("bfloat16",),
        False,
        "BF16 hidden states -> (packed NVFP4 activations, linear block scales)",
        "Cake warp-decode activation preparation (FlashInfer CakeWarpDecodeConfig.prepare_activations).",
    ),
    # --- BGMV MoE-LoRA -----------------------------------------------------------
    (
        "prepare_bgmv_moe",
        "moe_bgmv_lora",
        _SM90_TO_SM103,
        ("bfloat16", "float16"),
        False,
        "BF16/FP16 x [T,H], one LoRA slice rank {8,16,32,64}, H % 8 == 0 -> plan.run() f32 y_accum [T,H]",
        "Cake BGMV MoE-LoRA shrink+expand prepared plan (FlashInfer prepare_bgmv_moe(backend='cake')).",
    ),
    # --- Kimi-K3 fused router ---------------------------------------------------
    (
        "allocate_kimi_k3_route_plan",
        "moe_kimi_k3_router",
        _SM100_SM103,
        (),
        False,
        "worst-case KimiK3RoutePlan buffers for (num_tokens, block_m); no launch",
        "Cake Kimi-K3 fused router plan allocation (FlashInfer allocate_kimi_k3_route_plan).",
    ),
    (
        "prepare_kimi_k3_fused_router",
        "moe_kimi_k3_router",
        _SM100_SM103,
        ("float32",),
        True,
        "f32 logits [T,896] + bias [896], T in {1,2,4,...,8192}, block_m in {8,16} -> runner writing the route plan",
        "Cake Kimi-K3 fused router prepared runner (FlashInfer prepare_kimi_k3_fused_router).",
    ),
    (
        "kimi_k3_fused_router",
        "moe_kimi_k3_router",
        _SM100_SM103,
        ("float32",),
        True,
        "prepare + one launch -> KimiK3RoutePlan (moe_align_block_size layout)",
        "Cake Kimi-K3 fused router one-shot (FlashInfer kimi_k3_fused_router).",
    ),
    # --- Kimi-K3 LatentMoE front / tail ----------------------------------------
    (
        "prepare_kimi_k3_latent_moe_front",
        "moe_kimi_k3_latent",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "BF16 x [T,7168] -> f32 logits [T,896], BF16 latent [T,3584], BF16 SiTU shared_act [T,6144/tp]; 148 SMs",
        "Cake Kimi-K3 LatentMoE front prepared runner (FlashInfer prepare_kimi_k3_latent_moe_front).",
    ),
    (
        "kimi_k3_latent_moe_front",
        "moe_kimi_k3_latent",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "prepare + one launch -> (logits, latent, shared_act)",
        "Cake Kimi-K3 LatentMoE front one-shot (FlashInfer kimi_k3_latent_moe_front).",
    ),
    (
        "prepare_kimi_k3_latent_moe_tail",
        "moe_kimi_k3_latent",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "BF16 routed [P,T,3584] -> RMSNorm y_workspace [T,3584] + out [T,7168] rank partial; tp in {1,8}; 148 SMs",
        "Cake Kimi-K3 LatentMoE tail prepared runner (FlashInfer prepare_kimi_k3_latent_moe_tail).",
    ),
    (
        "kimi_k3_latent_moe_tail",
        "moe_kimi_k3_latent",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "prepare + one launch -> out",
        "Cake Kimi-K3 LatentMoE tail one-shot (FlashInfer kimi_k3_latent_moe_tail).",
    ),
    # --- DeepSeek-V3 fused routing ----------------------------------------------
    (
        "fused_topk_deepseek",
        "moe_deepseek_routing",
        _SM100_SM103,
        ("float16", "bfloat16", "float32"),
        True,
        "scores [T,E] + bias [E] -> in-place topk_values [T,topk] / int32 topk_indices; topk <= 8, grouped contract",
        "Cake DeepSeek-V3 fused NoAuxTc routing (FlashInfer fused_topk_deepseek(backend='cake')).",
    ),
    # --- MegaMoE v3 / source MegaMoE (DeepGEMM port) -------------------------------
    (
        "prepare_mega_moe_pipeline",
        "moe_mega_moe",
        _SM100_SM103,
        ("float8_e4m3fn",),
        False,
        "packed E4M3 x + FP4/FP8 experts, E=384 top-6 H=5120 I=2304, T in {1,16,128,512}; V3Plan.run() -> BF16 [T,H]",
        "Cake MegaMoE v3 complete pipeline prepared plan (FlashInfer mega_moe_v3.prepare_pipeline).",
    ),
    (
        "prepare_mega_moe_grouped_l2",
        "moe_mega_moe",
        _SM100_SM103,
        ("float8_e4m3fn",),
        False,
        "grouped gran32 E4M3 A x packed E2M1 B -> BF16; scales repacked at prepare / update_scales()",
        "Cake MegaMoE v3 grouped L2 GEMM prepared plan (FlashInfer mega_moe_v3.prepare_grouped_l2).",
    ),
    (
        "prepare_mega_moe_grouped_fused",
        "moe_mega_moe",
        _SM100_SM103,
        ("float8_e4m3fn",),
        False,
        "explicit packed bindings (K1,K2,Out,l1_arrival,...) for grouped L1+SwiGLU+L2; host reset inside run()",
        "Cake MegaMoE v3 grouped fused prepared plan (FlashInfer mega_moe_v3.prepare_grouped_fused).",
    ),
    (
        "prepare_mega_moe_grouped_l1",
        "moe_mega_moe",
        _SM100_SM103,
        ("float8_e4m3fn",),
        False,
        "grouped L1 + SwiGLU -> (C_fp8 E4M3 bytes, SF_out UE8M0 bytes)",
        "Cake MegaMoE v3 grouped L1 prepared plan (FlashInfer mega_moe_v3.prepare_grouped_l1).",
    ),
    (
        "bind_mega_moe_prepared",
        "moe_mega_moe",
        _SM100_SM103,
        (),
        False,
        "low-level catalog binding -> V3Plan; main stage grid must equal the catalog grid",
        "Cake MegaMoE v3 low-level route binding (FlashInfer mega_moe_v3.bind_prepared).",
    ),
    (
        "prepare_source_mega_moe",
        "moe_mega_moe",
        _SM100_SM103,
        ("float8_e4m3fn",),
        False,
        "single-rank complete MegaMoE (dispatch, 2 GEMMs, clamped SwiGLU, FP8 shared expert, combine) -> MegaMoEPlan",
        "Cake source MegaMoE prepared plan (FlashInfer source_mega_moe.prepare_mega_moe).",
    ),
    # --- Mega gate --------------------------------------------------------------
    (
        "prepare_mega_gate",
        "moe_mega_gate",
        _SM100_SM103,
        ("bfloat16",),
        False,
        "BF16 x [M,5120] @ weight [384,5120] sqrtsoftplus top-6 (+ physical map) -> plan.run() (int64 ids, f32 weights)",
        "Cake fused routing GEMM + mapping + normalized top-k prepared plan (FlashInfer mega_gate.prepare_mega_gate).",
    ),
    # --- MXFP8 MegaMoE EP16 (NVSHMEM, 16 ranks) ---------------------------------
    (
        "preprocess_mxfp8_megamoe_ep16_weights",
        "moe_mxfp8_megamoe_ep16",
        _SM103_ONLY,
        ("bfloat16",),
        False,
        "rank-local BF16 w13 [32,10240,3072] / w2 [32,3072,5120] -> MXFP8 e4m3 + packed UE8M0 scales",
        "Cake MXFP8 MegaMoE EP16 weight preprocessing (FlashInfer moe_ep.preprocess_cake_mxfp8_megamoe_ep16_weights).",
    ),
    (
        "create_mxfp8_megamoe_ep16_session",
        "moe_mxfp8_megamoe_ep16",
        _SM103_ONLY,
        ("bfloat16",),
        False,
        "collective 16-rank session factory; session.run(hidden [T,3072], topk_ids, topk_weights, out=) -> BF16; no graph capture",
        "Cake MXFP8 MegaMoE EP16 prepared session (FlashInfer moe_ep.CakeMxfp8MegaMoeEp16).",
    ),
    # --- SM90 push-cake BF16 mega-MoE -------------------------------------------
    (
        "sm90_push_cake_megamoe_config",
        "moe_sm90_push_cake",
        _SM90_ONLY,
        ("bfloat16",),
        False,
        "Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig for MegaConfig(megakernel=...) / MoEEpLayer",
        "Cake SM90 native BF16 push mega-MoE config (FlashInfer moe_ep.Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig).",
    ),
    (
        "preprocess_sm90_push_cake_bf16_mega_weights",
        "moe_sm90_push_cake",
        _SM90_ONLY,
        ("bfloat16",),
        False,
        "canonical BF16 MoEWeightPack -> gate/up-interleaved TransformedMegaWeights for the fused FC1 kernel",
        "Cake SM90 push mega-MoE weight preprocessing (FlashInfer moe_ep.preprocess_sm90_push_cake_bf16_mega_weights).",
    ),
    # --- MegaMoE top-k reducer ----------------------------------------------------
    (
        "load_megamoe_topk_reduce_module",
        "moe_megamoe_topk_reduce",
        _SM100_SM103,
        (),
        False,
        "load the frozen reducer module for a device before CUDA-graph capture",
        "Cake MegaMoE top-k reducer module loader (FlashInfer jit.cake_megamoe_topk_reduce).",
    ),
    (
        "megamoe_topk_reduce",
        "moe_megamoe_topk_reduce",
        _SM100_SM103,
        ("bfloat16",),
        True,
        "BF16 partials [cap in {256,4096}, 6, 4096] -> BF16 out [cap, 4096] for num_tokens rows (FP32 accumulate)",
        "Cake frozen MegaMoE top-k reducer (FlashInfer run_cake_megamoe_topk_reduce).",
    ),
)

for _name, _module, _caps, _dtypes, _in_place, _format, _desc in _REGISTRATIONS:
    register_kernel(
        KernelSpec(
            op=f"moe.{_name}",
            backend=KernelBackend.FLASHINFER,
            target=f"{_ADAPTERS}{_module}:{_name}",
            capabilities=_caps,
            format_signature=FormatSignature(
                supported_dtypes=_dtypes,
                in_place=_in_place,
                description=_format,
            ),
            description=_desc,
        )
    )
del _name, _module, _caps, _dtypes, _in_place, _format, _desc


def _cake(name: str):
    return get_kernel(f"moe.{name}", KernelBackend.FLASHINFER)


# --- Kimi-K3 SiTU fused MoE ---------------------------------------------------


def cake_kimi_k3_situ_fused_moe_workspace_size(
    max_num_tokens: int,
    *,
    tp_rank: int = 0,
    device: Optional[torch.device] = None,
) -> int:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _cake("kimi_k3_situ_fused_moe_workspace_size")(
        max_num_tokens, tp_rank=tp_rank, device=device
    )


def cake_kimi_k3_situ_fused_moe_prepare_workspace(
    workspace_buffer: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    return _cake("kimi_k3_situ_fused_moe_prepare_workspace")(
        workspace_buffer, num_tokens
    )


def cake_kimi_k3_situ_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: List[torch.Tensor],
    *,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    tp_rank: int = 0,
    enable_pdl: Optional[bool] = None,
    situ_beta: Optional[torch.Tensor] = None,
    situ_linear_beta: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    return _cake("kimi_k3_situ_fused_moe")(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        quant_scales,
        output=output,
        workspace_buffer=workspace_buffer,
        tp_rank=tp_rank,
        enable_pdl=enable_pdl,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        tune_max_num_tokens=tune_max_num_tokens,
    )


# --- NVFP4 warp decode ----------------------------------------------------------


def cake_warp_decode_config():
    return _cake("warp_decode_config")()


def cake_warp_decode_runner(config: Any, device: torch.device):
    return _cake("warp_decode_runner")(config, device)


def cake_warp_decode_prepare_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: Any = None,
    quant: Any = None,
    device: Optional[torch.device] = None,
    permute_cache: Any = None,
):
    return _cake("warp_decode_prepare_weights")(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        quant=quant,
        device=device,
        permute_cache=permute_cache,
    )


def cake_warp_decode_prepare_activations(
    hidden_states_bf16: torch.Tensor, *, quant: Any = None
):
    return _cake("warp_decode_prepare_activations")(hidden_states_bf16, quant=quant)


# --- BGMV MoE-LoRA ---------------------------------------------------------------


def cake_prepare_bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    *,
    fallback: bool = False,
    shrink_out: Optional[torch.Tensor] = None,
    y_accum: Optional[torch.Tensor] = None,
):
    return _cake("prepare_bgmv_moe")(
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
        fallback=fallback,
        shrink_out=shrink_out,
        y_accum=y_accum,
    )


# --- Kimi-K3 fused router ---------------------------------------------------------


def cake_allocate_kimi_k3_route_plan(
    num_tokens: int, block_m: int, device: torch.device
):
    return _cake("allocate_kimi_k3_route_plan")(num_tokens, block_m, device)


def cake_prepare_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
):
    return _cake("prepare_kimi_k3_fused_router")(
        logits, bias, block_m=block_m, plan=plan
    )


def cake_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
):
    return _cake("kimi_k3_fused_router")(logits, bias, block_m=block_m, plan=plan)


# --- Kimi-K3 LatentMoE front / tail -------------------------------------------------


def cake_prepare_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
):
    return _cake("prepare_kimi_k3_latent_moe_front")(
        x, gate_weight, down_weight, shared_gate_up_weight, logits, latent, shared_act
    )


def cake_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _cake("kimi_k3_latent_moe_front")(
        x, gate_weight, down_weight, shared_gate_up_weight, logits, latent, shared_act
    )


def cake_prepare_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
):
    return _cake("prepare_kimi_k3_latent_moe_tail")(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
    )


def cake_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> torch.Tensor:
    return _cake("kimi_k3_latent_moe_tail")(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
    )


# --- DeepSeek-V3 fused routing ---------------------------------------------------


def cake_fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    launch_with_pdl: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> None:
    _cake("fused_topk_deepseek")(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl,
        routing_replay_out,
    )


# --- MegaMoE v3 / source MegaMoE ----------------------------------------------------


def cake_prepare_mega_moe_pipeline(inputs: Mapping[str, Any]):
    return _cake("prepare_mega_moe_pipeline")(inputs)


def cake_prepare_mega_moe_grouped_l2(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    per_expert_M: Sequence[int],
    *,
    out: Optional[torch.Tensor] = None,
    packed_a: Optional[torch.Tensor] = None,
    packed_b: Optional[torch.Tensor] = None,
):
    return _cake("prepare_mega_moe_grouped_l2")(
        A, B, SFA, SFB, per_expert_M, out=out, packed_a=packed_a, packed_b=packed_b
    )


def cake_prepare_mega_moe_grouped_fused(
    bindings: Mapping[str, Any], per_expert_M: Sequence[int]
):
    return _cake("prepare_mega_moe_grouped_fused")(bindings, per_expert_M)


def cake_prepare_mega_moe_grouped_l1(
    bindings: Mapping[str, Any], per_expert_M: Sequence[int]
):
    return _cake("prepare_mega_moe_grouped_l1")(bindings, per_expert_M)


def cake_bind_mega_moe_prepared(
    surface: str,
    args: Mapping[str, Any],
    stage_bindings: Mapping[str, Mapping[str, Any]],
    outputs: Any,
    *,
    reset_storage: Any = None,
    reset_buffers: Sequence[Any] = (),
    preparation: Sequence[str] = (),
    owners: Sequence[Any] = (),
):
    return _cake("bind_mega_moe_prepared")(
        surface,
        args,
        stage_bindings,
        outputs,
        reset_storage=reset_storage,
        reset_buffers=reset_buffers,
        preparation=preparation,
        owners=owners,
    )


def cake_prepare_source_mega_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    weights: Mapping[str, torch.Tensor],
    num_experts: int,
    intermediate: int,
    routed_weight_dtype: str = "fp4",
    num_shared_experts: int = 1,
    activation_clamp: float = 10.0,
    fast_math: bool = True,
    workspace: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
):
    return _cake("prepare_source_mega_moe")(
        x,
        x_sf,
        topk_idx,
        topk_weights,
        weights=weights,
        num_experts=num_experts,
        intermediate=intermediate,
        routed_weight_dtype=routed_weight_dtype,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        workspace=workspace,
        out=out,
        descriptor_workspace=descriptor_workspace,
    )


# --- Mega gate -------------------------------------------------------------------


def cake_prepare_mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_topk: int = 6,
    *,
    scoring_func: str = "sqrtsoftplus",
    bias: Optional[torch.Tensor] = None,
    image_bias: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    force_random: Optional[Any] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
    use_shared_as_routed: bool = False,
    num_shared_experts: int = 1,
    routed_scaling_factor: float = 1.5,
    ep_rank: int = 0,
    out: Optional[Any] = None,
    deterministic: bool = False,
    scratch: Optional[torch.Tensor] = None,
    score_barriers: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
):
    return _cake("prepare_mega_gate")(
        x,
        weight,
        num_topk,
        scoring_func=scoring_func,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        mask=mask,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        fix_routing_mask=fix_routing_mask,
        force_random=force_random,
        unmapped_topk_idx=unmapped_topk_idx,
        use_shared_as_routed=use_shared_as_routed,
        num_shared_experts=num_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        out=out,
        deterministic=deterministic,
        scratch=scratch,
        score_barriers=score_barriers,
        descriptor_workspace=descriptor_workspace,
    )


# --- MXFP8 MegaMoE EP16 ----------------------------------------------------------


def cake_preprocess_mxfp8_megamoe_ep16_weights(w13: torch.Tensor, w2: torch.Tensor):
    return _cake("preprocess_mxfp8_megamoe_ep16_weights")(w13, w2)


def cake_create_mxfp8_megamoe_ep16_session(
    weights: Any,
    topk_ids: torch.Tensor,
    *,
    process_group: Any = None,
    backend: str = "cuda",
):
    """Collective: every rank of the 16-rank EP group must call this."""
    return _cake("create_mxfp8_megamoe_ep16_session")(
        weights, topk_ids, process_group=process_group, backend=backend
    )


# --- SM90 push-cake BF16 mega-MoE ---------------------------------------------------


def cake_sm90_push_cake_megamoe_config(
    intermediate_size: int,
    top_k: int,
    *,
    capacity_factor: float = 1.0,
    dedup_dispatch: bool = True,
    clamp_limit: Optional[float] = None,
    allow_unverified_p2p: bool = False,
    init_timeout_s: float = 600.0,
):
    return _cake("sm90_push_cake_megamoe_config")(
        intermediate_size,
        top_k,
        capacity_factor=capacity_factor,
        dedup_dispatch=dedup_dispatch,
        clamp_limit=clamp_limit,
        allow_unverified_p2p=allow_unverified_p2p,
        init_timeout_s=init_timeout_s,
    )


def cake_preprocess_sm90_push_cake_bf16_mega_weights(
    weights: Any,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
):
    return _cake("preprocess_sm90_push_cake_bf16_mega_weights")(
        weights,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_local_experts=num_local_experts,
    )


# --- MegaMoE top-k reducer ------------------------------------------------------------


def cake_load_megamoe_topk_reduce_module(device: Optional[torch.device] = None):
    return _cake("load_megamoe_topk_reduce_module")(device)


def cake_megamoe_topk_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return _cake("megamoe_topk_reduce")(partials, out, num_tokens)


__all__ = [
    "cake_allocate_kimi_k3_route_plan",
    "cake_bind_mega_moe_prepared",
    "cake_create_mxfp8_megamoe_ep16_session",
    "cake_fused_topk_deepseek",
    "cake_kimi_k3_fused_router",
    "cake_kimi_k3_latent_moe_front",
    "cake_kimi_k3_latent_moe_tail",
    "cake_kimi_k3_situ_fused_moe",
    "cake_kimi_k3_situ_fused_moe_prepare_workspace",
    "cake_kimi_k3_situ_fused_moe_workspace_size",
    "cake_load_megamoe_topk_reduce_module",
    "cake_megamoe_topk_reduce",
    "cake_prepare_bgmv_moe",
    "cake_prepare_kimi_k3_fused_router",
    "cake_prepare_kimi_k3_latent_moe_front",
    "cake_prepare_kimi_k3_latent_moe_tail",
    "cake_prepare_mega_gate",
    "cake_prepare_mega_moe_grouped_fused",
    "cake_prepare_mega_moe_grouped_l1",
    "cake_prepare_mega_moe_grouped_l2",
    "cake_prepare_mega_moe_pipeline",
    "cake_prepare_source_mega_moe",
    "cake_preprocess_mxfp8_megamoe_ep16_weights",
    "cake_preprocess_sm90_push_cake_bf16_mega_weights",
    "cake_sm90_push_cake_megamoe_config",
    "cake_warp_decode_config",
    "cake_warp_decode_prepare_activations",
    "cake_warp_decode_prepare_weights",
    "cake_warp_decode_runner",
]
