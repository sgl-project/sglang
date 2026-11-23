"""SAIL MegaMoE runtime: precision ABI, SM budget and symmetric workspaces.

Supports transformed MXFP4 experts with FP4, FP8 or BF16 activations on SM89+.
Shared experts run outside the vendor kernel. Empty ranks still enter dispatch
and compute so peers can complete the collective.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_HIT_LOGGER = logging.getLogger("mega_moe")

# Require only the entry points for the selected precision.
_PPU_MEGA_MOE_DEEP_GEMM_APIS = (
    "get_symm_buffer_for_mega_moe",
    "mega_moe_pre_dispatch",
)


# One-shot feature marker, owned by the runtime context rather than the model.
def _mega_moe_hit_state():
    from sglang.srt.runtime_context import get_buffer

    return get_buffer("mega_moe.hit_marker", lambda: {"logged": False})


def _log_mega_moe_hit(moe) -> None:
    state = _mega_moe_hit_state()
    if state["logged"]:
        return
    state["logged"] = True
    experts = moe.experts
    _MEGA_HIT_LOGGER.warning(
        "MEGA_MOE_HIT MegaMoE path is live (one-shot marker, per rank): "
        "first_hit_layer=%s experts=%s mma=%s n_shared_experts=%s",
        getattr(moe, "prefix", "?"),
        getattr(experts, "num_experts", "?"),
        _mega_moe_mma_type(experts),
        getattr(moe, "n_shared_experts", "?"),
    )


def _ppu_mega_moe_mma_type(mma_type: str) -> str:
    """Translate per-layer precision labels to the PPU DeepGEMM API."""
    return {
        "mxf4xmxf4": "fp4xfp4",
        "fp8xmxfp4": "fp8xfp4",
        "bf16xmxfp4": "bf16xfp4",
    }.get(mma_type, mma_type)


def _mega_moe_mma_type(experts) -> str:
    if experts._mega_moe_nvfp4:
        return "nvfp4xnvfp4"
    mma_type = getattr(experts, "_mega_moe_mma_type", None)
    if mma_type is not None:
        return mma_type
    if experts._mega_moe_w4a4:
        return "mxf4xmxf4"
    return "fp8xfp8" if experts.w13_weight.dtype == torch.float8_e4m3fn else "fp8xfp4"


@functools.lru_cache(maxsize=1)
def _mega_moe_max_num_sms() -> int:
    # Use the physical count so an outer DeepGEMM context does not compound reserves.
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    reserved_num_sms = max(envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.get(), 0)
    return max(2, num_sms - reserved_num_sms)


@contextmanager
def _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
    current_num_sms = deep_gemm.get_num_sms()
    target_num_sms = min(_mega_moe_max_num_sms(), current_num_sms)
    # SM89 permits odd CTA counts; newer kernels retain the even-count contract.
    if _device_sm != 89:
        target_num_sms -= target_num_sms % 2
    if target_num_sms == current_num_sms:
        yield
        return
    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    *,
    mma_type: str,
    activation: str = "swiglu",
) -> SymmBuffer:
    import deep_gemm

    mma_type = _ppu_mega_moe_mma_type(mma_type)
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        key = (
            id(group),
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
            mma_type,
            activation,
            num_shared_experts,
            deep_gemm.get_num_sms(),
        )
        buf = _MEGA_MOE_SYMM_BUFFER.get(key)
        if buf is None:
            buf = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                num_shared_experts=num_shared_experts,
                mma_type=mma_type,
                activation=activation,
            )
            _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def is_mega_moe_experts_ready(experts) -> bool:
    return (
        experts._mega_moe_weights_built
        and _device_sm >= 89
        and _mega_moe_mma_type(experts)
        in (
            "mxf4xmxf4",
            "fp8xmxfp4",
            "bf16xmxfp4",
        )
    )


def run_mega_routed_experts(
    experts,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_tokens: int,
    activation_clamp: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
    activation: str = "swiglu",
    num_shared_experts: int = 0,
    routing_in_buffer: bool = False,
    buf: Optional[SymmBuffer] = None,
    eplb_load_row: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.runtime_context import get_parallel

    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or lower "
        f"--cuda-graph-max-bs-decode / --chunked-prefill-size accordingly"
    )
    mma_type = _mega_moe_mma_type(experts)
    if buf is None:
        buf = _get_mega_moe_symm_buffer(
            get_parallel().moe_ep_group.device_group,
            num_experts=experts.num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=top_k,
            hidden=hidden_size,
            intermediate_hidden=intermediate_size,
            num_shared_experts=num_shared_experts,
            mma_type=mma_type,
            activation=activation,
        )
    # The supported SAIL precisions do not consume CUDA's in-buffer TopK handoff.
    assert not routing_in_buffer
    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int64)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int64)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)
    if routed_scaling_factor != 1.0:
        topk_weights_in = topk_weights_in * routed_scaling_factor
    deep_gemm.mega_moe_pre_dispatch(
        hidden_states.contiguous(),
        topk_ids_in.contiguous(),
        topk_weights_in.contiguous(),
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=32,
        mma_type=_ppu_mega_moe_mma_type(mma_type),
    )
    mega_kwargs = {"recipe": (1, 1, 32)} if mma_type == "mxf4xmxf4" else {}
    if eplb_load_row is not None:
        mega_kwargs["cumulative_local_expert_recv_stats"] = eplb_load_row
    y = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        mega_moe_kernel = getattr(
            deep_gemm,
            {
                "mxf4xmxf4": "fp4_fp4_mega_moe",
                "fp8xmxfp4": "fp8_fp4_mega_moe",
                "bf16xmxfp4": "bf16_fp4_mega_moe",
            }[mma_type],
        )
        mega_moe_kernel(
            y,
            experts.mega_l1_weights,
            experts.mega_l2_weights,
            buf,
            activation=activation,
            activation_clamp=activation_clamp,
            fast_math=True,
            **mega_kwargs,
        )
    return y[:num_tokens]


def mega_moe_buffer_kimi(moe):
    return _get_mega_moe_symm_buffer(
        get_parallel().moe_ep_group.device_group,
        num_experts=moe.experts.num_experts,
        num_max_tokens_per_rank=envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get(),
        num_topk=moe._mega_top_k,
        hidden=moe.moe_hidden_size,
        intermediate_hidden=moe._mega_intermediate_size,
        mma_type=_mega_moe_mma_type(moe.experts),
        activation="situ",
    )


def forward_mega_experts_kimi(
    moe,
    routed_input: torch.Tensor,
    topk_output,
    *,
    valid_token_count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run three SAIL precisions, masking padded NaNs before and after dispatch."""
    import deep_gemm

    num_tokens = routed_input.shape[0]
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; K3 has no non-mega fallback — raise "
        f"the env var to cover the per-rank rows"
    )
    mma_type = _mega_moe_mma_type(moe.experts)
    buf = moe._mega_moe_buffer()
    if num_tokens > 0:
        topk_ids_in = topk_output.topk_ids.to(torch.int64)
        topk_weights_in = topk_output.topk_weights.to(torch.float32)
    else:
        topk_ids_in = routed_input.new_empty((0, moe._mega_top_k), dtype=torch.int64)
        topk_weights_in = routed_input.new_empty(
            (0, moe._mega_top_k), dtype=torch.float32
        )
    valid_rows = None
    if valid_token_count is not None:
        valid_rows = (
            torch.arange(num_tokens, device=routed_input.device) < valid_token_count
        )
        routed_input = routed_input.masked_fill(~valid_rows[:, None], 0)
        topk_ids_in = topk_ids_in.masked_fill(~valid_rows[:, None], -1)
        topk_weights_in = topk_weights_in.masked_fill(~valid_rows[:, None], 0)
    deep_gemm.mega_moe_pre_dispatch(
        routed_input.contiguous(),
        topk_ids_in.contiguous(),
        topk_weights_in.contiguous(),
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
        group_size=32,
        mma_type=_ppu_mega_moe_mma_type(mma_type),
    )
    y = torch.empty(
        (num_tokens, moe.moe_hidden_size),
        dtype=torch.bfloat16,
        device=routed_input.device,
    )
    with _configure_mega_moe_deep_gemm_num_sms(deep_gemm):
        mega_moe_kernel = getattr(
            deep_gemm,
            {
                "mxf4xmxf4": "fp4_fp4_mega_moe",
                "fp8xmxfp4": "fp8_fp4_mega_moe",
                "bf16xmxfp4": "bf16_fp4_mega_moe",
            }[mma_type],
        )
        situ_kwargs = {
            "situ_beta": moe.experts.moe_runner_config.gemm1_alpha,
            "situ_linear_beta": moe.experts.moe_runner_config.gemm1_clamp_limit,
        }
        if mma_type == "mxf4xmxf4":
            situ_kwargs["recipe"] = (1, 1, 32)
        mega_moe_kernel(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            activation="situ",
            fast_math=True,
            **situ_kwargs,
        )
    y = y[:num_tokens]
    if valid_rows is not None:
        y.masked_fill_(~valid_rows[:, None], 0)
    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        if moe.routed_scaling_factor is not None and moe.routed_scaling_factor != 1.0:
            y.mul_(moe.routed_scaling_factor)
    return y


def _require_ppu_mega_moe_deep_gemm_apis() -> None:
    try:
        import deep_gemm
    except ImportError as exc:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU requires the PPU DeepGEMM build"
        ) from exc

    from sglang.srt.hardware_backend.sail.quantization.mxfp4 import mega_moe_precision

    precision_apis = {
        "mxf4xmxf4": ("fp4_fp4_mega_moe", "transform_weights_for_mega_moe"),
        "fp8xmxfp4": ("fp8_fp4_mega_moe", "transform_weights_for_w4a8_mega_moe"),
        "bf16xmxfp4": ("bf16_fp4_mega_moe", "transform_weights_for_w4a16_mega_moe"),
    }[mega_moe_precision()]
    missing = [
        name
        for name in (*_PPU_MEGA_MOE_DEEP_GEMM_APIS, *precision_apis)
        if not callable(getattr(deep_gemm, name, None))
    ]
    if missing:
        raise ValueError(
            "--moe-a2a-backend megamoe on PPU requires a DeepGEMM build with the "
            "PPU MegaMoE entry points; missing: " + ", ".join(missing)
        )


def validate_mega_moe(device_sm: int) -> None:
    if device_sm < 89:
        raise ValueError("--moe-a2a-backend megamoe on PPU requires SM89 or newer.")
    _require_ppu_mega_moe_deep_gemm_apis()
