"""SAIL MegaGate coverage and fallback reporting; shared routing stays unchanged."""

import functools

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.utils import _device_sm
from sglang.srt.runtime_context import get_forward
from sglang.srt.utils.log_utils import print_info_once

_FALLBACK_MSG = "MegaGate fallback: the regular router keeps routing"
# DeepGEMM bf16_mega_gate scores one token block per launch.
_MEGA_GATE_MAX_ROWS = 2048


def log_fallback() -> None:
    print_info_once(_FALLBACK_MSG)


@functools.lru_cache(maxsize=1)
def _sail_mega_gate_available() -> bool:
    # The checked PPU kernels dispatch architecture major 8; other SAIL devices
    # retain the regular router until their library exposes a verified path.
    if _device_sm < 89 or _device_sm // 10 != 8:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    fn = getattr(deep_gemm, "bf16_mega_gate", None)
    if not callable(fn):
        return False
    # The target facade always forwards its physical-routing and output kwargs,
    # even when their values are None. Older same-name PPU APIs reject them.
    import inspect
    import re

    required = {
        "use_shared_as_routed",
        "num_shared_experts",
        "routed_scaling_factor",
        "ep_rank",
        "scoring_func",
        "mask",
        "bias",
        "image_bias",
        "image_token_mask",
        "fix_routing_mask",
        "unmapped_topk_idx",
        "to_physical_map",
        "logical_count",
        "out",
    }
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        doc = fn.__doc__ or ""
        return all(re.search(r"\b" + name + r"\b", doc) for name in required)
    return required <= params.keys() or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def should_use_mega_gate(moe, hidden_states, forward_batch) -> bool:
    if not envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.get():
        return False
    if forward_batch is None:
        log_fallback()
        return False
    mode = forward_batch.forward_mode
    if mode.is_idle():
        return False
    max_rows = min(
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get(),
        _MEGA_GATE_MAX_ROWS,
    )
    import torch

    use_mega_gate = (
        (mode.is_decode() or mode.is_target_verify())
        and moe.mega_gate_supported
        and 0 < hidden_states.shape[0] <= max_rows
        and not get_forward().sp_active
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
    )
    if not use_mega_gate:
        log_fallback()
    return use_mega_gate
