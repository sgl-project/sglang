"""Block-FP8 forward for true-on-policy serving, shared with the trainer.

TOP uses these operations whenever the served checkpoint is block-wise FP8
(E4M3 payloads, dynamic activations, 128x128 weight blocks; or DeepSeek-V4.1's
32x32 blocks with UE8M0 scales). The choice follows the checkpoint, like the rest
of SGLang's quantization; there is no separate switch. Other quantization formats
are not supported under TOP.
"""

import contextlib
import logging

import torch

BLOCK = [128, 128]
# DeepSeek-V4.1's dense format: 32x32 blocks, power-of-two (UE8M0) scales; activations quantized per 32 channels.
BLOCK32 = [32, 32]


def checkpoint_block(quantization_config):
    """The TOP block-FP8 format a HF ``quantization_config`` describes: BLOCK, BLOCK32 or None."""
    if quantization_config is None:
        return None
    if not isinstance(quantization_config, dict):
        quantization_config = quantization_config.to_dict()
    if (
        quantization_config.get("quant_method") != "fp8"
        or quantization_config.get("fmt", "e4m3") != "e4m3"
        or quantization_config.get("activation_scheme") != "dynamic"
    ):
        return None
    block = list(quantization_config.get("weight_block_size") or [])
    if block == BLOCK:
        return BLOCK
    if block == BLOCK32 and quantization_config.get("scale_fmt") == "ue8m0":
        return BLOCK32
    return None


def is_block_fp8(quantization_config) -> bool:
    """Whether a HF ``quantization_config`` describes TOP's 128x128 block-FP8 format."""
    return checkpoint_block(quantization_config) == BLOCK


_SERVED_BLOCK: list = []


def served_block():
    """The block-FP8 format TOP is serving (BLOCK or BLOCK32), None when TOP or FP8 is off; resolved once per process.

    None, and left unresolved, while no server args are published (kernel tests, benchmarks, offline scripts).
    """
    if _SERVED_BLOCK:
        return _SERVED_BLOCK[0]
    from sglang.srt.runtime_context import get_server_args, process_model_config
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    try:
        get_server_args()
    except ValueError:
        return None
    hf_config = (
        process_model_config().hf_config if is_true_on_policy_enabled() else None
    )
    _SERVED_BLOCK.append(
        None
        if hf_config is None
        else checkpoint_block(getattr(hf_config, "quantization_config", None))
    )
    return _SERVED_BLOCK[0]


def enabled() -> bool:
    """TOP is serving a 128x128 block-FP8 checkpoint."""
    return served_block() == BLOCK


def enabled_b32() -> bool:
    """TOP is serving a 32x32 UE8M0 block-FP8 checkpoint (DeepSeek-V4.1)."""
    return served_block() == BLOCK32


def validate_quant_config(quant_config) -> None:
    """Reject serving-side quantization TOP cannot reproduce in the trainer."""
    if quant_config is None:
        return
    if enabled_b32():
        validate_b32_quant_config(quant_config)
        return
    if not enabled() or quant_config.get_name() != "fp8":
        raise NotImplementedError(
            "[top] only unquantized or block-wise FP8 (E4M3, dynamic, 128x128) "
            "checkpoints are supported"
        )
    if (
        quant_config.weight_block_size != BLOCK
        or quant_config.activation_scheme != "dynamic"
        or not quant_config.is_checkpoint_fp8_serialized
        or getattr(quant_config, "use_mxfp8", False)
    ):
        raise NotImplementedError("[top] FP8 requires serialized dynamic 128x128 E4M3")


def validate_b32_quant_config(quant_config) -> None:
    """The 32x32 program runs SM90 DeepGEMM's BLOCK32 GEMM; Blackwell serves 32x32 as MXFP8, which is not qualified."""
    if (
        quant_config.get_name() != "fp8"
        or quant_config.weight_block_size != BLOCK32
        or quant_config.activation_scheme != "dynamic"
        or not quant_config.is_checkpoint_fp8_serialized
        or getattr(quant_config, "use_mxfp8", False)
    ):
        raise NotImplementedError(
            "[top] FP8 32x32 requires serialized dynamic 32x32 E4M3 with UE8M0 scales"
        )
    if torch.cuda.get_device_capability() != (9, 0):
        raise NotImplementedError(
            "[top] FP8 32x32 is qualified on Hopper (SM90) only; Blackwell serves it as MXFP8"
        )


def uses_hopper_scales() -> bool:
    """SM90 DeepGEMM reads FP32 block scales instead of packed UE8M0 bytes.

    Scales are UE8M0 on every architecture; Hopper only changes their storage,
    so results are meant to match on Hopper and Blackwell.
    """
    return torch.cuda.get_device_capability() == (9, 0)


# Bumped by every serving entry point that writes weight bytes (weight updates,
# memory resume, check_weights); values derived under an older one are stale.
_WEIGHT_GENERATION: list[int] = [0]
# The generation the derived caches were last rewritten at (refresh_derived).
_REFRESHED: list[int] = [0]


def weights_changed() -> None:
    """Serving wrote weight bytes: values derived from the old ones are stale."""
    _WEIGHT_GENERATION[0] += 1


def refresh_derived() -> None:
    """Rewrite values derived from weight bytes after a write, while the weights are resident.

    Serving calls it when a weight write is complete, before the next forward:
    CUDA graph replays read the derived buffers without any Python call. A
    no-op compare when no weights changed, and when TOP FP8 cached nothing.
    """
    if _REFRESHED[0] != _WEIGHT_GENERATION[0]:
        from sglang.srt.true_on_policy.fp8.hopper_scales import refresh_cached_scales

        refresh_cached_scales()


def validate_device() -> None:
    major = torch.cuda.get_device_capability()[0]
    if major not in (9, 10):
        raise NotImplementedError(
            "[top] FP8 requires Hopper (SM90) or Blackwell (SM100)"
        )
    if major == 10:
        logging.getLogger(__name__).warning(
            "[top] FP8 on Blackwell (SM100) is admitted but not verified: packed-UE8M0 "
            "DeepGEMM, FlashMLA and DeepSeek-V4 wo_a are untested on SM100"
        )


def requant_ue8m0_experts(weight, weight_scale):
    """requant_ue8m0 for routed experts: also DeepSeek-V4.1's, cast from MXFP4 to 128x128 FP8 at load, so they
    reach DS4's FP8 expert program in its scale format (the cast's power-of-two scales requantize exactly)."""
    if enabled_b32():
        if not getattr(weight_scale, "format_ue8m0", False):
            _requant_ue8m0_in_place(weight, weight_scale)
            weight_scale.format_ue8m0 = True
        return True
    return requant_ue8m0(weight, weight_scale)


def requant_ue8m0(weight, weight_scale):
    """TOP's UE8M0 requant for DeepGEMM; False when TOP FP8 is off and the caller requants."""
    if not enabled():
        return False
    if not getattr(weight_scale, "format_ue8m0", False):
        _requant_ue8m0_in_place(weight, weight_scale)
        weight_scale.format_ue8m0 = True
    return True


def _requant_ue8m0_in_place(weight, weight_scale):
    """fp8_utils.requant_weight_ue8m0, written into the weight's own storage with temporaries outside its pool.

    torch_memory_saver's weight MemPool never returns freed blocks, so a replacement tensor or interleaved temporaries
    would strand memory there for good (~6.7 GB per GPU on GLM-5.2 at EP8, all of it taken from the KV pool).
    """
    from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

    assert weight.is_cuda and weight.data.is_contiguous(), (weight.device, weight.shape)
    *group_dims, n, k = weight.shape
    with _outside_weight_pool():
        raw = _requant_groups(
            weight.data.view(-1, n, k),
            weight_scale.data.reshape(-1, *weight_scale.shape[-2:]),
        )
        new_scales = transform_scale_ue8m0(
            raw.view(*group_dims, *raw.shape[-2:]) if group_dims else raw[0], mn=n
        )
        del raw
    kept = torch.empty_strided(
        new_scales.size(),
        new_scales.stride(),
        dtype=new_scales.dtype,
        device=new_scales.device,
    )
    kept.copy_(new_scales)
    del new_scales
    weight_scale.data = kept


def _requant_groups(groups, scales):
    """Requant each [n, k] group in place; returns the untransformed UE8M0 scales, [groups, *group scale shape]."""
    from sglang.srt.layers.quantization.fp8_utils import (
        block_quant_dequant,
        quant_weight_ue8m0,
    )

    new_scales = None
    for g in range(groups.shape[0]):
        dequant = block_quant_dequant(groups[g], scales[g], [128, 128], torch.bfloat16)
        w_g, s_g = quant_weight_ue8m0(
            weight_dequant=dequant, weight_block_size=[128, 128]
        )
        groups[g].copy_(w_g)
        if new_scales is None:
            new_scales = torch.empty(
                (groups.shape[0], *s_g.shape), dtype=s_g.dtype, device=s_g.device
            )
        new_scales[g] = s_g
    return new_scales


def _outside_weight_pool():
    """torch_memory_saver.disable() while a memory-saver region is active (weight load), else a no-op."""
    try:
        from torch_memory_saver import torch_memory_saver as tms

        active = tms._impl is not None and bool(
            tms._impl._binary_wrapper.cdll.tms_get_interesting_region()
        )
    except (ImportError, AttributeError):
        active = False
    return tms.disable() if active else contextlib.nullcontext()


def maybe_linear(method, layer, x, bias):
    """TOP FP8 linear for Fp8LinearMethod.apply; None when TOP FP8 is off."""
    if enabled_b32():
        from sglang.srt.true_on_policy.fp8.fp8_b32_linear import linear as linear_b32

        if (
            not method.block_quant
            or method.use_mxfp8
            or method.block_fp8_as_mxfp8
            or method.weight_block_size != BLOCK32
        ):
            raise NotImplementedError(
                "TOP FP8 32x32 linear requires the qualified 32x32 block format"
            )
        return linear_b32(x, layer.weight, layer.weight_scale_inv, bias)
    if not enabled():
        return None
    from sglang.srt.true_on_policy.fp8.fp8_linear import linear

    if (
        not method.block_quant
        or method.use_mxfp8
        or method.weight_block_size != [128, 128]
    ):
        raise NotImplementedError(
            "TOP FP8 linear requires the qualified block128 format"
        )
    return linear(x, layer.weight, layer.weight_scale_inv, bias, cache_scales=True)


def fp8_dispatch(*, quantized_experts: bool) -> bool:
    """TOP's FP8 expert program serves this MoE layer: both DeepEP modes send its quantize_expert_input bytes."""
    return quantized_experts and (enabled() or enabled_b32())


def maybe_expert_forward(
    runner_input, quant_info, *, weight_dtype, activation, swiglu_limit
):
    """TOP FP8 expert forward for the DeepGEMM MoE runner; None when TOP FP8 is off."""
    if not (enabled() or enabled_b32()):
        return None
    from sglang.srt.true_on_policy.fp8.fp8_expert_forward import expert_forward

    if (
        weight_dtype != torch.float8_e4m3fn
        or runner_input.hidden_states.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or activation != "silu"
        or swiglu_limit not in (None, 10.0)
    ):
        raise NotImplementedError(
            "TOP FP8 requires E4M3 SwiGLU experts (unclamped or DeepSeek-V4's 10) "
            "and BF16 or TOP-quantized FP8 dispatch"
        )
    masked = runner_input.use_masked_gemm
    prequantized = runner_input.hidden_states.dtype == torch.float8_e4m3fn
    if masked and not prequantized:
        raise NotImplementedError(
            "[top] FP8 experts take DeepEP's FP8 low-latency receive, not BF16 rows"
        )
    return expert_forward(
        runner_input.hidden_states,
        (quant_info.w13_weight, quant_info.w13_scale),
        (quant_info.w2_weight, quant_info.w2_scale),
        x_scale=runner_input.hidden_states_scale if prequantized else None,
        m_indices=None if masked else runner_input.m_indices,
        masked_m=runner_input.masked_m if masked else None,
        expected_m=runner_input.expected_m if masked else None,
        swiglu_limit=swiglu_limit,
        cache_scales=True,
    )


def maybe_clamped_mlp(mlp, x):
    """TOP FP8 clamped-SwiGLU dense MLP (DeepSeek-V4 / V4.1 shared expert); None otherwise."""
    if enabled_b32():
        return _clamped_mlp_b32(mlp, x)
    if not enabled() or mlp.swiglu_limit is None:
        return None
    from sglang.srt.true_on_policy.fp8.fp8_linear import clamped_swiglu_mlp

    down = mlp.down_proj
    if (
        mlp.swiglu_limit != 10.0
        or down.reduce_results
        or down.input_size_per_partition != down.input_size
        or not isinstance(x, torch.Tensor)
    ):
        raise NotImplementedError(
            "[top] FP8 clamped SwiGLU MLP requires full-K, unreduced BF16 projections"
        )
    return clamped_swiglu_mlp(
        x,
        (mlp.gate_up_proj.weight, mlp.gate_up_proj.weight_scale_inv),
        (down.weight, down.weight_scale_inv),
        mlp.swiglu_limit,
        cache_scales=True,
    )


def _clamped_mlp_b32(mlp, x):
    from sglang.srt.true_on_policy.fp8.fp8_b32_linear import clamped_swiglu_mlp

    down = mlp.down_proj
    if (
        mlp.swiglu_limit is None
        or down.reduce_results
        or down.input_size_per_partition != down.input_size
        or not isinstance(x, torch.Tensor)
    ):
        raise NotImplementedError(
            "[top] FP8 32x32 MLP requires the clamped SwiGLU over full-K, unreduced BF16 projections"
        )
    return clamped_swiglu_mlp(
        x,
        (mlp.gate_up_proj.weight, mlp.gate_up_proj.weight_scale_inv),
        (down.weight, down.weight_scale_inv),
        mlp.swiglu_limit,
    )
