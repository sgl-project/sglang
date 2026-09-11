"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

import functools
import logging

import torch
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import GenericDot, MMAOperand, MMAType
from wave_lang.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
    get_paged_decode_intermediate_arrays_shapes,
    paged_decode_attention_shape,
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

logger = logging.getLogger(__name__)
import os

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


@functools.lru_cache(maxsize=None)
def _is_rocm10_gfx942(device_index: int) -> bool:
    """Return whether a device needs the ROCm 10 Wave decode workaround."""
    hip_version = torch.version.hip
    if hip_version is None:
        return False

    try:
        hip_major_minor = tuple(int(part) for part in hip_version.split(".")[:2])
    except ValueError:
        return False

    arch = torch.cuda.get_device_properties(device_index).gcnArchName.split(":", 1)[0]
    # torch 2.11's ROCm 10 build reports HIP 7.15.
    return arch == "gfx942" and hip_major_minor >= (7, 15)


def _needs_triton_fallback(q, k_buffer, v_buffer) -> bool:
    # Wave's paged decode kernel returns NaNs for this shape on gfx942 with
    # ROCm 10. Keep Wave enabled for every other shape and architecture.
    shape = (q.shape[1], k_buffer.shape[1], q.shape[2], v_buffer.shape[2])
    if shape != (128, 1, 576, 512):
        return False

    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_rocm10_gfx942(device_index)


@functools.lru_cache(maxsize=4096)
def get_wave_kernel(
    shape: paged_decode_attention_shape,
    max_kv_splits,
    input_dtype,
    output_dtype,
    logit_cap,
):
    mha = (shape.num_query_heads // shape.num_kv_heads) == 1

    # Get the kernels (either compile or load from cache).
    if mha:
        mfma_variant = (
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),
        )
    else:
        mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)

    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_1,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        max_kv_splits,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        logit_cap=logit_cap,
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=False,
        use_buffer_ops=True,
        waves_per_eu=2,
        dynamic_symbols=dynamic_symbols_0,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=False,
        use_buffer_ops=False,
        waves_per_eu=4,
        dynamic_symbols=dynamic_symbols_1,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    return phase_0, phase_1


def decode_attention_intermediate_arrays_shapes(
    num_seqs, head_size_kv, num_query_heads, max_kv_splits
):
    # Not all fields are used, but we need to pass them to the function
    shape = paged_decode_attention_shape(
        num_query_heads=num_query_heads,
        num_kv_heads=0,
        head_size=0,
        head_size_kv=head_size_kv,
        block_size=0,
        num_seqs=num_seqs,
    )
    return get_paged_decode_intermediate_arrays_shapes(shape, max_kv_splits)


def decode_attention_wave(
    q,
    k_buffer,
    v_buffer,
    o,
    b_req_idx,
    req_to_token,
    attn_logits,
    attn_logits_max,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
):
    num_seqs, num_query_heads, head_size = q.shape
    _, num_kv_heads, _ = k_buffer.shape
    _, _, head_size_kv = v_buffer.shape

    if _needs_triton_fallback(q, k_buffer, v_buffer):
        # The Wave and Triton intermediates contain the same number of values,
        # but use different dimension orders. Reuse their storage so the
        # fallback does not add an allocation to the decode path.
        from sglang.kernels.ops.attention.decode_attention import (
            decode_attention_fwd_grouped as triton_decode_attention_fwd_grouped,
        )

        triton_decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            b_req_idx,
            req_to_token,
            attn_logits.reshape(num_seqs, num_query_heads, max_kv_splits, head_size_kv),
            attn_logits_max.reshape(num_seqs, num_query_heads, max_kv_splits),
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            1.0,
            logit_cap,
        )
        return

    block_size = 32
    shape = paged_decode_attention_shape(
        num_query_heads,
        num_kv_heads,
        head_size,
        head_size_kv,
        block_size,
        num_seqs,
    )

    phase_0, phase_1 = get_wave_kernel(
        shape, max_kv_splits, q.dtype, o.dtype, logit_cap
    )

    mb_qk = phase_0(
        q,
        k_buffer,
        v_buffer,
        b_req_idx,
        req_to_token,
        attn_logits,
        attn_logits_max,
    )
    if dump_generated_mlir:
        filename = f"wave_decode_attention_phase0_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())

    mb_sv = phase_1(attn_logits, attn_logits_max, b_req_idx, o)
    if dump_generated_mlir:
        filename = f"wave_decode_attention_phase1_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_sv.module_op.get_asm())


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    b_req_idx,
    req_to_token,
    attn_logits,
    attn_logits_max,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    decode_attention_wave(
        q,
        k_buffer,
        v_buffer,
        o,
        b_req_idx,
        req_to_token,
        attn_logits,
        attn_logits_max,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
