"""
Memory-efficient attention for prefill.
It support page size = 1.
"""

import functools
import os

import torch
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.templates.extend_attention import get_extend_attention_kernel
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


@functools.lru_cache
def get_wave_kernel(
    shape: AttentionShape,
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    size_dtype: torch.dtype,
    is_causal: bool,
    logit_cap: float,
    layer_scaling: float,
):
    assert shape.num_query_heads % shape.num_kv_heads == 0

    mfma_variant = (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16)
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        size_dtype=size_dtype,
        is_causal=is_causal,
        layer_scaling=layer_scaling,
        logit_cap=logit_cap,
    )

    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        use_buffer_ops=True,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    extend_attention = wave_compile(options, extend_attention)

    return extend_attention


def extend_attention_wave(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_indptr,
    max_seq_len,
    output,
    is_causal=True,
    layer_scaling=None,
    logit_cap=0,
):
    shape = AttentionShape(
        num_query_heads=q_extend.shape[1],
        num_kv_heads=k_extend.shape[1],
        head_size=q_extend.shape[2],
        head_size_kv=k_extend.shape[2],
        num_seqs=kv_indptr.shape[0] - 1,
        max_seq_len=max_seq_len,
    )

    # Run the wave kernel.
    extend_attention = get_wave_kernel(
        shape,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        input_dtype=q_extend.dtype,
        output_dtype=output.dtype,
        size_dtype=qo_indptr.dtype,
        is_causal=is_causal,
        layer_scaling=layer_scaling,
        logit_cap=logit_cap,
    )

    mb = extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        max_seq_len,
        output,
    )

    if dump_generated_mlir:
        shape_list = [
            q_extend.shape[0],
            q_extend.shape[1],
            k_extend.shape[1],
            q_extend.shape[2],
            k_extend.shape[2],
        ]
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
        with open(filename, "w") as f:
            f.write(mb.module_op.get_asm())
