"""
Memory-efficient attention for prefill.
It support page size = 1.
"""

import math
import os

from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from wave_lang.kernel.wave.utils.general_utils import get_default_scheduling_params
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


def prefill_attention_wave(
    q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=True
):

    shape = AttentionShape(
        num_query_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_size=q.shape[2],
        head_size_kv=k.shape[2],
        num_seqs=b_seq_len.shape[0],
        max_seq_len=max_seq_len,
        total_seq_len=q.shape[0],
    )

    assert shape.num_query_heads % shape.num_kv_heads == 0

    output_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size_kv)
    # Run the wave kernel.
    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
    (prefill, hyperparams) = get_prefill_attention_kernel(
        shape,
        mfma_variant,
        q.shape,
        k.shape,
        v.shape,
        output_shape,
        input_dtype=q.dtype,
        output_dtype=o.dtype,
        size_dtype=b_seq_len.dtype,
    )

    hyperparams.update(get_default_scheduling_params())

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        use_scheduling_barriers=False,
    )
    options = set_default_run_config(options)
    prefill = wave_compile(options, prefill)

    mb = prefill(
        q * dk_sqrt * log2e,
        k,
        v,
        b_start_loc,
        b_seq_len,
        o,
    )
    if dump_generated_mlir:
        shape_list = [q.shape[0], q.shape[1], k.shape[1], q.shape[2], k.shape[2]]
        filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
        with open(filename, "w") as f:
            f.write(mb.module_op.get_asm())
