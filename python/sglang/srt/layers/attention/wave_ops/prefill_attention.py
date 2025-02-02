# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1

import torch
import math

import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import MMAType

from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)

import os
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))

def prefill_attention_wave(
    q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=True
):

    # if not is_causal:
    #     raise NotImplementedError("non causal mask not supported yet on prefill_attention wave backend.")
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
    mfma_variant =(MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
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
    config = get_default_run_config()

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
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
