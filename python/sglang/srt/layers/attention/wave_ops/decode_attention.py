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
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import math
import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.constraints import MMAType, GenericDot, MMAOperand
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
    paged_decode_attention_shape,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile

import logging

import functools

logger = logging.getLogger(__name__)
import os
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))


@functools.lru_cache(maxsize=4096)
def get_wave_kernel(
    shape: paged_decode_attention_shape,
    max_kv_splits,
    input_dtype,
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
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        max_kv_splits,
        input_dtype=input_dtype,
        mha=mha,
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=False,
        use_buffer_load_ops=False,
        use_buffer_store_ops=False,
        waves_per_eu=2,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=False,
        use_buffer_load_ops=False,
        use_buffer_store_ops=False,
        waves_per_eu=4,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=True,
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    return phase_0, phase_1


def view_trunc(tensor, shape):
    size = math.prod(shape)
    return tensor.view(-1)[:size].view(shape)


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
    logit_cap=0.0,
):
    num_seqs, num_query_heads, head_size = q.shape
    total_tokens, num_kv_heads, _ = k_buffer.shape
    _, _, head_size_kv = v_buffer.shape
    seq_len = total_tokens // num_seqs
    block_size = 32
    shape = paged_decode_attention_shape(
        num_query_heads,
        num_kv_heads,
        head_size,
        head_size_kv,
        block_size,
        num_seqs,
        seq_len,
    )

    k_buffer = view_trunc(k_buffer, (num_seqs, seq_len, num_kv_heads, head_size))
    v_buffer = view_trunc(v_buffer, (num_seqs, seq_len, num_kv_heads, head_size_kv))

    phase_0, phase_1 = get_wave_kernel(shape, max_kv_splits, q.dtype)

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
    # assert max_kv_splits == attn_logits.shape[2]
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
