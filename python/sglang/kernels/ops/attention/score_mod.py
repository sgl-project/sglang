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
"""Generic score_mod for the Triton attention kernels, mirroring FA4's
``score_mod``/``aux_tensors``. A Triton score_mod is a ``@triton.jit`` function
inlined into the kernels as a constexpr argument:

    score_mod(qk, q_pos, kv_pos, q_idx, head, mask,
              Aux0, aux0_stride_t, aux0_stride_h, aux0_len) -> qk

The kernels pre-broadcast q_pos/kv_pos/q_idx/head to ``qk``'s shape, so an
elementwise score_mod works at every call site. ``aux_tensors`` supports one
3D tensor ``[num_q_tokens, num_q_heads, D]`` with a contiguous last dim.
"""

import triton
import triton.language as tl


def unpack_aux_tensors(score_mod, aux_tensors):
    if score_mod is None:
        return None, 0, 0, 0
    assert (
        aux_tensors is not None and len(aux_tensors) == 1
    ), "Triton score_mod currently requires exactly one aux tensor"
    aux0 = aux_tensors[0]
    assert aux0.dim() == 3 and aux0.stride(2) == 1, (
        f"aux_tensors[0] must be 3D with a contiguous last dim, "
        f"got shape={tuple(aux0.shape)} stride={aux0.stride()}"
    )
    return aux0, aux0.stride(0), aux0.stride(1), aux0.shape[2]


@triton.jit
def relative_bias_score_mod(
    qk, q_pos, kv_pos, q_idx, head, mask, Aux0, aux0_stride_t, aux0_stride_h, aux0_len
):
    """Add ``Aux0[q_idx, head, q_pos - kv_pos]`` when 0 <= q_pos - kv_pos < aux0_len."""
    rel_dist = q_pos - kv_pos
    rel_idx = tl.minimum(tl.maximum(rel_dist, 0), aux0_len - 1)
    bias = tl.load(
        Aux0 + q_idx * aux0_stride_t + head * aux0_stride_h + rel_idx,
        mask=mask & (rel_dist >= 0) & (rel_dist < aux0_len),
        other=0.0,
    )
    return qk + bias
