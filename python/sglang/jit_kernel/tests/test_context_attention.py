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
Correctness tests for context_attention_fwd (Triton prefill attention kernel).
Compares against F.scaled_dot_product_attention across causal/non-causal,
GQA, and varying head dimensions.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
ATOL = 1e-2
RTOL = 1e-2


def ref_context_attention(q_packed, k_packed, v_packed, b_seq_len, is_causal):
    """
    Reference implementation using F.scaled_dot_product_attention per sequence.
    q_packed, k_packed, v_packed: [total_tokens, num_heads, head_dim]
    b_seq_len: list of ints
    Returns: [total_tokens, num_q_heads, head_dim]
    """
    num_q_heads = q_packed.shape[1]
    num_kv_heads = k_packed.shape[1]
    head_dim = q_packed.shape[2]
    outputs = []
    offset = 0
    for seq_len in b_seq_len:
        q = q_packed[offset : offset + seq_len]  # [s, nq, d]
        k = k_packed[offset : offset + seq_len]  # [s, nk, d]
        v = v_packed[offset : offset + seq_len]  # [s, nk, d]

        # SDPA expects [batch, heads, seq, dim]
        q_ = q.transpose(0, 1).unsqueeze(0).to(torch.float32)  # [1, nq, s, d]
        k_ = k.transpose(0, 1).unsqueeze(0).to(torch.float32)  # [1, nk, s, d]
        v_ = v.transpose(0, 1).unsqueeze(0).to(torch.float32)  # [1, nk, s, d]

        out = F.scaled_dot_product_attention(
            q_, k_, v_, is_causal=is_causal, enable_gqa=(num_q_heads != num_kv_heads)
        )  # [1, nq, s, d]
        out = out.squeeze(0).transpose(0, 1)  # [s, nq, d]
        outputs.append(out)
        offset += seq_len

    return torch.cat(outputs, dim=0).to(q_packed.dtype)


BATCH_SEQ_CONFIGS = get_ci_test_range(
    [
        ([64], 64),
        ([128], 64),
        ([64, 32], 64),
        ([128, 64, 32], 128),
        ([512], 512),
        ([256, 128], 256),
    ],
    [([64], 64), ([64, 32], 64), ([256, 128], 256)],
)

HEAD_DIM_LIST = get_ci_test_range([64, 128], [64, 128])

NUM_HEADS_CONFIGS = get_ci_test_range(
    [(8, 8), (8, 2), (8, 1)],  # (num_q_heads, num_kv_heads)
    [(8, 8), (8, 2)],
)

DTYPE_LIST = [torch.float16]


@pytest.mark.parametrize("seq_config", BATCH_SEQ_CONFIGS)
@pytest.mark.parametrize("head_dim", HEAD_DIM_LIST)
@pytest.mark.parametrize("num_heads", NUM_HEADS_CONFIGS)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_context_attention_fwd(seq_config, head_dim, num_heads, is_causal, dtype):
    from sglang.srt.layers.attention.triton_ops.prefill_attention import (
        context_attention_fwd,
    )

    b_seq_len_list, max_input_len = seq_config
    num_q_heads, num_kv_heads = num_heads

    total_tokens = sum(b_seq_len_list)
    batch = len(b_seq_len_list)

    q = torch.randn(total_tokens, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    o = torch.empty_like(q)

    b_start_loc = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    b_seq_len = torch.tensor(b_seq_len_list, dtype=torch.int32, device=DEVICE)
    for i in range(1, batch):
        b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

    context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=is_causal
    )

    ref = ref_context_attention(q, k, v, b_seq_len_list, is_causal=is_causal)

    assert torch.allclose(o, ref, atol=ATOL, rtol=RTOL), (
        f"max error: {(o - ref).abs().max().item():.4e} "
        f"seq_lens={b_seq_len_list} head_dim={head_dim} "
        f"num_heads={num_heads} causal={is_causal} dtype={dtype}"
    )
