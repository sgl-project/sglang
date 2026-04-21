"""FA3 numerical parity for MLA prefill CP.

Verifies the rank-local zigzag-split FA3 path (``_mla_cp_attn`` +
``cp_attn_forward_extend`` in ``flashattention_backend.py``) matches a
single non-CP ``flash_attn_with_kvcache`` over the full sequence.

Single-process, single-layer, pre-populated paged KV cache. Requires
FA3 ver=3 (Hopper+).
"""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.utils.cp_utils import (
    ContextParallelMetadata,
    cp_attn_forward_extend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip(reason="CUDA required for FA3", allow_module_level=True)

_cap = torch.cuda.get_device_capability(0)
if _cap[0] < 9:
    pytest.skip(
        reason=f"FA3 ver=3 requires Hopper (sm90+); got sm{_cap[0]}{_cap[1]}",
        allow_module_level=True,
    )

try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
except ImportError as e:
    pytest.skip(
        reason=f"sgl_kernel.flash_attn unavailable: {e}",
        allow_module_level=True,
    )

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Default shape is DeepSeek V3/R1 TP=8 MLA: 16 heads, v=512, rope=64.
NUM_HEADS = 16
V_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
PAGE_SIZE = 1


def _build_cache_and_q(seq_len):
    """Pre-populated paged KV cache + full-sequence q.

    Pre-population mirrors upstream ``rebuild_cp_kv_cache``, which all-gathers
    rank-local KV into the global pool before the attention call, so each
    rank's FA3 invocation sees the same fully-populated cache.
    """
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    c_kv_cache = torch.randn(
        num_pages, PAGE_SIZE, 1, V_HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    k_rope_cache = torch.randn(
        num_pages, PAGE_SIZE, 1, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    q_nope = torch.randn(seq_len, NUM_HEADS, V_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    q_rope = torch.randn(
        seq_len, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    return c_kv_cache, k_rope_cache, q_nope, q_rope, page_table


def _full_seq_attn(
    seq_len, q_nope, q_rope, c_kv_cache, k_rope_cache, page_table, softmax_scale
):
    """Non-CP reference: single flash_attn_with_kvcache over the full seq."""
    return flash_attn_with_kvcache(
        q=q_rope,
        qv=q_nope,
        k_cache=k_rope_cache,
        v_cache=c_kv_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE),
        cu_seqlens_k_new=None,
        max_seqlen_q=seq_len,
        softmax_scale=softmax_scale,
        causal=True,
        ver=3,
    )


def _cp_attn_for_rank(
    rank,
    cp_size,
    block_size,
    q_nope,
    q_rope,
    c_kv_cache,
    k_rope_cache,
    page_table,
    softmax_scale,
):
    """Run the rank-local CP closure from ``flashattention_backend.py``.

    Zigzag layout: rank r gets blocks [r, num_blocks - 1 - r] where
    num_blocks = cp_size * 2. kv_len for each half is the cumulative KV
    extent through the end of that block.
    """
    num_blocks = cp_size * 2
    b_prev, b_next = rank, num_blocks - 1 - rank
    prev_slice = slice(b_prev * block_size, (b_prev + 1) * block_size)
    next_slice = slice(b_next * block_size, (b_next + 1) * block_size)

    q_nope_local = torch.cat([q_nope[prev_slice], q_nope[next_slice]], dim=0)
    q_rope_local = torch.cat([q_rope[prev_slice], q_rope[next_slice]], dim=0)
    q_fused = torch.cat([q_nope_local, q_rope_local], dim=-1)

    cp_meta = ContextParallelMetadata(
        kv_len_prev_tensor=torch.tensor(
            [(b_prev + 1) * block_size], dtype=torch.int32, device=DEVICE
        ),
        kv_len_next_tensor=torch.tensor(
            [(b_next + 1) * block_size], dtype=torch.int32, device=DEVICE
        ),
        actual_seq_q_prev=block_size,
        actual_seq_q_next=block_size,
    )
    fb = SimpleNamespace(attn_cp_metadata=cp_meta)

    def _mla_cp_attn(q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp):
        q_nope_chunk = q_chunk[..., :V_HEAD_DIM]
        q_rope_chunk = q_chunk[..., V_HEAD_DIM:]
        return flash_attn_with_kvcache(
            q=q_rope_chunk,
            qv=q_nope_chunk,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens_cp,
            cu_seqlens_q=cu_seqlens_q_cp,
            cu_seqlens_k_new=None,
            max_seqlen_q=max_seqlen_q_cp,
            softmax_scale=softmax_scale,
            causal=True,
            ver=3,
        )

    local_out = cp_attn_forward_extend(fb, q_fused, DEVICE, _mla_cp_attn)
    return local_out, prev_slice, next_slice


@pytest.mark.parametrize(
    "cp_size, block_size",
    [
        (2, 64),  # DSv3 TP=8 baseline
        (2, 128),  # longer per-block seq
        (4, 32),  # multi-rank zigzag: rank r gets blocks [r, 7-r]
    ],
)
def test_cp_parity(cp_size, block_size):
    torch.manual_seed(0)
    seq_len = block_size * cp_size * 2
    softmax_scale = 1.0 / math.sqrt(V_HEAD_DIM + QK_ROPE_HEAD_DIM)

    c_kv_cache, k_rope_cache, q_nope, q_rope, page_table = _build_cache_and_q(seq_len)
    ref_out = _full_seq_attn(
        seq_len, q_nope, q_rope, c_kv_cache, k_rope_cache, page_table, softmax_scale
    )

    for rank in range(cp_size):
        local_out, prev_slice, next_slice = _cp_attn_for_rank(
            rank,
            cp_size,
            block_size,
            q_nope,
            q_rope,
            c_kv_cache,
            k_rope_cache,
            page_table,
            softmax_scale,
        )
        torch.testing.assert_close(
            local_out[:block_size],
            ref_out[prev_slice],
            rtol=1e-3,
            atol=5e-3,
            msg=f"rank={rank} prev-half mismatch",
        )
        torch.testing.assert_close(
            local_out[block_size:],
            ref_out[next_slice],
            rtol=1e-3,
            atol=5e-3,
            msg=f"rank={rank} next-half mismatch",
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
