"""Benchmark: the two DCP target-verify helpers, Triton vs torch.

Both run on the speculative target-verify path only, and only on dcp_rank 0:

* stage B  -- ``dense_causal_mla_attn_base2``: dense causal attention over the
  in-hand ``gamma + 1`` window, returning ``(out, base-2 lse)``.
* the merge -- ``lse_combine_base2``: folds stage B into stage A (the rank's
  committed KV shard) over disjoint key sets.

Both were torch before; the ``torch`` provider below is the implementation they
replaced, kept here so the swap stays measurable. The merge in particular was
~10 elementwise launches, which is what a single Triton program should win back
at these shapes.

``extend_attention`` is a third provider for stage B only: the generic
``extend_attention_fwd(is_causal=True, skip_prefix=True, lse_extend=...)``
computes the same thing and would have needed no new kernel, so its cost is
worth having on record. It turned out to be both slower here (it carries the
whole paged-prefix machinery) and not the drop-in it looks like -- ``k_buffer``,
``v_buffer``, ``kv_indices``, ``k_scale`` and ``v_scale`` must all be non-None
even with ``skip_prefix=True``.

Shapes are Kimi-K3 at tp8 dcp8: 96 gathered heads, kv_lora_rank 512,
qk_rope_head_dim 64, bf16.
"""

import math

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dcp_kernels import (
    dense_causal_mla_attn_base2,
    lse_combine_base2,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-benchmark", runner_config="amd")

KV_LORA_RANK = 512
QK_ROPE = 64
D = KV_LORA_RANK + QK_ROPE
NUM_HEADS = 96  # K3 gathered heads at tp8 dcp8
DTYPE = torch.bfloat16
DEV = "cuda"

_LOG2E = 1.4426950408889634


def _torch_dense_causal(q, k_window, scaling, bs, q_len, kv_lora_rank):
    """The torch stage B this kernel replaced."""
    num_heads = q.shape[1]
    qb = q.view(bs, q_len, num_heads, -1)
    kb = k_window.view(bs, q_len, -1)
    scores = torch.einsum("bihd,bjd->bhij", qb, kb).float() * scaling
    causal = torch.ones(q_len, q_len, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse_e = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse_e.unsqueeze(-1))
    vb = kb[..., :kv_lora_rank].float()
    out = torch.einsum("bhij,bjd->bihd", probs, vb)
    return (
        out.reshape(bs * q_len, num_heads, kv_lora_rank),
        (lse_e * _LOG2E).permute(0, 2, 1).reshape(bs * q_len, num_heads),
    )


def _torch_lse_combine(out_a, lse_a, out_b, lse_b, out_dtype):
    """The torch merge this kernel replaced (~10 elementwise launches)."""
    m = torch.maximum(lse_a, lse_b)
    w_a = torch.nan_to_num(torch.exp2(lse_a - m), nan=0.0, posinf=0.0, neginf=0.0)
    w_b = torch.nan_to_num(torch.exp2(lse_b - m), nan=0.0, posinf=0.0, neginf=0.0)
    denom = w_a + w_b
    out = out_a.float() * w_a.unsqueeze(-1) + out_b.float() * w_b.unsqueeze(-1)
    out = out / denom.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    lse = torch.where(denom == 0.0, float("-inf"), m + torch.log2(denom))
    return out.to(out_dtype), lse


def _extend_attention_dense_causal(q, k_window, scaling, bs, q_len, kv_lora_rank):
    """Same stage B through the generic extend kernel, for comparison."""
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    n_rows = bs * q_len
    k2 = k_window.view(n_rows, 1, D)
    v2 = k2[..., :kv_lora_rank]
    out = torch.empty(n_rows, q.shape[1], kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(n_rows, q.shape[1], dtype=torch.float32, device=q.device)
    qo_indptr = torch.arange(0, n_rows + 1, q_len, dtype=torch.int32, device=q.device)
    zeros = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    extend_attention_fwd(
        q,
        k2,
        v2,
        out,
        # k_buffer/v_buffer are never read (skip_prefix=True, kv_indptr all
        # zeros) but cannot be None: extend_attention_fwd extracts their strides
        # unconditionally. So it is not quite the drop-in it looks like.
        k2,
        v2,
        qo_indptr,
        zeros,  # kv_indptr: no committed prefix
        torch.zeros(1, dtype=torch.int32, device=q.device),  # kv_indices
        None,  # custom_mask
        True,  # is_causal
        None,  # mask_indptr
        q_len,  # max_len_extend
        1.0,  # k_scale -- also cannot be None, it is applied unconditionally
        1.0,  # v_scale
        sm_scale=scaling,
        skip_prefix=True,
        lse_extend=lse,
    )
    return out, lse * _LOG2E


@marker.parametrize("q_len", [2, 4, 8, 16], [2, 8])
@marker.parametrize("bs", [1, 8, 32, 64], [1, 32])
@marker.benchmark("impl", ["triton", "torch"])
def benchmark_lse_combine(bs: int, q_len: int, impl: str):
    n_rows = bs * q_len
    out_a = torch.randn(n_rows, NUM_HEADS, KV_LORA_RANK, dtype=DTYPE, device=DEV)
    out_b = torch.randn(n_rows, NUM_HEADS, KV_LORA_RANK, dtype=DTYPE, device=DEV)
    lse_a = torch.randn(n_rows, NUM_HEADS, dtype=torch.float32, device=DEV)
    lse_b = torch.randn(n_rows, NUM_HEADS, dtype=torch.float32, device=DEV)
    fn = lse_combine_base2 if impl == "triton" else _torch_lse_combine
    return marker.do_bench(
        fn,
        input_args=(out_a, lse_a, out_b, lse_b, DTYPE),
        memory_args=(out_a, out_b),
    )


@marker.parametrize("q_len", [2, 4, 8, 16], [2, 8])
@marker.parametrize("bs", [1, 8, 32, 64], [1, 32])
@marker.benchmark("impl", ["triton", "torch", "extend_attention"])
def benchmark_dense_causal(bs: int, q_len: int, impl: str):
    n_rows = bs * q_len
    q = torch.randn(n_rows, NUM_HEADS, D, dtype=DTYPE, device=DEV) * 0.3
    k_window = torch.randn(n_rows, 1, D, dtype=DTYPE, device=DEV) * 0.3
    scaling = 1.0 / math.sqrt(D)
    fn = {
        "triton": dense_causal_mla_attn_base2,
        "torch": _torch_dense_causal,
        "extend_attention": _extend_attention_dense_causal,
    }[impl]
    return marker.do_bench(
        fn,
        input_args=(q, k_window, scaling, bs, q_len, KV_LORA_RANK),
        memory_args=(q, k_window),
    )


if __name__ == "__main__":
    benchmark_lse_combine.run()
    benchmark_dense_causal.run()
