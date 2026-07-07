"""Correctness tests for the opt-in DSA MXFP4 index-K cache (--enable-dsa-fp4-indexer).

Covers the three contracts the DSA FP4 index path relies on:
  1. The Triton FP4 quantizer emits exactly DeepGEMM's packed MXFP4 format
     (E2M1 pairs + packed UE8M0/32 scales).
  2. store_fp4_index_k_cache and the index_buf_accessor gather agree bit-exactly
     on the 68 B/token paged layout (write path == ragged read path).
  3. The DeepGEMM FP4 MQA-logits kernels (ragged + paged) track the FP8 baseline
     within quantization noise, measured as top-k index-set overlap — the metric
     that decides indexer selection quality (SM100+ only).
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.layers.attention.dsa.index_buf_accessor import (
    _get_k_and_s_triton,
    _set_k_and_s_triton,
)
from sglang.srt.layers.attention.dsa.triton_kernel import act_quant
from sglang.srt.layers.attention.dsv4.fp4_indexer import (
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

HEAD_DIM = 128
PAGE_SIZE = 64
FP4_BYTES = HEAD_DIM // 2
INDEX_TOPK = 2048

requires_sm100 = pytest.mark.skipif(
    not is_sm100_supported(),
    reason="DeepGEMM FP4 MQA-logits kernels require SM100+",
)


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    ia = a.topk(k, dim=-1).indices
    ib = b.topk(k, dim=-1).indices
    overlaps = []
    for i in range(a.shape[0]):
        sa = set(ia[i].tolist())
        sb = set(ib[i].tolist())
        overlaps.append(len(sa & sb) / k)
    return sum(overlaps) / len(overlaps)


def test_quantizer_matches_deepgemm_reference() -> None:
    from deep_gemm.utils.math import per_token_cast_to_fp4

    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 2.0

    k_fp4, k_sf = quantize_fp4_indexer_tensor(x)
    assert k_fp4.shape == (n, FP4_BYTES) and k_fp4.dtype == torch.int8
    assert k_sf.shape == (n,) and k_sf.dtype == torch.int32

    ref_fp4, ref_sf = per_token_cast_to_fp4(
        x.float(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    assert torch.equal(k_fp4, ref_fp4)
    assert torch.equal(k_sf, ref_sf.view(-1))


def test_store_and_gather_round_trip() -> None:
    torch.manual_seed(1)
    num_tokens = 8 * 1024
    total_pages = num_tokens // PAGE_SIZE
    k = torch.randn(num_tokens, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    cache = torch.zeros(
        total_pages, PAGE_SIZE * (FP4_BYTES + 4), dtype=torch.uint8, device="cuda"
    )
    # Scatter through a permutation to exercise non-linear cache locations.
    loc = torch.randperm(num_tokens, device="cuda")
    store_fp4_index_k_cache(k, cache, loc, page_size=PAGE_SIZE)

    # Gather everything back in loc order via the accessor kernel at the FP4
    # byte width (the same call _get_topk_ragged makes through GetKAndS).
    inv = torch.empty_like(loc)
    inv[loc] = torch.arange(num_tokens, device="cuda")
    seq_lens = torch.tensor([num_tokens], dtype=torch.int64, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        1, -1
    )
    k_out, s_out = _get_k_and_s_triton(
        buf=cache,
        page_indices=block_tables,
        seq_lens=seq_lens,
        seq_len_sum=num_tokens,
        max_seq_len=num_tokens,
        page_size=PAGE_SIZE,
        index_head_dim=FP4_BYTES,
    )

    k_ref, sf_ref = quantize_fp4_indexer_tensor(k)
    assert torch.equal(k_out.view(torch.int8)[loc], k_ref)
    assert torch.equal(s_out.view(torch.int32).squeeze(-1)[loc], sf_ref)


@requires_sm100
def test_ragged_logits_fp4_tracks_fp8() -> None:
    import deep_gemm

    torch.manual_seed(2)
    m, h, n_kv = 128, 64, 8192
    q = torch.randn(m, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(n_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(m, h, device="cuda", dtype=torch.float32)
    ks = torch.zeros(m, dtype=torch.int32, device="cuda")
    ke = torch.full((m,), n_kv, dtype=torch.int32, device="cuda")

    # FP8 baseline arm: per-token q_scale folded into the head-gate weights,
    # exactly like the production FP8 path.
    q_fp8, q_scale = act_quant(q, 128, None)
    k_fp8, k_scale = act_quant(kv, 128, None)
    logits_fp8 = deep_gemm.fp8_mqa_logits(
        q_fp8,
        (k_fp8, k_scale.squeeze(-1)),
        (w.unsqueeze(-1) * q_scale).squeeze(-1),
        ks,
        ke,
        clean_logits=False,
    )

    # FP4 arm: kernel consumes q_sf / k_sf directly, weights carry no q_scale.
    q4, qsf = quantize_fp4_indexer_tensor(q.view(-1, HEAD_DIM))
    k4, ksf = quantize_fp4_indexer_tensor(kv)
    logits_fp4 = deep_gemm.fp8_fp4_mqa_logits(
        (q4.view(m, h, FP4_BYTES), qsf.view(m, h)),
        (k4, ksf),
        w,
        ks,
        ke,
        clean_logits=False,
        max_seqlen_k=n_kv,
    )

    lf8, lf4 = logits_fp8.float(), logits_fp4.float()
    finite = torch.isfinite(lf8) & torch.isfinite(lf4)
    corr = torch.corrcoef(torch.stack([lf8[finite], lf4[finite]]))[0, 1].item()
    overlap = _topk_overlap(lf8, lf4, INDEX_TOPK)
    assert corr > 0.97, f"FP4 ragged logits do not track FP8 (corr={corr:.4f})"
    assert overlap > 0.80, f"FP4 ragged top-k overlap too low ({overlap:.4f})"


@requires_sm100
def test_paged_logits_fp4_tracks_fp8_and_matches_ragged() -> None:
    import deep_gemm

    torch.manual_seed(3)
    B, h, ctx = 4, 64, 4096
    total_pages = B * ctx // PAGE_SIZE
    kv_all = torch.randn(B * ctx, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    loc = torch.arange(B * ctx, dtype=torch.int64, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        B, -1
    )

    cache_fp4 = torch.zeros(
        total_pages, PAGE_SIZE * (FP4_BYTES + 4), dtype=torch.uint8, device="cuda"
    )
    store_fp4_index_k_cache(kv_all, cache_fp4, loc, page_size=PAGE_SIZE)

    cache_fp8 = torch.zeros(
        total_pages, PAGE_SIZE * (HEAD_DIM + 4), dtype=torch.uint8, device="cuda"
    )
    k8, k8s = act_quant(kv_all, 128, None)
    _set_k_and_s_triton(
        buf=cache_fp8, loc=loc, index_k=k8, index_k_scale=k8s, page_size=PAGE_SIZE
    )

    qd = torch.randn(B, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    wd = torch.rand(B, h, device="cuda", dtype=torch.float32)
    ctx_lens = torch.full((B, 1), ctx, dtype=torch.int32, device="cuda")
    sched = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_lens, PAGE_SIZE, deep_gemm.get_num_sms()
    )

    qd8, qd8s = act_quant(qd, 128, None)
    logits_p8 = deep_gemm.fp8_paged_mqa_logits(
        qd8.unsqueeze(1),
        cache_fp8.view(total_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
        (wd.unsqueeze(-1) * qd8s).squeeze(-1),
        ctx_lens,
        block_tables,
        sched,
        ctx,
        clean_logits=False,
    )

    qd4, qd4sf = quantize_fp4_indexer_tensor(qd.view(-1, HEAD_DIM))
    qd4 = qd4.view(B, h, FP4_BYTES)
    qd4sf = qd4sf.view(B, h)
    logits_p4 = deep_gemm.fp8_fp4_paged_mqa_logits(
        (qd4.unsqueeze(1), qd4sf.unsqueeze(1)),
        cache_fp4.view(total_pages, PAGE_SIZE, 1, FP4_BYTES + 4),
        wd,
        ctx_lens,
        block_tables,
        sched,
        ctx,
        clean_logits=False,
    )

    lp8, lp4 = logits_p8.float(), logits_p4.float()
    finite = torch.isfinite(lp8) & torch.isfinite(lp4)
    corr = torch.corrcoef(torch.stack([lp8[finite], lp4[finite]]))[0, 1].item()
    overlap = _topk_overlap(lp8, lp4, INDEX_TOPK)
    assert corr > 0.97, f"FP4 paged logits do not track FP8 (corr={corr:.4f})"
    assert overlap > 0.80, f"FP4 paged top-k overlap too low ({overlap:.4f})"

    # Ragged and paged FP4 kernels must agree on identical inputs: the prefill
    # (ragged) and decode (paged) paths read the same cache and must select the
    # same tokens.
    k4_all, ksf_all = quantize_fp4_indexer_tensor(kv_all)
    ks1 = torch.zeros(1, dtype=torch.int32, device="cuda")
    ke1 = torch.full((1,), ctx, dtype=torch.int32, device="cuda")
    for i in range(B):
        li = deep_gemm.fp8_fp4_mqa_logits(
            (qd4[i : i + 1], qd4sf[i : i + 1]),
            (
                k4_all[i * ctx : (i + 1) * ctx],
                ksf_all[i * ctx : (i + 1) * ctx],
            ),
            wd[i : i + 1],
            ks1,
            ke1,
            clean_logits=False,
            max_seqlen_k=ctx,
        )
        diff = (li[0, :ctx] - lp4[i, :ctx]).abs()
        fin = torch.isfinite(li[0, :ctx]) & torch.isfinite(lp4[i, :ctx])
        assert diff[fin].max().item() < 1e-3, "ragged/paged FP4 logits diverge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
