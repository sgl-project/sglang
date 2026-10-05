"""DeepSeek-V3.2 lightning-indexer logits through the ``dsa_indexer`` Cake route.

Registry resolution of the four DeepGEMM-signature entries, the admission
rules on the exact engine tensors, and the Cake FlashInfer entries against
DeepGEMM as the oracle on both engine paths:

* ragged prefill: ``fp8_mqa_logits(q, (kv, kv_scales), weights, ks, ke,
  clean_logits=False)`` -- finite logits within FP32-accumulation tolerance of
  exact FP8 products (atol 1e-2 of the row max, rtol 1e-2), the window mask
  exact, and equality of the top-k index *sets* after the engine's ``+inf``
  init / local-token scatter (exact away from the selection boundary, where
  FP32 accumulation order may swap values within the logits tolerance);
* paged decode / verify: ``get_paged_mqa_logits_metadata`` +
  ``fp8_paged_mqa_logits`` with the same gates (skips while the installed
  FlashInfer catalog has no paged route for the shape);
* the ``Indexer`` helper methods pick the Cake entry when the route admits and
  the stock DeepGEMM call otherwise, on the real tensors;
* both route helpers captured in a CUDA graph (the engine's decode / verify
  path) replay the eager bits -- the FlashInfer entries launch on the capture
  stream, an empty capture fails loudly.

GPU tests skip when FlashInfer lacks the modules, the device is outside
sm_100a / sm_103a, or DeepGEMM is not importable. Shapes the installed catalog
does not serve are skipped with the admission reason (today: 32 heads and
``K % 256 == 0``; 64-head, KV-tail and paged shapes admit as their programs
are exported).
"""

import os
import sys
from unittest import mock

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_sparse as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_fp8_mqa_logits,
    cake_fp8_paged_mqa_logits,
    cake_get_paged_mqa_logits_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "fp8_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "fp8_paged_mqa_logits",
    "prepare_paged_mqa_logits",
)
HEAD_DIM = 128
INIT_TOKENS, LOCAL_TOKENS = 1, 2  # DeepSeek-V3.2 indexer masking constants


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_sparse:")
    assert spec.load() is not None


def _skip_unless_device(*modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.MQA_ARCHS:
        pytest.skip(
            f"Cake indexer programs are built for {cake.MQA_ARCHS}, device is {cc}"
        )


def _deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        pytest.skip("deep_gemm (the oracle) is not importable")
    for name in (
        "fp8_mqa_logits",
        "fp8_paged_mqa_logits",
        "get_paged_mqa_logits_metadata",
    ):
        if not hasattr(deep_gemm, name):
            pytest.skip(f"deep_gemm build lacks {name}")
    return deep_gemm


def _assert_logits_close(got, ref):
    """FP32 accumulation of exact FP8 products: atol 1e-2 of the row max, rtol 1e-2,
    identical finiteness pattern on the compared cells."""
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(got), finite)
    row_max = (
        ref.masked_fill(~finite, 0.0).abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    )
    err = (got - ref).abs().masked_fill(~finite, 0.0)
    tol = 1e-2 * row_max + 1e-2 * ref.abs().masked_fill(~finite, 0.0)
    assert bool((err <= tol).all()), f"max |err| {float(err.max())}"


def _mask_init_and_local_tokens(logits, lengths, row_starts=None):
    """The engine's ``Indexer._mask_init_and_local_tokens`` (forced includes)."""
    if row_starts is None:
        row_starts = lengths.new_zeros(lengths.shape[0])
    init_idxs = (
        torch.arange(INIT_TOKENS, dtype=lengths.dtype, device=lengths.device)[None, :]
        + row_starts[:, None]
    ).clamp_max(logits.shape[-1] - 1)
    logits.scatter_(dim=1, index=init_idxs, value=float("inf"))
    local_idxs = (
        lengths[:, None]
        - 1
        + row_starts[:, None]
        - torch.arange(LOCAL_TOKENS, dtype=lengths.dtype, device=lengths.device)[
            None, :
        ]
    ).clamp_min(0)
    logits.scatter_(dim=1, index=local_idxs, value=float("inf"))
    return logits


def _assert_topk_sets_match(got, ref, topk):
    """Top-k index sets after the engine's forced-include scatter agree exactly away
    from the selection boundary. Both sides accumulate exact FP8 products in FP32
    in different orders, so a reference value within the logits tolerance of the
    k-th reference value may legitimately land on either side of the cut; every
    index that differs between the two sets must lie in that boundary band, and
    both selections keep the same number of finite entries."""
    k = min(topk, ref.shape[1])
    ref_values, ref_indices = torch.topk(ref, k=k, dim=1)
    got_values, got_indices = torch.topk(got, k=k, dim=1)
    # Same forced includes and the same finite cells selected on both sides.
    assert torch.equal(
        (got_values != float("-inf")).sum(dim=1),
        (ref_values != float("-inf")).sum(dim=1),
    )
    ref_finite = ref.masked_fill(~torch.isfinite(ref), 0.0)
    row_max = ref_finite.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    kth = ref_values[:, k - 1 : k]
    kth = torch.where(torch.isfinite(kth), kth, torch.zeros_like(kth))
    band = 1e-2 * row_max + 1e-2 * kth.abs()
    in_ref = torch.zeros_like(ref, dtype=torch.bool).scatter_(1, ref_indices, True)
    in_got = torch.zeros_like(got, dtype=torch.bool).scatter_(1, got_indices, True)
    differs = (in_ref ^ in_got) & (ref != float("-inf"))
    off_band = differs & ((ref_finite - kth).abs() > band)
    assert not bool(off_band.any()), (
        f"{int(off_band.sum())} top-k indices differ outside the boundary band"
    )


# ---------------------------------------------------------------------------
# ragged prefill path
# ---------------------------------------------------------------------------


def _ragged_inputs(queries, keys, heads, device, seed=11):
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(queries, heads, HEAD_DIM, device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    kv = torch.randn(keys, HEAD_DIM, device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    kv_scales = torch.rand(keys, device=device, generator=generator) + 0.5
    weights = torch.rand(queries, heads, device=device, generator=generator) + 0.1
    # Causal-style windows: query i sees [0, keys - queries + i + 1); the first
    # window starts past zero to exercise ks > 0.
    ke = (keys - queries + torch.arange(queries, device=device) + 1).to(torch.int32)
    ks = torch.zeros(queries, device=device, dtype=torch.int32)
    ks[0] = min(8, int(ke[0]))
    return q, kv, kv_scales, weights, ks, ke


RAGGED_SHAPES = [
    # (queries, keys, heads): the 32-head K % 256 rows are the shipped routes
    # (plumbing validation); 64-head and K-tail rows admit as they are exported.
    (16, 4096, 32),
    (3, 2048, 32),
    (128, 8192, 32),
    (37, 4100, 64),
    (128, 8192, 64),
    (1024, 1024 + 4100, 64),
]


@pytest.mark.parametrize("queries,keys,heads", RAGGED_SHAPES)
def test_ragged_fp8_mqa_logits_matches_deep_gemm(queries, keys, heads):
    _skip_unless_device(cake.FI_DENSE_MQA_MODULE, cake.FI_DENSE_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    device = torch.device("cuda")
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs(queries, keys, heads, device)
    if not cake.supports_fp8_mqa_logits(q, kv, kv_scales, weights, ks, ke):
        pytest.skip(
            f"installed FlashInfer catalog does not serve H={heads}, Q={queries}, K={keys}"
        )
    num_sms = deep_gemm.get_num_sms()
    ref = deep_gemm.fp8_mqa_logits(
        q, (kv, kv_scales), weights, ks, ke, clean_logits=False
    )
    got = cake_fp8_mqa_logits(
        q, (kv, kv_scales), weights, ks, ke, clean_logits=False, sm_count=num_sms
    )
    torch.cuda.synchronize()
    assert tuple(got.shape) == (queries, keys) == tuple(ref.shape)
    assert got.dtype == torch.float32 and got.stride(1) == 1 and got.stride(0) % 4 == 0
    # Compare inside the windows: clean_logits=False leaves the rest unspecified
    # on both sides (the shipped 32-head programs write -inf there, the 64-head
    # programs store raw tiles like DeepGEMM).
    position = torch.arange(keys, device=device)[None, :]
    inside = (position >= ks[:, None]) & (position < ke[:, None])
    _assert_logits_close(
        got.masked_fill(~inside, float("-inf")), ref.masked_fill(~inside, float("-inf"))
    )
    # Engine tail on both: forced-include scatter, then top-k set equality.
    lengths = ke - ks
    topk = min(2048, keys)
    ref_masked = _mask_init_and_local_tokens(
        ref.masked_fill(~inside, float("-inf")).clone(), lengths, ks
    )
    got_masked = _mask_init_and_local_tokens(
        got.masked_fill(~inside, float("-inf")).clone(), lengths, ks
    )
    _assert_topk_sets_match(got_masked, ref_masked, topk)


def test_ragged_admission_rules():
    _skip_unless_device(cake.FI_DENSE_MQA_MODULE, cake.FI_DENSE_MQA_BACKEND_MODULE)
    device = torch.device("cuda")
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs(16, 4096, 32, device)
    assert cake.supports_fp8_mqa_logits(q, kv, kv_scales, weights, ks, ke)
    # Head counts outside {32, 64}, wrong dtypes and shape mismatches never admit.
    assert not cake.supports_fp8_mqa_logits(
        q[:, :16], kv, kv_scales, weights[:, :16], ks, ke
    )
    assert not cake.supports_fp8_mqa_logits(
        q.to(torch.bfloat16), kv, kv_scales, weights, ks, ke
    )
    assert not cake.supports_fp8_mqa_logits(
        q, kv, kv_scales.to(torch.bfloat16), weights, ks, ke
    )
    assert not cake.supports_fp8_mqa_logits(q, kv, kv_scales, weights, ks[:8], ke)
    assert not cake.supports_fp8_mqa_logits(
        q, kv, kv_scales, weights.to(torch.bfloat16), ks, ke
    )
    # The catalog decides K / Q coverage; a never-served K (not a multiple of 8)
    # is rejected by the route table, not by a kernel error.
    kv_tail, scales_tail = kv[:4091].contiguous(), kv_scales[:4091].contiguous()
    admitted_tail = cake.supports_fp8_mqa_logits(
        q, kv_tail, scales_tail, weights, ks, ke.clamp_max(4091)
    )
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as runtime

    arch = runtime.device_arch(device)
    assert admitted_tail == bool(runtime.dense_route_available(32, 16, 4091, arch=arch))
    # The dense verdict is per architecture (policy.dense_admission): the adapter's answer for the
    # 64-head family equals the runtime's for this device's arch, including a withheld tier (sm_100a
    # fallback) where the arch-less query would refuse to decide.
    q64, kv64, sc64, w64, ks64, ke64 = _ragged_inputs(16, 4096, 64, device)
    assert cake.supports_fp8_mqa_logits(q64, kv64, sc64, w64, ks64, ke64) == bool(
        runtime.dense_route_available(64, 16, 4096, arch=arch)
    )
    withheld = runtime.h64_admission(arch)["withheld_routes"]
    if withheld:
        # A tier withheld on this device's architecture never reaches the Cake path here.
        assert all(not r.endswith(":h32") for r in withheld)


# ---------------------------------------------------------------------------
# paged decode / verify path
# ---------------------------------------------------------------------------


def _paged_inputs(batch, next_n, heads, avg_ctx, device, seed=5):
    generator = torch.Generator(device=device).manual_seed(seed)
    block_kv = cake.DSA_MQA_PAGE
    lo, hi = max(block_kv, int(0.7 * avg_ctx)), max(block_kv + 1, int(1.3 * avg_ctx))
    ctx_max = torch.randint(
        lo, hi, (batch,), device=device, dtype=torch.int32, generator=generator
    )
    # Per-token lengths: the last token carries the request's length, earlier
    # speculative tokens their own shorter length (the engine's verify layout).
    ctx_2d = (
        ctx_max[:, None]
        - (next_n - 1 - torch.arange(next_n, device=device, dtype=torch.int32))[None, :]
    )
    ctx_2d = ctx_2d.clamp_min(1).contiguous()
    max_context_len = int(ctx_max.max())
    blocks_per_req = (ctx_max + block_kv - 1) // block_kv
    table_width = int(blocks_per_req.max())
    pages = int(blocks_per_req.sum()) + batch
    perm = torch.randperm(pages, device=device, generator=generator).to(torch.int32)
    block_table = torch.zeros(batch, table_width, device=device, dtype=torch.int32)
    offset = 0
    for b in range(batch):
        n = int(blocks_per_req[b])
        block_table[b, :n] = perm[offset : offset + n]
        offset += n
    q = torch.randn(
        batch, next_n, heads, HEAD_DIM, device=device, generator=generator
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(pages, block_kv, HEAD_DIM, device=device, generator=generator)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    kv_scale = (amax / 448.0).squeeze(-1)
    kv_fp8 = (kv / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    fused = torch.empty(
        pages, block_kv * (HEAD_DIM + 4), device=device, dtype=torch.uint8
    )
    fused[:, : block_kv * HEAD_DIM] = kv_fp8.reshape(pages, -1).view(torch.uint8)
    fused[:, block_kv * HEAD_DIM :] = kv_scale.reshape(pages, block_kv).view(
        torch.uint8
    )
    kv_cache = fused.view(pages, block_kv, 1, HEAD_DIM + 4)
    weights = (
        torch.rand(batch * next_n, heads, device=device, generator=generator) + 0.1
    )
    return q, kv_cache, weights, ctx_2d, block_table, max_context_len


PAGED_SHAPES = [
    # (batch, next_n, heads, avg_ctx); page 64 throughout.
    (4, 1, 64, 1024),
    (16, 2, 64, 4096),
    (3, 4, 64, 2048),
    (8, 1, 32, 1024),
    (200, 1, 64, 1024),  # above the SM count: the engine would chunk this
]


@pytest.mark.parametrize("batch,next_n,heads,avg_ctx", PAGED_SHAPES)
def test_paged_fp8_mqa_logits_matches_deep_gemm(batch, next_n, heads, avg_ctx):
    _skip_unless_device(cake.FI_PAGED_MQA_MODULE, cake.FI_PAGED_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    device = torch.device("cuda")
    q, kv_cache, weights, ctx_2d, block_table, max_len = _paged_inputs(
        batch, next_n, heads, avg_ctx, device
    )
    if not cake.supports_fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table
    ):
        pytest.skip(
            f"installed FlashInfer catalog does not serve H={heads}, page 64, next_n={next_n}"
        )
    num_sms = deep_gemm.get_num_sms()
    # Oracle: DeepGEMM exactly as the engine calls it (chunked above num_sms).
    ref_chunks = []
    for start in range(0, batch, num_sms):
        end = min(start + num_sms, batch)
        meta = deep_gemm.get_paged_mqa_logits_metadata(
            ctx_2d[start:end].contiguous(), 64, num_sms
        )
        ref_chunks.append(
            deep_gemm.fp8_paged_mqa_logits(
                q[start:end],
                kv_cache,
                weights[start * next_n : end * next_n],
                ctx_2d[start:end].contiguous(),
                block_table[start:end],
                meta,
                max_len,
                clean_logits=False,
            )
        )
    ref = torch.cat(ref_chunks, dim=0)
    cake_meta = cake_get_paged_mqa_logits_metadata(ctx_2d, 64, num_sms)
    assert cake_meta.dtype == torch.int32 and tuple(cake_meta.shape) == (num_sms + 1, 2)
    got = cake_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        ctx_2d,
        block_table,
        cake_meta,
        max_len,
        clean_logits=False,
    )
    torch.cuda.synchronize()
    assert tuple(got.shape) == (batch * next_n, max_len) == tuple(ref.shape)
    assert got.dtype == torch.float32 and got.stride(1) == 1 and got.stride(0) % 4 == 0
    # clean_logits=False: compare the cells inside every row's own length; the
    # engine masks the rest through topk_transform.
    lengths = ctx_2d.reshape(-1)
    position = torch.arange(max_len, device=device)[None, :]
    inside = position < lengths[:, None]
    _assert_logits_close(
        got.masked_fill(~inside, float("-inf")), ref.masked_fill(~inside, float("-inf"))
    )
    topk = min(2048, max_len)
    ref_masked = _mask_init_and_local_tokens(
        ref.masked_fill(~inside, float("-inf")).clone(), lengths
    )
    got_masked = _mask_init_and_local_tokens(
        got.masked_fill(~inside, float("-inf")).clone(), lengths
    )
    _assert_topk_sets_match(got_masked, ref_masked, topk)


def test_paged_admission_rules():
    _skip_unless_device(cake.FI_PAGED_MQA_MODULE, cake.FI_PAGED_MQA_BACKEND_MODULE)
    device = torch.device("cuda")
    q, kv_cache, weights, ctx_2d, block_table, _ = _paged_inputs(4, 2, 64, 1024, device)
    from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa as runtime

    assert cake.supports_fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table
    ) == bool(runtime.paged_route_available(64, 64, 2))
    # 1-D context lengths, page 128 caches, 16 heads and BF16 weights never admit.
    assert not cake.supports_fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d[:, -1].contiguous(), block_table
    )
    even_pages = kv_cache.shape[0] // 2 * 2
    page128 = kv_cache[:even_pages].reshape(-1).view(-1, 128, 1, 132)
    assert not cake.supports_fp8_paged_mqa_logits(
        q, page128, weights, ctx_2d, block_table
    )
    assert not cake.supports_fp8_paged_mqa_logits(
        q[:, :, :16], kv_cache, weights[:, :16], ctx_2d, block_table
    )
    assert not cake.supports_fp8_paged_mqa_logits(
        q, kv_cache, weights.to(torch.bfloat16), ctx_2d, block_table
    )
    # Per-architecture admission rules of the shipped catalog (policy.paged.admission, keyed by the call's
    # batch and max_context_len): the adapter's verdict equals the runtime's for the exact call, and the
    # block table's capacity stands in for a missing max_context_len.
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa

    arch = dense_mqa.device_arch(device)
    wide = torch.zeros(4, (1 << 20) // 64, device=device, dtype=torch.int32)
    wide[:, : block_table.shape[1]] = block_table
    for ctx in (1024, 8192, 32768, 32769, 131072, 1 << 20):
        assert cake.supports_fp8_paged_mqa_logits(
            q, kv_cache, weights, ctx_2d, wide, ctx
        ) == bool(runtime.paged_route_available(64, 64, 2, arch=arch, batch=4, max_context_len=ctx)), ctx
    assert cake.supports_fp8_paged_mqa_logits(q, kv_cache, weights, ctx_2d, wide) == bool(
        runtime.paged_route_available(64, 64, 2, arch=arch, batch=4, max_context_len=1 << 20)
    )
    rules = runtime.paged_admission_rules(arch, runtime.paged_route_name(64, 64, 2))
    if rules is not None and rules[0][0] is not None:
        # One request above the first rule's batch ceiling at a context only that rule admits falls back.
        big = rules[0][0] + 1
        q_b, kv_b, w_b, ctx_b, bt_b, _ = _paged_inputs(big, 2, 64, 1024, device)
        wide_b = torch.zeros(big, (1 << 20) // 64, device=device, dtype=torch.int32)
        wide_b[:, : bt_b.shape[1]] = bt_b
        assert not cake.supports_fp8_paged_mqa_logits(q_b, kv_b, w_b, ctx_b, wide_b, 1 << 20)


@pytest.mark.parametrize(
    "next_n,batch,avg_ctx",
    [(2, 2, 4096), (2, 2, 40000), (4, 2, 4096), (4, 2, 12000), (2, 17, 1024), (2, 65, 40000), (1, 129, 40000)],
)
def test_paged_route_honours_the_arch_admission_rules(next_n, batch, avg_ctx):
    """The engine route helper takes the Cake path where the catalog admits the
    (arch, route, batch, max_context_len) and falls back (``None``) where the
    shipped policy withholds it (sm_100a batch > 16; sm_103a large-batch
    long-context n1/n2); the taken path matches DeepGEMM inside every row's
    length."""
    _skip_unless_device(cake.FI_PAGED_MQA_MODULE, cake.FI_PAGED_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa
    from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa as runtime
    from sglang.srt.layers.attention.dsa import cake_indexer_routes

    device = torch.device("cuda")
    if not runtime.paged_route_available(64, 64, next_n):
        pytest.skip(f"installed FlashInfer catalog does not serve H=64, page 64, next_n={next_n}")
    q, kv_cache, weights, ctx_2d, block_table, max_len = _paged_inputs(
        batch, next_n, 64, avg_ctx, device
    )
    admitted = runtime.paged_route_admitted(
        dense_mqa.device_arch(device), runtime.paged_route_name(64, 64, next_n), batch, max_len
    )
    num_sms = deep_gemm.get_num_sms()
    with _route_on():
        got = cake_indexer_routes.cake_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            ctx_2d,
            block_table,
            max_len,
            block_kv=cake.DSA_MQA_PAGE,
            num_sms=num_sms,
        )
        torch.cuda.synchronize()
    cake_indexer_routes.reset_cake_route_state_for_tests()
    if not admitted:
        assert got is None, f"route taken outside the admission rules (batch {batch}, max_context_len {max_len})"
        with pytest.raises(ValueError, match="is not admitted on"):
            cake_fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx_2d,
                block_table,
                cake_get_paged_mqa_logits_metadata(ctx_2d, 64, num_sms),
                max_len,
                clean_logits=False,
            )
        return
    assert got is not None, "route fell back on an admitted (arch, route, context)"
    meta = deep_gemm.get_paged_mqa_logits_metadata(ctx_2d, 64, num_sms)
    ref = deep_gemm.fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table, meta, max_len, clean_logits=False
    )
    inside = torch.arange(max_len, device=device)[None, :] < ctx_2d.reshape(-1)[:, None]
    _assert_logits_close(
        got.masked_fill(~inside, float("-inf")), ref.masked_fill(~inside, float("-inf"))
    )


# ---------------------------------------------------------------------------
# Indexer helper methods (route switch on the real tensors)
# ---------------------------------------------------------------------------


def _indexer_stub():
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
    from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
        DSAPagedMQALogitsBackend,
    )

    indexer = Indexer.__new__(Indexer)
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.DEEPGEMM
    return indexer


def test_indexer_ragged_helper_switches_on_route():
    _skip_unless_device(cake.FI_DENSE_MQA_MODULE, cake.FI_DENSE_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    from sglang.kernels.cake_kernels import _routes
    from sglang.srt.layers.attention.dsa import cake_indexer_routes

    device = torch.device("cuda")
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs(16, 4096, 32, device)
    if not cake.supports_fp8_mqa_logits(q, kv, kv_scales, weights, ks, ke):
        pytest.skip(
            "installed FlashInfer catalog does not serve the 32-head K=4096 row"
        )
    indexer = _indexer_stub()
    stock = deep_gemm.fp8_mqa_logits(
        q, (kv, kv_scales), weights, ks, ke, clean_logits=False
    )
    position = torch.arange(4096, device=device)[None, :]
    inside = (position >= ks[:, None]) & (position < ke[:, None])
    for route_on in (False, True):
        env = {_routes.ENV_VAR: "dsa_indexer"} if route_on else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if not route_on:
                os.environ.pop(_routes.ENV_VAR, None)
            _routes.reset_cache_for_tests()
            cake_indexer_routes.reset_cake_route_state_for_tests()
            with mock.patch.object(
                cake_indexer_routes,
                "_cake_ragged_kernels",
                wraps=cake_indexer_routes._cake_ragged_kernels,
            ) as kernels:
                out = indexer._fp8_mqa_logits_cuda(q, (kv, kv_scales), weights, ks, ke)
            torch.cuda.synchronize()
            assert kernels.call_count == (1 if route_on else 0)
            _assert_logits_close(
                out.masked_fill(~inside, float("-inf")),
                stock.masked_fill(~inside, float("-inf")),
            )
    _routes.reset_cache_for_tests()
    cake_indexer_routes.reset_cake_route_state_for_tests()


def _route_on():
    from sglang.kernels.cake_kernels import _routes
    from sglang.srt.layers.attention.dsa import cake_indexer_routes

    _routes.reset_cache_for_tests()
    cake_indexer_routes.reset_cake_route_state_for_tests()
    return mock.patch.dict(os.environ, {_routes.ENV_VAR: "dsa_indexer"}, clear=False)


def _assert_graph_replay(launch, poison, compare):
    """``launch()`` eagerly (warm-up: JIT build + capture-gate registration), then
    captured in a CUDA graph on a side stream; the captured output is poisoned
    before every replay so an empty capture (kernels launched outside the
    capture stream) fails loudly. ``compare(got, ref)`` must raise on mismatch."""
    eager = launch()
    torch.cuda.synchronize()
    assert eager is not None, "route not taken eagerly"
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = launch()
    assert captured is not None, "route fell back inside CUDA-graph capture"
    with torch.cuda.stream(stream):
        poison(captured)
        graph.replay()
    stream.synchronize()
    compare(captured, eager)
    finite = torch.isfinite(captured)
    assert bool(finite.any()) and bool((captured[finite] != 0).any())
    return graph, captured, eager


def test_ragged_route_graph_replay_matches_eager():
    """sglang decode/verify runs the indexer under CUDA-graph capture: the Cake
    ragged entry must launch on the capture stream (replay == eager bits)."""
    _skip_unless_device(cake.FI_DENSE_MQA_MODULE, cake.FI_DENSE_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    from sglang.srt.layers.attention.dsa import cake_indexer_routes

    device = torch.device("cuda")
    q, kv, kv_scales, weights, ks, ke = _ragged_inputs(16, 4096, 32, device)
    if not cake.supports_fp8_mqa_logits(q, kv, kv_scales, weights, ks, ke):
        pytest.skip(
            "installed FlashInfer catalog does not serve the 32-head K=4096 row"
        )
    num_sms = deep_gemm.get_num_sms()
    position = torch.arange(4096, device=device)[None, :]
    inside = (position >= ks[:, None]) & (position < ke[:, None])

    def compare(got, ref):
        assert torch.equal(got[inside], ref[inside]), "graph replay != eager logits"

    with _route_on():
        graph, captured, eager = _assert_graph_replay(
            lambda: cake_indexer_routes.cake_fp8_mqa_logits(
                q, (kv, kv_scales), weights, ks, ke, num_sms=num_sms
            ),
            lambda out: out.fill_(float("nan")),
            compare,
        )
        stock = deep_gemm.fp8_mqa_logits(
            q, (kv, kv_scales), weights, ks, ke, clean_logits=False
        )
        torch.cuda.synchronize()
        _assert_logits_close(
            captured.masked_fill(~inside, float("-inf")),
            stock.masked_fill(~inside, float("-inf")),
        )
        # Changed operands reach the replayed kernels.
        weights.neg_()
        captured.fill_(float("nan"))
        graph.replay()
        changed = cake_indexer_routes.cake_fp8_mqa_logits(
            q, (kv, kv_scales), weights, ks, ke, num_sms=num_sms
        )
        torch.cuda.synchronize()
        assert torch.equal(captured[inside], changed[inside])
        assert not torch.equal(captured[inside], eager[inside])
    cake_indexer_routes.reset_cake_route_state_for_tests()


def test_paged_route_graph_replay_matches_eager():
    """One paged decode call (metadata + logits) captured in a CUDA graph through
    the engine route helper replays the eager bits inside every row's length."""
    _skip_unless_device(cake.FI_PAGED_MQA_MODULE, cake.FI_PAGED_MQA_BACKEND_MODULE)
    deep_gemm = _deep_gemm()
    from sglang.srt.layers.attention.dsa import cake_indexer_routes

    device = torch.device("cuda")
    q, kv_cache, weights, ctx_2d, block_table, max_len = _paged_inputs(
        4, 1, 64, 1024, device
    )
    if not cake.supports_fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_2d, block_table
    ):
        pytest.skip(
            "installed FlashInfer catalog does not serve H=64, page 64, next_n=1"
        )
    num_sms = deep_gemm.get_num_sms()
    inside = torch.arange(max_len, device=device)[None, :] < ctx_2d.reshape(-1)[:, None]

    def compare(got, ref):
        assert torch.equal(got[inside], ref[inside]), "graph replay != eager logits"

    with _route_on():
        graph, captured, eager = _assert_graph_replay(
            lambda: cake_indexer_routes.cake_fp8_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                ctx_2d,
                block_table,
                max_len,
                block_kv=cake.DSA_MQA_PAGE,
                num_sms=num_sms,
            ),
            lambda out: out.fill_(float("nan")),
            compare,
        )
        weights.neg_()
        captured.fill_(float("nan"))
        graph.replay()
        changed = cake_indexer_routes.cake_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            ctx_2d,
            block_table,
            max_len,
            block_kv=cake.DSA_MQA_PAGE,
            num_sms=num_sms,
        )
        torch.cuda.synchronize()
        assert torch.equal(captured[inside], changed[inside])
        assert not torch.equal(captured[inside], eager[inside])
    cake_indexer_routes.reset_cake_route_state_for_tests()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
