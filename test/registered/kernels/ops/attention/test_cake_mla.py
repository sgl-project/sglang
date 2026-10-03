"""Cake MLA decode kernels (Kimi-K3 FP8, varq DCP, NVFP4 DSv4, DSv4 sparse,
concat_mla_k) through sglang.kernels.

Registry resolution for every ``attention_mla`` adapter; bitwise parity of the
facade against direct FlashInfer calls (var-Q DCP decode: bitwise when the host
plan cannot split a request, FlashInfer's documented replay spread otherwise,
each result checked against the reference first); FP32 torch references within the
precision's tolerance (BF16 1e-2, FP8 0.1, NVFP4 block-scaled 0.1 as measured
by FlashInfer's own tests). GPU tests skip when FlashInfer lacks the module,
when no generated program is registered for the cell, or when the device is
outside sm_100a / sm_103a (SM120 NVFP4 DSv4 entries are parity-tested only
when an SM120/121 device is present).
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_mla as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_concat_mla_k,
    cake_kimi_k3_mla_fp8_paged_attention,
    cake_mla_varq_dcp_decode,
    cake_prepare_kimi_k3_mla_fp8_paged_attention,
    cake_prepare_nvfp4_batch_decode_with_kv_cache_mla,
    cake_trtllm_batch_decode_sparse_mla_dsv4,
    cake_trtllm_batch_decode_with_kv_cache_mla,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "trtllm_batch_decode_sparse_mla_dsv4",
    "create_sparse_mla_sm120_wrapper",
    "sparse_mla_sm120_dsv4_nvfp4_decode",
    "sparse_mla_sm120_dsv4_nvfp4_prefill",
    "trtllm_batch_decode_with_kv_cache_mla",
    "kimi_k3_mla_fp8_paged_attention",
    "prepare_kimi_k3_mla_fp8_paged_attention",
    "prepare_nvfp4_batch_decode_with_kv_cache_mla",
    "mla_varq_dcp_decode",
    "prepare_mla_varq_dcp_decode",
    "concat_mla_k",
)
LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
PAGE = 64


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_mla:")
    assert spec.load() is not None


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake kernel is built for {archs}, device is {cc}")


def _ceil_div(a, b):
    return -(-a // b)


def _fp8(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


# --------------------------------------------------------------------------
# Kimi-K3 FP8 paged MLA (public backend="cake" route B + direct entries)
# --------------------------------------------------------------------------


def _kimi_case(batch, q_lens, kv_lens, num_heads, *, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per = [_ceil_div(k, PAGE) for k in kv_lens]
    total_pages = sum(pages_per)
    pool_pages = total_pages + 8
    kv_cache = _fp8(
        torch.randn((pool_pages, PAGE, QK_DIM), generator=gen, device=device) * 0.5
    )
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages]
    block_tables = torch.zeros(
        (batch, max(pages_per)), dtype=torch.int32, device=device
    )
    off = 0
    for b, n in enumerate(pages_per):
        block_tables[b, :n] = perm[off : off + n].to(torch.int32)
        off += n
    query = _fp8(
        torch.randn((sum(q_lens), num_heads, QK_DIM), generator=gen, device=device)
        * 0.5
    )
    q_indptr = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        num_heads=num_heads,
        scale=1.0 / math.sqrt(QK_DIM),
    )


def _kimi_reference(case):
    cache, num_heads = case["kv_cache"], case["num_heads"]
    q_rows = case["query"].float()
    out = torch.zeros((q_rows.shape[0], num_heads, LATENT), device=q_rows.device)
    q_indptr = case["q_indptr"].tolist()
    for b, (q_len, kv_len) in enumerate(zip(case["q_lens"], case["kv_lens"])):
        pages = case["block_tables"][b, : _ceil_div(kv_len, PAGE)].long()
        values = cache[pages].reshape(-1, QK_DIM)[:kv_len].float()
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len * num_heads, QK_DIM)
        logits = (q @ values.T) * case["scale"]
        if q_len > 1:
            pos = torch.arange(kv_len, device=q.device)
            limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
            mask = pos[None, :] < limit[:, None]
            logits = (
                logits.reshape(q_len, num_heads, kv_len)
                .masked_fill(~mask[:, None, :], float("-inf"))
                .reshape(q_len * num_heads, kv_len)
            )
        probs = torch.softmax(logits, dim=-1)
        out[q_indptr[b] : q_indptr[b + 1]] = (probs @ values[:, :LATENT]).reshape(
            q_len, num_heads, LATENT
        )
    return out


def test_kimi_k3_mla_public_route_and_direct_entries():
    _skip_unless(cake.ARCHS, cake.FI_KIMI_K3_MLA_MODULE, cake.FI_KIMI_K3_MLA_JIT_MODULE)
    device = torch.device("cuda")
    case = _kimi_case(2, (1, 1), (200, 64), 12, seed=11, device=device)
    query = case["query"].reshape(2, 1, case["num_heads"], QK_DIM)
    assert cake.supports_trtllm_batch_decode_with_kv_cache_mla(
        query,
        case["kv_cache"],
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
    )
    assert cake.supports_kimi_k3_mla_fp8_paged_attention(query, case["kv_cache"])
    workspace = torch.zeros(
        cake.kimi_k3_mla_workspace_bytes(2 * case["num_heads"], 256),
        dtype=torch.uint8,
        device=device,
    )
    max_seq_len = int(case["block_tables"].shape[1]) * PAGE
    out = torch.full(
        (2, 1, case["num_heads"], LATENT),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    kwargs = dict(
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=max_seq_len,
        bmm1_scale=case["scale"],
        bmm2_scale=1.0,
    )
    returned = cake_trtllm_batch_decode_with_kv_cache_mla(
        query, case["kv_cache"], workspace, out=out, **kwargs
    )
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    out_fi = trtllm_batch_decode_with_kv_cache_mla(
        query, case["kv_cache"], workspace, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert returned is out
    assert torch.equal(out, out_fi)
    ref = _kimi_reference(case).reshape(out.shape)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)

    # Direct one-shot and prepared entries reproduce the public route.
    out_direct = torch.empty_like(out)
    cake_kimi_k3_mla_fp8_paged_attention(
        query,
        case["kv_cache"],
        case["block_tables"],
        case["seq_lens"],
        out_direct,
        workspace,
        bmm1_scale=case["scale"],
        max_seq_len=max_seq_len,
    )
    out_prepared = torch.empty_like(out)
    prepared = cake_prepare_kimi_k3_mla_fp8_paged_attention(
        query=query,
        kv_cache=case["kv_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        out=out_prepared,
        workspace_buffer=workspace,
        bmm1_scale=case["scale"],
        max_seq_len=max_seq_len,
    )
    prepared.launch()
    torch.cuda.synchronize()
    assert torch.equal(out_direct, out)
    assert torch.equal(out_prepared, out)
    # Prepared launch follows new query contents without re-planning.
    query.copy_(_fp8(torch.randn(query.shape, device=device) * 0.5))
    prepared.launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out_prepared.float(),
        _kimi_reference(case).reshape(out.shape),
        atol=1e-2,
        rtol=1e-2,
    )


def _varq_reference(query, cum_q, kv_rows, kv_lens, scale):
    """FP32 var-Q reference (same as ``test_cake_mla_varq._varq_reference``)."""
    total_q, num_heads, _ = query.shape
    out = torch.zeros((total_q, num_heads, LATENT), device=query.device)
    lse = torch.full((total_q, num_heads), -math.inf, device=query.device)
    offsets = cum_q.tolist()
    for b, (q0, q1) in enumerate(zip(offsets[:-1], offsets[1:])):
        q_len, g = q1 - q0, kv_lens[b]
        keys = kv_rows[b][:g].float()
        bounds = g - q_len + torch.arange(q_len, device=query.device)
        visible = torch.arange(g, device=query.device)[None, :] <= bounds[:, None]
        scores = torch.einsum("qhd,kd->qhk", query[q0:q1].float(), keys) * scale
        scores = scores.masked_fill(~visible[:, None, :], -math.inf)
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - row_lse[..., None])
        out[q0:q1] = torch.einsum("qhk,kd->qhd", probs, keys[:, :LATENT])
        lse[q0:q1] = row_lse
    return out, lse


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "FlashInfer 46340689a5ab: the first var-Q DCP decode launch after a "
        "Kimi-K3 FP8 MLA (cake) launch in the same process returns wrong rows "
        "(Linear CAKE-922); the second launch is correct. Strict: flips to a "
        "failure once FlashInfer fixes it so this guard can be removed."
    ),
)
def test_varq_after_kimi_k3_mla(request):
    """Cross-kernel interference guard (see ``test_cake_mla_varq.py``).

    Runs the Kimi-K3 cake MLA public route once, then the var-Q one-shot with
    the same inputs as the standalone var-Q test and checks it against the FP32
    reference.  Deterministic on GB300 and B200; not a stream race (survives
    synchronize + sleep, a fresh stream and ``CUDA_LAUNCH_BLOCKING=1``).
    """
    test_kimi_k3_mla_public_route_and_direct_entries()
    _skip_unless(cake.ARCHS, cake.FI_VARQ_DCP_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(12)
    kv_lens, q_lens, num_heads = [300, 70], [2, 1], 12
    total_q = sum(q_lens)
    query = (
        torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device) * 0.1
    ).to(torch.bfloat16)
    pages_per = [_ceil_div(k, PAGE) for k in kv_lens]
    kv_cache = (
        torch.randn((sum(pages_per), PAGE, QK_DIM), generator=gen, device=device) * 0.1
    ).to(torch.bfloat16)
    page_table = torch.zeros((2, max(pages_per)), dtype=torch.int32, device=device)
    kv_rows, off = [], 0
    for b, n in enumerate(pages_per):
        page_table[b, :n] = torch.arange(off, off + n, dtype=torch.int32, device=device)
        kv_rows.append(kv_cache[off : off + n].reshape(-1, QK_DIM))
        off += n
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    cum_q = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(LATENT)
    ref_out, ref_lse = _varq_reference(query, cum_q, kv_rows, kv_lens, scale)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        cake.max_mla_varq_dcp_decode_workspace_size(
            batch_size=2, max_q_len=2, num_heads=num_heads, num_sms=num_sms
        ),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.full(
        (total_q, num_heads, LATENT), math.nan, dtype=torch.bfloat16, device=device
    )
    lse = torch.full((total_q, num_heads), math.nan, dtype=torch.float32, device=device)
    cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        workspace,
        page_table,
        seq_lens,
        max(kv_lens),
        scale,
        cum_seq_lens_q=cum_q,
        max_q_len=2,
        out=out,
        lse=lse,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# NVFP4 DeepSeek-V4 MLA decode (prepared)
# --------------------------------------------------------------------------

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_nvfp4(packed, scale):
    lo, hi = packed & 0x0F, packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    values = table[(codes & 0x7).long()] * torch.where((codes & 0x8) != 0, -1.0, 1.0)
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 16, 16)
    scale = scale.view(torch.float8_e4m3fn).float()
    return (blocks * scale.unsqueeze(-1)).reshape(*values.shape)


def test_nvfp4_mla_decode_prepare_launch_matches_reference():
    _skip_unless(cake.ARCHS, cake.FI_NVFP4_MLA_MODULE)
    from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
        generated_program_available,
    )

    device = torch.device("cuda")
    if not generated_program_available(device):
        pytest.skip("no generated NVFP4 MLA decode program registered")
    gen = torch.Generator(device=device).manual_seed(13)
    kv_lens, num_heads, q_len = [256, 300], 64, 6
    batch = len(kv_lens)
    pages_per = [_ceil_div(k, PAGE) for k in kv_lens]
    total_pages = sum(pages_per)
    k_full = torch.randn((total_pages, PAGE, LATENT), generator=gen, device=device)
    perm = torch.randperm(total_pages, generator=gen, device=device).to(torch.int32)
    block_tables = torch.zeros(
        (batch, max(pages_per)), dtype=torch.int32, device=device
    )
    off = 0
    for b, n in enumerate(pages_per):
        block_tables[b, :n] = perm[off : off + n]
        off += n
    q = torch.randn((batch * q_len, num_heads, LATENT), generator=gen, device=device)
    for b in range(batch):  # peak the softmax so NVFP4 rounding stays bounded
        for i in range(q_len):
            visible = kv_lens[b] - q_len + i + 1
            target = int(torch.randint(0, visible, (1,), generator=gen, device=device))
            page = int(block_tables[b, target // PAGE])
            q[b * q_len + i] += 0.2 * k_full[page, target % PAGE]
    query, query_scale = cake.quantize_nvfp4(q)
    kv_cache, kv_scale = cake.quantize_nvfp4(k_full)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    assert cake.supports_nvfp4_batch_decode_with_kv_cache_mla(
        query, query_scale, kv_cache, kv_scale
    )
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.empty(
        cake.nvfp4_mla_decode_workspace_size(
            kv_lens, num_heads, num_sms=num_sms, q_len=q_len
        ),
        dtype=torch.uint8,
        device=device,
    )
    scale = LATENT**-0.5
    out = torch.empty(
        (batch * q_len, num_heads, LATENT), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch * q_len, num_heads), float("nan"), device=device)
    runner = cake_prepare_nvfp4_batch_decode_with_kv_cache_mla(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace,
        sm_scale=scale,
        out=out,
        lse=lse,
        return_lse=True,
    )
    result = runner()
    assert result[0] is out and result[1] is lse
    torch.cuda.synchronize()

    from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla

    out_fi = torch.empty_like(out)
    lse_fi = torch.empty_like(lse)
    prepare_nvfp4_batch_decode_with_kv_cache_mla(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        torch.empty_like(workspace),
        sm_scale=scale,
        out=out_fi,
        lse=lse_fi,
        return_lse=True,
        backend="cake",
    )()
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(lse, lse_fi)

    q_all = _dequantize_nvfp4(query, query_scale)
    for b, kv_len in enumerate(kv_lens):
        pages = block_tables[b, : _ceil_div(kv_len, PAGE)].long()
        k = _dequantize_nvfp4(kv_cache[pages], kv_scale[pages]).reshape(-1, LATENT)
        k = k[:kv_len]
        qb = q_all[b * q_len : (b + 1) * q_len]
        logits = torch.einsum("rhd,nd->hrn", qb, k) * scale
        pos = torch.arange(kv_len, device=device)
        limit = kv_len - q_len + torch.arange(q_len, device=device) + 1
        logits = logits.masked_fill(~(pos[None, :] < limit[:, None])[None], -math.inf)
        ref_lse = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(logits - ref_lse[..., None])
        ref = torch.einsum("hrn,nd->rhd", probs, k)
        rows = slice(b * q_len, (b + 1) * q_len)
        torch.testing.assert_close(out[rows].float(), ref, atol=0.1, rtol=0.1)
        torch.testing.assert_close(
            lse[rows], ref_lse.transpose(0, 1), atol=0.05, rtol=0.05
        )


# --------------------------------------------------------------------------
# DeepSeek-V4 sparse MLA decode (SM100/103 Cake route)
# --------------------------------------------------------------------------


def test_dsv4_sparse_mla_decode_cake_matches_flashinfer_and_reference():
    """Combined-table DSv4 route: SWA rows (128) + compressed rows, dense query.

    Indices are storage row offsets into the natural-order pools so the
    position and storage conventions coincide.
    """
    _skip_unless(cake.ARCHS, cake.FI_DSV4_MODULE, cake.FI_DSV4_JIT_MODULE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(14)
    batch, num_heads, q_len = 3, 32, 1
    swa_tokens, compressed_tokens = 512, 1024
    compressed_topk = 256
    sparse_topk = cake.DSV4_SWA_TOPK + compressed_topk
    query = (
        torch.randn((batch, q_len, num_heads, LATENT), generator=gen, device=device)
        * 0.1
    ).to(torch.bfloat16)
    swa_kv_cache = (
        torch.randn((swa_tokens // PAGE, 1, PAGE, LATENT), generator=gen, device=device)
        * 0.1
    ).to(torch.bfloat16)
    compressed_kv_cache = (
        torch.randn(
            (compressed_tokens // PAGE, 1, PAGE, LATENT), generator=gen, device=device
        )
        * 0.1
    ).to(torch.bfloat16)
    rows = batch * q_len
    swa_idx = torch.stack(
        [
            torch.randperm(swa_tokens, generator=gen, device=device)[
                : cake.DSV4_SWA_TOPK
            ]
            .sort()
            .values
            for _ in range(rows)
        ]
    )
    comp_idx = torch.stack(
        [
            torch.randperm(compressed_tokens, generator=gen, device=device)[
                :compressed_topk
            ]
            .sort()
            .values
            for _ in range(rows)
        ]
    )
    sparse_indices = torch.cat((swa_idx, comp_idx), dim=1).to(torch.int32).contiguous()
    comp_lens = torch.tensor([compressed_topk, 200, 128], device=device)
    sparse_topk_lens = (comp_lens + cake.DSV4_SWA_TOPK).to(torch.int32)
    seq_lens = torch.full(
        (batch,), swa_tokens + compressed_tokens, dtype=torch.int32, device=device
    )
    assert cake.supports_trtllm_batch_decode_sparse_mla_dsv4(
        query, swa_kv_cache, compressed_kv_cache=compressed_kv_cache
    )
    workspace = torch.empty(
        cake.get_cake_dsv4_workspace_bytes(rows, num_heads, sparse_topk, query.dtype),
        dtype=torch.uint8,
        device=device,
    )
    cake.cake_dsv4_workspace_reset(workspace)
    scale = LATENT**-0.5
    out = torch.full(query.shape, float("nan"), dtype=torch.bfloat16, device=device)
    kwargs = dict(
        sparse_indices=sparse_indices,
        compressed_kv_cache=compressed_kv_cache,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        enable_pdl=False,
    )
    try:
        returned = cake_trtllm_batch_decode_sparse_mla_dsv4(
            query, swa_kv_cache, workspace, out=out, **kwargs
        )
    except NotImplementedError as error:  # no generated variant for this cell
        pytest.skip(str(error))
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    out_fi = trtllm_batch_decode_sparse_mla_dsv4(
        query, swa_kv_cache, workspace, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert returned is out
    assert torch.equal(out, out_fi)

    swa_rows = swa_kv_cache.reshape(-1, LATENT).float()
    comp_rows = compressed_kv_cache.reshape(-1, LATENT).float()
    for t in range(rows):
        n_comp = int(comp_lens[t])
        keys = torch.cat(
            (swa_rows[swa_idx[t].long()], comp_rows[comp_idx[t, :n_comp].long()])
        )
        qt = query.reshape(rows, num_heads, LATENT)[t].float()
        probs = torch.softmax((qt @ keys.T) * scale, dim=-1)
        ref = probs @ keys
        torch.testing.assert_close(
            out.reshape(rows, num_heads, LATENT)[t].float(), ref, atol=1e-2, rtol=1e-2
        )


# --------------------------------------------------------------------------
# SM120 NVFP4 DSv4 (parity only; no SM120 runner in CI)
# --------------------------------------------------------------------------


def test_sm120_dsv4_nvfp4_public_route_parity():
    _skip_unless(
        cake.SM120_ARCHS, cake.FI_SM120_DSV4_MODULE, cake.FI_SM120_DSV4_JIT_MODULE
    )
    from flashinfer.mla import (
        nvfp4_quantize_pack_sparse_mla_cache,
        trtllm_batch_decode_sparse_mla_dsv4,
    )

    device = torch.device("cuda")
    torch.manual_seed(15)
    num_tokens, num_heads, topk, page_size, num_pages = 4, 16, 256, 64, 8
    q = (torch.randn(num_tokens, num_heads, LATENT, device=device) / 10).clamp(-1, 1)
    q = q.to(torch.bfloat16)
    latent = (torch.randn(num_pages, page_size, LATENT, device=device) / 10).clamp(
        -1, 1
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(latent.to(torch.bfloat16))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device=device
    )
    lengths = torch.randint(
        topk // 2, topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )
    assert cake.supports_trtllm_batch_decode_sparse_mla_dsv4(
        q, cache, kv_cache_format="nvfp4"
    )
    workspace = torch.empty(
        cake.sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(num_tokens, num_heads, topk),
        dtype=torch.uint8,
        device=device,
    )
    kwargs = dict(
        sparse_indices=indices,
        swa_topk_lens=lengths,
        bmm1_scale=LATENT**-0.5,
        kv_cache_format="nvfp4",
    )
    out = cake_trtllm_batch_decode_sparse_mla_dsv4(q, cache, workspace, **kwargs)
    out_fi = trtllm_batch_decode_sparse_mla_dsv4(
        q, cache, workspace, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, out_fi)


# --------------------------------------------------------------------------
# concat_mla_k
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_tokens", [1, 33, 1024])
def test_concat_mla_k_bitwise(num_tokens, dtype):
    _skip_unless(cake.ARCHS, cake.FI_CONCAT_MODULE, cake.FI_CONCAT_JIT_MODULE)
    device = torch.device("cuda")
    torch.manual_seed(16)
    k_nope = torch.randn(num_tokens, 128, 128, device=device, dtype=torch.bfloat16).to(
        dtype
    )
    k_rope = torch.randn(num_tokens, 1, 64, device=device, dtype=torch.bfloat16).to(
        dtype
    )
    k = torch.empty(num_tokens, 128, 192, dtype=dtype, device=device)
    assert cake.supports_concat_mla_k(k, k_nope, k_rope)
    cake_concat_mla_k(k, k_nope, k_rope)
    ref = torch.empty_like(k)
    ref[..., :128] = k_nope
    ref[..., 128:] = k_rope
    torch.cuda.synchronize()
    assert torch.equal(k.view(torch.uint8), ref.view(torch.uint8))

    from flashinfer.concat_ops import concat_mla_k

    k_fi = torch.empty_like(k)
    concat_mla_k(k_fi, k_nope, k_rope, backend="cake")
    torch.cuda.synchronize()
    assert torch.equal(k.view(torch.uint8), k_fi.view(torch.uint8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
