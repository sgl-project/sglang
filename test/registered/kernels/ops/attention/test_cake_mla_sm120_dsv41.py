"""Cake SM120/121 DeepSeek-V4.1 mixed-cache sparse-MLA decode through sglang.kernels.

FlashInfer PR flashinfer-ai/flashinfer#5983 (commit ``2c1c05250``; read at main
``e4f94f948``): ``cake_sparse_mla_sm120_dsv41_mixed_decode`` over a 528 B/token
FP8 + UE8M0 group-32 main cache and an optional 288 B/token V41_FP4 extra cache,
plus the DSv4.1 FP8 main-cache writers ``dsv41_fp8_quantize_{pack,append}``.

GPU-free: the registry resolves every op id to the FlashInfer backend and the
``supports_*`` gates reject CPU tensors and contract deviations without
raising. GPU (SM120 / SM121 only; skips elsewhere, when the installed
FlashInfer lacks the modules, or when the generated family is absent from the
tree): the FP8 pack is bitwise equal to a direct FlashInfer call and to the
pure-torch DSv4.1 FOOTER quantizer (FlashInfer's own reference), its
dequantization matches the BF16 latent within FP8 tolerance
(``atol = rtol = 0.1``); the slot append reproduces the pack bitwise; the extra
cache (pre-baseline FP4 writer, reached directly) dequantizes within FP4
tolerance (``atol = 1.0, rtol = 0.1``); the facade decode is bitwise equal to
the direct FlashInfer call and to the wrapper route, and matches an fp32
reference over the exactly dequantized caches at BF16 tolerance (``1e-2``) on
the BF16 route and FP8 tolerance (``0.1``) on the FP8 route when exported.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_mla_sm120_dsv41 as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_create_sparse_mla_sm120_dsv41_mixed_wrapper,
    cake_dsv41_fp8_quantize_append_sparse_mla_cache,
    cake_dsv41_fp8_quantize_pack_sparse_mla_cache,
    cake_sparse_mla_sm120_dsv41_mixed_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OPS = (
    "sparse_mla_sm120_dsv41_mixed_decode",
    "create_sparse_mla_sm120_dsv41_mixed_wrapper",
    "dsv41_fp8_quantize_pack_sparse_mla_cache",
    "dsv41_fp8_quantize_append_sparse_mla_cache",
)
D = 512
SM_SCALE = D**-0.5
MAIN_BPT = 528
EXTRA_BPT = 288
TOL = {
    "bf16": dict(atol=1e-2, rtol=1e-2),
    "fp8": dict(atol=0.1, rtol=0.1),
    "fp4": dict(atol=1.0, rtol=0.1),
}


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.attention_mla_sm120_dsv41:"
    )
    assert spec.load() is not None


def test_supports_rejects_cpu_and_contract_deviations():
    q = torch.zeros(2, 16, D, dtype=torch.bfloat16)
    cache = torch.zeros(4, 1, 64, MAIN_BPT, dtype=torch.uint8)
    indices = torch.zeros(2, 128, dtype=torch.int32)
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(q, cache, indices)
    latent = torch.zeros(4, 64, D, dtype=torch.bfloat16)
    assert not cake.supports_dsv41_fp8_quantize_pack(latent)
    slots = torch.arange(4 * 64, dtype=torch.int64)
    assert not cake.supports_dsv41_fp8_quantize_append(latent.view(-1, D), slots, cache)
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    q, cache, indices, latent, slots = (
        t.to(dev) for t in (q, cache, indices, latent, slots)
    )
    # Deviations never raise and are rejected on any device.
    nvfp4_cache = torch.zeros(4, 1, 64, 384, dtype=torch.uint8, device=dev)
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q, nvfp4_cache, indices
    )
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q.float(), cache, indices
    )
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q, cache, indices.long()
    )
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q, cache, indices, extra_indices=indices
    )  # extra cache missing
    assert not cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q, cache, indices, compute_precision="nvfp4"
    )
    assert not cake.supports_dsv41_fp8_quantize_pack(latent, kv_layout="HDN")
    assert not cake.supports_dsv41_fp8_quantize_pack(latent.float())
    assert not cake.supports_dsv41_fp8_quantize_append(
        latent.view(-1, D), slots.view(4, 64), cache
    )


# --------------------------------------------------------------------------
# GPU gates
# --------------------------------------------------------------------------


def _skip_unless_sm12x(*modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in cake.ARCHS:
        pytest.skip(f"Cake DSv4.1 mixed-cache route is SM120/121 only, device is {cc}")


def _skip_unless_family():
    _skip_unless_sm12x(
        cake.FI_MODULE, cake.FI_JIT_MODULE, cake.FI_API_MODULE, cake.FI_API_JIT_MODULE
    )
    if not cake.kernels_available():
        pytest.skip("generated Cake SM120 DSv4.1 mixed-cache family not in this tree")


# --------------------------------------------------------------------------
# Pure-torch oracles (FlashInfer test references, vectorised)
# --------------------------------------------------------------------------


def _quantize_dsv41_fp8(latent):
    """BF16 ``[P, ps, 512]`` -> DSv4.1 FP8 FOOTER cache ``[P, ps, 1, 528]`` (NHD).

    Groups of 32 share one UE8M0 scale ``2**ceil(log2(max(amax / 448, 1e-4)))``
    with ``amax`` clamped at ``1e-4``; values divided by the scale, saturated to
    +-448, rounded to nearest-even E4M3; data rows then the scale rows per page.
    """
    P, ps, _ = latent.shape
    x = latent.float().view(P, ps, 16, 32)
    amax = x.abs().amax(dim=-1).clamp(min=1e-4)
    scale = torch.pow(2, torch.clamp_min(amax / 448.0, 1e-4).log2().ceil())
    fp8 = (x / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    ue8m0 = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    data = fp8.view(torch.uint8).reshape(P, ps * 512)
    scales = ue8m0.reshape(P, ps * 16)
    return torch.cat([data, scales], dim=1).view(P, ps, 1, MAIN_BPT)


def _dequantize_dsv41_fp8(cache_nhd):
    """DSv4.1 FP8 FOOTER cache ``[P, ps, 1, 528]`` -> BF16 rows ``[P * ps, 512]``."""
    P, ps, _, bpt = cache_nhd.shape
    assert bpt == MAIN_BPT
    p = cache_nhd.reshape(P, ps * bpt)
    data = p[:, : ps * 512].reshape(P, ps, 16, 32).view(torch.float8_e4m3fn).float()
    scale = torch.pow(2.0, p[:, ps * 512 :].reshape(P, ps, 16).float() - 127.0)
    return (data * scale.unsqueeze(-1)).to(torch.bfloat16).reshape(P * ps, D)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_dsv41_fp4(cache_nhd):
    """V41_FP4 cache ``[P, ps, 1, 288]`` -> BF16 rows ``[P * ps, 512]`` (exact)."""
    P, ps, _, bpt = cache_nhd.shape
    assert bpt == EXTRA_BPT
    p = cache_nhd.reshape(P, ps * bpt)
    data = p[:, : ps * 256].reshape(P, ps, 256)
    scales = p[:, ps * 256 :].reshape(P, ps, 32).view(torch.float8_e4m3fn).float()
    codes = torch.empty(P, ps, D, dtype=torch.uint8, device=cache_nhd.device)
    codes[..., 0::2] = data & 0xF
    codes[..., 1::2] = data >> 4
    mags = torch.tensor(_E2M1_MAGNITUDES, device=cache_nhd.device)
    vals = mags[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -vals, vals)
    out = vals.view(P, ps, 32, 16) * scales.unsqueeze(-1)
    return out.reshape(P * ps, D).to(torch.bfloat16)


def _hnd_to_nhd(cache_hnd):
    return cache_hnd.permute(0, 2, 1, 3).contiguous()


def _mask_tail(indices, lengths):
    ref = indices.clone()
    if lengths is not None:
        positions = torch.arange(indices.shape[1], device=indices.device)
        ref[positions.unsqueeze(0) >= lengths.unsqueeze(1)] = -1
    return ref


def _reference_sparse_attention(q, kv, indices, sm_scale, attn_sink=None):
    """fp32 sparse attention; ``-1`` masks a slot; LSE in base 2."""
    num_tokens, num_heads, dim = q.shape
    topk = indices.shape[1]
    invalid = indices < 0
    gathered = kv.index_select(0, indices.clamp_min(0).reshape(-1).long())
    gathered = gathered.reshape(num_tokens, topk, dim)
    logits = torch.einsum("thd,tkd->thk", q, gathered) * sm_scale
    logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    safe_lse = torch.where(torch.isneginf(lse), torch.inf, lse)
    weights = torch.exp(logits - safe_lse.unsqueeze(-1))
    output = torch.einsum("thk,tkd->thd", weights, gathered)
    if attn_sink is not None:
        sink = attn_sink.float().unsqueeze(0)
        output *= torch.sigmoid(lse - sink).unsqueeze(-1)
        lse = torch.logaddexp(lse, sink)
    return output.to(torch.bfloat16), lse * math.log2(math.e)


def _reference(
    q,
    main_cache,
    indices,
    *,
    lengths=None,
    extra_cache=None,
    extra_indices=None,
    extra_lengths=None,
    attn_sink=None,
    lse_scale=1.0,
):
    kv = _dequantize_dsv41_fp8(_hnd_to_nhd(main_cache)).float()
    ref_indices = _mask_tail(indices, lengths)
    if extra_cache is not None:
        main_rows = kv.shape[0]
        kv = torch.cat((kv, _dequantize_dsv41_fp4(_hnd_to_nhd(extra_cache)).float()))
        ref_extra = _mask_tail(extra_indices, extra_lengths)
        ref_indices = torch.cat(
            (ref_indices, torch.where(ref_extra < 0, ref_extra, ref_extra + main_rows)),
            dim=1,
        )
    output, lse = _reference_sparse_attention(
        q.float(), kv, ref_indices, SM_SCALE, attn_sink=attn_sink
    )
    return output, (lse * lse_scale).float()


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _latent(num_pages, page_size, device):
    return (
        torch.randn(num_pages, page_size, D, dtype=torch.bfloat16, device=device) / 10
    ).clamp(-1, 1)


def _query(num_tokens, num_heads, device):
    return (
        torch.randn(num_tokens, num_heads, D, dtype=torch.bfloat16, device=device) / 10
    ).clamp(-1, 1)


def _indices(num_tokens, topk, num_slots, device):
    return torch.randint(
        0, num_slots, (num_tokens, topk), dtype=torch.int32, device=device
    )


def _lengths(num_tokens, topk, device):
    return torch.randint(
        topk // 2, topk + 1, (num_tokens,), dtype=torch.int32, device=device
    )


def _decode(fn, q, main, indices, **kwargs):
    """Allocation-free decode with caller-owned scratch covering every plan."""
    T, H = q.shape[:2]
    extra_indices = kwargs.get("extra_indices")
    chunks = cake.sparse_mla_sm120_dsv41_mixed_num_chunks(
        int(indices.shape[1]),
        int(extra_indices.shape[1]) if extra_indices is not None else 0,
    )
    output = torch.empty_like(q)
    out_lse = torch.empty(T, H, dtype=torch.float32, device=q.device)
    mid_out = torch.empty(T, H, chunks, D, dtype=torch.bfloat16, device=q.device)
    mid_lse = torch.empty(T, H, chunks, dtype=torch.float32, device=q.device)
    plan = fn(
        q,
        main,
        indices,
        output,
        out_lse,
        SM_SCALE,
        mid_out=mid_out,
        mid_lse=mid_lse,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert plan["num_splits"] <= chunks
    return output, out_lse, plan


# --------------------------------------------------------------------------
# GPU: FP8 main-cache writers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [64, 61])
def test_fp8_pack_and_append_match_flashinfer_and_reference(page_size):
    _skip_unless_sm12x(cake.FI_API_MODULE, cake.FI_API_JIT_MODULE)
    from flashinfer.mla import dsv41_fp8_quantize_append_sparse_mla_cache as fi_append
    from flashinfer.mla import dsv41_fp8_quantize_pack_sparse_mla_cache as fi_pack

    device = torch.device("cuda")
    torch.manual_seed(20261003 + page_size)
    num_pages = 8
    latent = _latent(num_pages, page_size, device)
    assert cake.supports_dsv41_fp8_quantize_pack(latent)
    assert cake.supports_dsv41_fp8_quantize_pack(latent, kv_layout="NHD")
    for layout in ("HND", "NHD"):
        cache = cake_dsv41_fp8_quantize_pack_sparse_mla_cache(latent, kv_layout=layout)
        expected_shape = (
            (num_pages, 1, page_size, MAIN_BPT)
            if layout == "HND"
            else (num_pages, page_size, 1, MAIN_BPT)
        )
        assert cache.dtype == torch.uint8 and tuple(cache.shape) == expected_shape
        assert torch.equal(cache, fi_pack(latent, kv_layout=layout))
        nhd = cache if layout == "NHD" else _hnd_to_nhd(cache)
        # Bit-identical to the torch DSv4.1 FOOTER quantizer (FlashInfer's reference).
        assert torch.equal(nhd, _quantize_dsv41_fp8(latent))
        torch.testing.assert_close(
            _dequantize_dsv41_fp8(nhd).float(),
            latent.reshape(-1, D).float(),
            **TOL["fp8"],
        )

    # Slot append over every slot (shuffled, int64 and int32 maps) reproduces the pack.
    packed = cake_dsv41_fp8_quantize_pack_sparse_mla_cache(latent, kv_layout="NHD")
    rows = latent.reshape(-1, D).contiguous()
    perm = torch.randperm(rows.shape[0], device=device)
    for slot_dtype in (torch.int64, torch.int32):
        slots = perm.to(slot_dtype)
        appended = torch.zeros_like(packed)
        appended_fi = torch.zeros_like(packed)
        assert cake.supports_dsv41_fp8_quantize_append(rows[perm], slots, appended)
        cake_dsv41_fp8_quantize_append_sparse_mla_cache(rows[perm], slots, appended)
        fi_append(rows[perm], slots, appended_fi)
        torch.cuda.synchronize()
        assert torch.equal(appended, appended_fi)
        assert torch.equal(appended, packed)


# --------------------------------------------------------------------------
# GPU: mixed-cache decode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("with_extra", [False, True])
def test_mixed_decode_matches_flashinfer_and_reference(num_heads, with_extra):
    _skip_unless_family()
    from flashinfer.mla import cake_sparse_mla_sm120_dsv41_mixed_decode as fi_decode
    from flashinfer.mla import (
        dsv41_fp4_quantize_pack_sparse_mla_cache,
    )

    device = torch.device("cuda")
    torch.manual_seed(20261003 + num_heads + int(with_extra))
    num_tokens, topk, extra_topk = 8, 128, 512
    main_page, extra_page = 64, 32
    main_pages, extra_pages = 8, 16
    if num_heads not in cake.sparse_mla_sm120_dsv41_mixed_supported_heads():
        pytest.skip(f"{num_heads} heads not exported by this family")
    q = _query(num_tokens, num_heads, device)
    main_latent = _latent(main_pages, main_page, device)
    main = cake_dsv41_fp8_quantize_pack_sparse_mla_cache(main_latent)
    indices = _indices(num_tokens, topk, main_pages * main_page, device)
    indices[:, topk - 8 :] = -1
    lengths = _lengths(num_tokens, topk, device)
    sink = torch.randn(num_heads, device=device)
    kwargs = dict(topk_length=lengths, attn_sink=sink, lse_scale=0.5)
    ref_kwargs = dict(lengths=lengths, attn_sink=sink, lse_scale=0.5)
    if with_extra:
        extra_latent = _latent(extra_pages, extra_page, device)
        # Pre-baseline hand-written FP4 writer (not a Cake entry): reached directly.
        extra = dsv41_fp4_quantize_pack_sparse_mla_cache(extra_latent)
        torch.testing.assert_close(
            _dequantize_dsv41_fp4(_hnd_to_nhd(extra)).float(),
            extra_latent.reshape(-1, D).float(),
            **TOL["fp4"],
        )
        extra_indices = _indices(
            num_tokens, extra_topk, extra_pages * extra_page, device
        )
        extra_indices[::2, :16] = -1
        extra_lengths = _lengths(num_tokens, extra_topk, device)
        kwargs.update(
            extra_kv_cache=extra,
            extra_indices=extra_indices,
            extra_topk_length=extra_lengths,
        )
        ref_kwargs.update(
            extra_cache=extra, extra_indices=extra_indices, extra_lengths=extra_lengths
        )
    assert cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        main,
        indices,
        extra_kv_cache=kwargs.get("extra_kv_cache"),
        extra_indices=kwargs.get("extra_indices"),
    )

    out, lse, plan = _decode(
        cake_sparse_mla_sm120_dsv41_mixed_decode, q, main, indices, **kwargs
    )
    out_fi, lse_fi, plan_fi = _decode(fi_decode, q, main, indices, **kwargs)
    assert plan == plan_fi
    assert torch.equal(out, out_fi) and torch.equal(lse, lse_fi)

    expected = _reference(q, main, indices, **ref_kwargs)
    torch.testing.assert_close(out, expected[0], **TOL["bf16"])
    torch.testing.assert_close(lse, expected[1], **TOL["bf16"])

    # Wrapper route (wrapper-owned scratch) is bitwise identical to the direct entry.
    wrapper = cake_create_sparse_mla_sm120_dsv41_mixed_wrapper(device=device)
    output_w = torch.empty_like(q)
    lse_w = wrapper.run(
        q,
        main,
        indices,
        output_w,
        SM_SCALE,
        return_lse=True,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(output_w, out) and torch.equal(lse_w, lse)


def test_mixed_decode_fp8_route_matches_reference():
    _skip_unless_family()
    if "fp8" not in cake.sparse_mla_sm120_dsv41_mixed_compute_precisions():
        pytest.skip("compute_precision='fp8' not exported by this family")
    device = torch.device("cuda")
    torch.manual_seed(20261004)
    num_tokens, num_heads, topk, extra_topk = 8, 64, 128, 512
    from flashinfer.mla import dsv41_fp4_quantize_pack_sparse_mla_cache

    q = _query(num_tokens, num_heads, device)
    main = cake_dsv41_fp8_quantize_pack_sparse_mla_cache(_latent(8, 64, device))
    extra = dsv41_fp4_quantize_pack_sparse_mla_cache(_latent(8, 64, device))
    indices = _indices(num_tokens, topk, 8 * 64, device)
    extra_indices = _indices(num_tokens, extra_topk, 8 * 64, device)
    sink = torch.randn(num_heads, device=device)
    assert cake.supports_sparse_mla_sm120_dsv41_mixed_decode(
        q,
        main,
        indices,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        compute_precision="fp8",
    )
    out, lse, _ = _decode(
        cake_sparse_mla_sm120_dsv41_mixed_decode,
        q,
        main,
        indices,
        extra_kv_cache=extra,
        extra_indices=extra_indices,
        attn_sink=sink,
        compute_precision="fp8",
    )
    expected = _reference(
        q, main, indices, extra_cache=extra, extra_indices=extra_indices, attn_sink=sink
    )
    torch.testing.assert_close(out, expected[0], **TOL["fp8"])
    torch.testing.assert_close(lse, expected[1], **TOL["fp8"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
