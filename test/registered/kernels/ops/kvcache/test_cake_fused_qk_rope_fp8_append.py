"""Cake fused QK RMSNorm + NeoX RoPE + FP8 quantize + paged KV append via sglang.kernels.

FlashInfer entry ``flashinfer.cake_fused_qk_rope_fp8_append``
(``cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache``), added after
the ``46340689a5ab`` baseline by FlashInfer PR flashinfer-ai/flashinfer#5956
(commit ``335c4eb4e``; read at main ``e4f94f948``). Checks four things for the
Cake adapter: the registry resolves the explicit FlashInfer backend; the
``supports_*`` gate admits the documented contract and rejects deviations; the
facade result is bitwise identical to calling FlashInfer directly; and the fused
result matches a pure-torch fp32 reference (FlashInfer's own oracle: unrounded
``x / scale`` payloads, ``amax / upper_max`` or static Q scale, last-page tail
clear, device-side ``q_indptr`` validation) within the FP8 tolerance
``atol = rtol = 0.1`` on both the FP8 payloads and the dequantized values;
``q_scale`` at ``rtol = 1e-5``; flags, cleared tails and every untouched byte
bitwise. Skips (with the reason) when the installed FlashInfer lacks the Cake
module or the GPU is outside sm_90a / sm_100a / sm_103a (no SM120/121 cubin).
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import kvcache as cake_kvcache
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.kvcache.cake import (
    cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

OP = "kvcache.fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache"
HEAD_DIM = 128
FP8_MAX = 448.0
AMAX_FLOOR = 1e-6
EPS = 1e-6
POISON_BYTE = 0xFF  # NaN in float8_e4m3fn: never produced by the kernel
FP8_TOL = dict(atol=0.1, rtol=0.1)


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.kvcache:")
    assert spec.load() is not None


def test_supports_rejects_outside_the_contract(monkeypatch):
    # CPU tensors are never admitted (no CUDA device of a supported arch).
    qkv = torch.zeros(2, 10 * HEAD_DIM, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=1
    )
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    qkv = qkv.to(device)
    cache = cache.to(device)
    ok = cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=1
    )
    # Deviations are rejected whatever the device: BF16 cache, unsupported head
    # configuration, unknown quant policy, out-of-range upper_max, missing module.
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache.to(torch.bfloat16), quant_policy=1
    )
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv[:, : 9 * HEAD_DIM].contiguous(), cache, quant_policy=1
    )  # (7, 1) heads
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=3
    )
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=1, upper_max=448.01
    )
    monkeypatch.setattr(cake_kvcache, "flashinfer_module_available", lambda *a: False)
    assert not cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=1
    )
    monkeypatch.undo()
    assert ok == cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        qkv, cache, quant_policy=1
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_kvcache.FI_FP8_MODULE, cake_kvcache.FI_FP8_JIT_MODULE
    ):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.cake_fused_qk_rope_fp8_append"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in cake_kvcache.FP8_ARCHS:
        pytest.skip(
            f"Cake FP8 rope-append is built for sm_90a/100a/103a, device is {cc}"
        )


# --------------------------------------------------------------------------
# fp32 reference (FlashInfer's oracle): unrounded payloads in output layout
# --------------------------------------------------------------------------


def _cos_sin_table(max_positions: int, device, base: float = 10000.0):
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def _rmsnorm(x, w):
    inv = torch.rsqrt(x.square().sum(dim=-1, keepdim=True) / x.shape[-1] + EPS)
    return x * inv * w.float()


def _rope(x, cos_sin, positions):
    half = HEAD_DIM // 2
    cos = cos_sin[positions, :half].float().unsqueeze(1)
    sin = cos_sin[positions, half:].float().unsqueeze(1)
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)


def _reference(case):
    """Expected unrounded payloads; caches hold NaN where the kernel must not write."""
    qkv, dev = case["qkv"], case["qkv"].device
    T = qkv.shape[0]
    hq, hkv = case["num_q_heads"], case["num_kv_heads"]
    B = case["seq_lens"].shape[0]
    page_size = case["page_size"]
    qi = case["q_indptr"].tolist()
    sl = case["seq_lens"].tolist()
    pi = case["page_indices"].cpu()
    quant, norm = case["quant_policy"], case["qk_norm_policy"]
    dyn_prefill = quant == 1 and case["is_prefill"]
    aligned = (case["max_seqlen"] + 127) // 128 * 128 if dyn_prefill else 0

    batch_ids = torch.empty(T, dtype=torch.int64)
    positions = torch.empty(T, dtype=torch.int64)
    for b in range(B):
        lo, hi = qi[b], qi[b + 1]
        if hi > lo:
            batch_ids[lo:hi] = b
            positions[lo:hi] = torch.arange(lo, hi) + sl[b] - hi

    x = qkv.float()
    q = x[:, : hq * HEAD_DIM].reshape(T, hq, HEAD_DIM)
    k = x[:, hq * HEAD_DIM : (hq + hkv) * HEAD_DIM].reshape(T, hkv, HEAD_DIM)
    v = x[:, (hq + hkv) * HEAD_DIM :].reshape(T, hkv, HEAD_DIM)
    qw, kw = case["q_norm_weight"], case["k_norm_weight"]
    if norm == 2:
        q, k = _rmsnorm(q, qw), _rmsnorm(k, kw)
    pos_dev = positions.to(dev)
    q, k = _rope(q, case["cos_sin"], pos_dev), _rope(k, case["cos_sin"], pos_dev)
    if norm == 1:
        q, k = _rmsnorm(q, qw), _rmsnorm(k, kw)
    if quant == 1:
        scale = torch.clamp(q.abs().amax(dim=-1), min=AMAX_FLOOR) / case["upper_max"]
        q_payload = q / scale.unsqueeze(-1)
        if dyn_prefill:
            q_scale = torch.full((B, hq, aligned), float("nan"), device=dev)
            for b in range(B):
                lo, hi = qi[b], qi[b + 1]
                if hi > lo:
                    q_scale[b, :, : hi - lo] = scale[lo:hi].t()
        else:
            q_scale = scale
    else:
        q_payload = q * case["q_scale_inv"].float().reshape(())
        q_scale = torch.empty(0, device=dev)
    k_payload = k / case["k_scale"].float().reshape(())
    v_payload = v / case["v_scale"].float().reshape(())
    key_cache = torch.full(case["cache_shape"], float("nan"), device=dev)
    value_cache = key_cache.clone()
    phys = pi[batch_ids, positions // page_size].to(dev)
    slot = (positions % page_size).to(dev)
    key_cache[phys, slot] = k_payload
    value_cache[phys, slot] = v_payload
    for b in range(B):
        last = sl[b] - 1
        page = int(pi[b, last // page_size])
        zero_from = last % page_size + 1
        if zero_from < page_size:
            key_cache[page, zero_from:] = 0.0
            value_cache[page, zero_from:] = 0.0
    return dict(
        q=q,
        k=k,
        v=v,
        out_q=q_payload,
        q_scale=q_scale,
        key_cache=key_cache,
        value_cache=value_cache,
        phys=phys,
        slot=slot,
    )


def _make_case(
    *, q_lens, ctx_lens, heads, page_size, qk_norm_policy, quant_policy, seed
):
    hq, hkv = heads
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = len(q_lens)
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    max_pages = max(pages_per_req)
    total_pages = sum(pages_per_req) + 3
    perm = torch.randperm(total_pages, generator=g)
    page_indices = torch.zeros((B, max_pages), dtype=torch.int32)
    cursor = 0
    for b in range(B):
        n = pages_per_req[b]
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    width = (hq + 2 * hkv) * HEAD_DIM
    qkv = torch.randn((T, width), generator=g).to(torch.bfloat16).to(device)
    is_prefill = max(q_lens) > 1
    max_seqlen = max(q_lens) if quant_policy == 1 and is_prefill else 0
    return dict(
        qkv=qkv,
        cos_sin=_cos_sin_table(max(seq_lens) + 1, device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        cache_shape=(total_pages, page_size, hkv, HEAD_DIM),
        page_size=page_size,
        num_q_heads=hq,
        num_kv_heads=hkv,
        qk_norm_policy=qk_norm_policy,
        quant_policy=quant_policy,
        is_prefill=is_prefill,
        max_seqlen=max_seqlen,
        upper_max=FP8_MAX,
        q_norm_weight=(torch.rand(HEAD_DIM, generator=g) + 0.5)
        .to(torch.bfloat16)
        .float()
        .to(device),
        k_norm_weight=(torch.rand(HEAD_DIM, generator=g) + 0.5)
        .to(torch.bfloat16)
        .float()
        .to(device),
        k_scale=torch.tensor([0.02], dtype=torch.float32, device=device),
        v_scale=torch.tensor([0.03], dtype=torch.float32, device=device),
        q_scale_inv=torch.tensor([1.0 / 0.02], dtype=torch.float32, device=device),
    )


def _poisoned_fp8(shape, device):
    t = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    t.view(torch.uint8).fill_(POISON_BYTE)
    return t


def _buffers(case):
    dev = case["qkv"].device
    T, hq, hkv = case["qkv"].shape[0], case["num_q_heads"], case["num_kv_heads"]
    B = case["seq_lens"].shape[0]
    if case["quant_policy"] == 2:
        q_scale = torch.empty(0, dtype=torch.float32, device=dev)
    elif case["is_prefill"]:
        aligned = (case["max_seqlen"] + 127) // 128 * 128
        q_scale = torch.full((B, hq, aligned), float("nan"), device=dev)
    else:
        q_scale = torch.full((T, hq), float("nan"), device=dev)
    return dict(
        key_cache=_poisoned_fp8(case["cache_shape"], dev),
        value_cache=_poisoned_fp8(case["cache_shape"], dev),
        out_q=_poisoned_fp8((T, hq, HEAD_DIM), dev),
        q_scale=q_scale,
        split_k_flag=torch.full((B, hkv), 7, dtype=torch.int32, device=dev),
    )


def _launch(fn, case, bufs):
    norm = case["qk_norm_policy"]
    return fn(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        (bufs["key_cache"], bufs["value_cache"]),
        case["is_prefill"],
        case["k_scale"],
        case["v_scale"],
        case["quant_policy"],
        max_seqlen=case["max_seqlen"],
        upper_max=case["upper_max"],
        q_scale_inv=case["q_scale_inv"] if case["quant_policy"] == 2 else None,
        q_norm_weight=case["q_norm_weight"] if norm else None,
        k_norm_weight=case["k_norm_weight"] if norm else None,
        qk_norm_policy=norm,
        out_q=bufs["out_q"],
        q_scale=bufs["q_scale"],
        split_k_flag=bufs["split_k_flag"],
    )


def _assert_fp8_payload(got_fp8, expected_f32, label):
    """Written elements within FP8 tolerance; NaN in ``expected`` means untouched poison."""
    written = ~torch.isnan(expected_f32)
    torch.testing.assert_close(
        got_fp8.float()[written],
        expected_f32[written],
        **FP8_TOL,
        msg=lambda m: f"{label}: {m}",
    )
    untouched = got_fp8.view(torch.uint8)[~written]
    assert bool((untouched == POISON_BYTE).all()), f"{label}: wrote outside its range"


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [1, 2])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_matches_flashinfer_and_reference(heads, qk_norm_policy, quant_policy, mode):
    _skip_unless_supported()
    if mode == "decode":
        q_lens, ctx_lens, page_size = [1] * 5, [0, 5, 63, 64, 130], 16
    else:
        q_lens, ctx_lens, page_size = [16, 7, 70], [60, 0, 125], 64
    case = _make_case(
        q_lens=q_lens,
        ctx_lens=ctx_lens,
        heads=heads,
        page_size=page_size,
        qk_norm_policy=qk_norm_policy,
        quant_policy=quant_policy,
        seed=893 + page_size + quant_policy,
    )
    bufs = _buffers(case)
    assert cake_kvcache.supports_fused_qk_rmsnorm_rope_quantize_fp8_append(
        case["qkv"],
        bufs["key_cache"],
        quant_policy=quant_policy,
        qk_norm_policy=qk_norm_policy,
    )
    ret = _launch(
        cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache, case, bufs
    )
    assert ret[0] is bufs["out_q"] and ret[1] is bufs["q_scale"]
    assert ret[2] is bufs["split_k_flag"]

    # Facade vs direct FlashInfer: bitwise on every output buffer.
    from flashinfer.cake_fused_qk_rope_fp8_append import (
        cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache as fi_direct,
    )

    bufs_fi = _buffers(case)
    _launch(fi_direct, case, bufs_fi)
    torch.cuda.synchronize()
    for name in ("out_q", "key_cache", "value_cache"):
        assert torch.equal(
            bufs[name].view(torch.uint8), bufs_fi[name].view(torch.uint8)
        )
    assert torch.equal(
        bufs["q_scale"].nan_to_num(-1.0), bufs_fi["q_scale"].nan_to_num(-1.0)
    )
    assert torch.equal(bufs["split_k_flag"], bufs_fi["split_k_flag"])

    # Pure-torch fp32 reference.
    exp = _reference(case)
    assert bool((bufs["split_k_flag"] == 0).all())
    _assert_fp8_payload(bufs["out_q"], exp["out_q"], "out_q")
    _assert_fp8_payload(bufs["key_cache"], exp["key_cache"], "key_cache")
    _assert_fp8_payload(bufs["value_cache"], exp["value_cache"], "value_cache")
    if quant_policy == 1:
        valid = ~torch.isnan(exp["q_scale"])
        torch.testing.assert_close(
            bufs["q_scale"][valid], exp["q_scale"][valid], rtol=1e-5, atol=1e-6
        )
        assert bool(torch.isnan(bufs["q_scale"][~valid]).all())
        if case["is_prefill"]:
            qi = case["q_indptr"].tolist()
            scale_rows = torch.cat(
                [
                    bufs["q_scale"][b, :, : qi[b + 1] - qi[b]].t()
                    for b in range(len(qi) - 1)
                ]
            )
        else:
            scale_rows = bufs["q_scale"]
        deq_q = bufs["out_q"].float() * scale_rows.unsqueeze(-1)
    else:
        assert bufs["q_scale"].numel() == 0
        deq_q = bufs["out_q"].float() / case["q_scale_inv"].reshape(())
    # Dequantized values against the fp32 reference (same FP8 tolerance).
    torch.testing.assert_close(deq_q, exp["q"], **FP8_TOL)
    deq_k = bufs["key_cache"].float()[exp["phys"], exp["slot"]] * case["k_scale"]
    deq_v = bufs["value_cache"].float()[exp["phys"], exp["slot"]] * case["v_scale"]
    torch.testing.assert_close(deq_k, exp["k"], **FP8_TOL)
    torch.testing.assert_close(deq_v, exp["v"], **FP8_TOL)


def test_invalid_q_indptr_leaves_outputs_untouched():
    _skip_unless_supported()
    case = _make_case(
        q_lens=[4, 1, 3],
        ctx_lens=[1, 3, 60],
        heads=(8, 1),
        page_size=16,
        qk_norm_policy=2,
        quant_policy=1,
        seed=9,
    )
    case["q_indptr"] = torch.tensor(
        [0, 5, 4, 8], dtype=torch.int32, device=case["qkv"].device
    )  # non-monotone
    bufs = _buffers(case)
    _launch(cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache, case, bufs)
    torch.cuda.synchronize()
    assert bool((bufs["split_k_flag"] == -1).all())
    for name in ("out_q", "key_cache", "value_cache"):
        assert bool((bufs[name].view(torch.uint8) == POISON_BYTE).all()), name
    assert bool(torch.isnan(bufs["q_scale"]).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
