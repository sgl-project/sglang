"""ROCm sparse MLA over packed FP8 KV: numerical references and launch tuning."""

from types import SimpleNamespace

import pytest
import torch
import triton

import sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 as mla_gfx950
from sglang.kernels.ops.attention.dsa import sparse_mla_prefill_gfx950
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _cu_count,
    _cu_count_for_device,
    _is_gfx950_sparse_mla_fp8,
    _page_offsets_fit_i32,
    _reduce_d_chunk,
    _sparse_mla_reduce_kernel,
    triton_sparse_mla_fwd,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    _get_splitk_bufs,
    _splitk_bufs,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=35, stage="jit-kernel-unit", runner_config="amd")
# backend-specific: the topk_length prefill path is gfx950-only.
register_amd_ci(est_time=155, suite="stage-b-test-1-gpu-small-amd-mi35x")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")

_IS_GFX950 = is_hip() and _is_gfx950_sparse_mla_fp8(
    torch.float8_e4m3fn, 16, 512, 64, 576
)


_DEVICE = "cuda"
_DV, _DR, _TOPK, _N = 512, 64, 2048, 16384
_RAGGED = [256, 0, 1, 63, 65, 2048, 300, 200]
_PREFILL_POOL = 8192


def test_cu_count_uses_tensor_device(monkeypatch):
    requested = []

    def get_device_core_count(device):
        requested.append(device)
        return 123

    _cu_count_for_device.cache_clear()
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.dsa.triton_sparse_mla.get_device_core_count",
        get_device_core_count,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("current_device must not be used"),
    )
    assert _cu_count(torch.device("cuda:7")) == 123
    assert requested == [7]
    _cu_count_for_device.cache_clear()


def test_page_offset_i32_boundary():
    max_pages = ((1 << 31) - 1) // 576
    assert _page_offsets_fit_i32(max_pages, 576)
    assert not _page_offsets_fit_i32(max_pages + 1, 576)


def test_gfx950_fp8_gate_supports_tp8_prefill(monkeypatch):
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.dsa.triton_sparse_mla.is_gfx95_supported",
        lambda: True,
    )
    for heads in (8, 16):
        assert _is_gfx950_sparse_mla_fp8(torch.float8_e4m3fn, heads, 512, 64, 576)
    assert not _is_gfx950_sparse_mla_fp8(torch.float8_e4m3fn, 4, 512, 64, 576)


def test_splitk_workspaces_are_graph_and_stream_safe():
    workspace = []
    device = torch.device("cuda")
    _get_splitk_bufs(1, 1, 16, 4, device, workspace)
    first = workspace[0]
    grown_bs = 2 * first[0].numel() // 16
    _get_splitk_bufs(grown_bs, 1, 16, 4, device, workspace)
    assert len(workspace) == 2
    assert workspace[0][0] is first[0]
    assert workspace[0][1] is first[1]

    # Capacity is rounded up, so a run of growing shapes must not allocate per shape.
    for extra in range(1, 9):
        _get_splitk_bufs(grown_bs + extra, 1, 16, 4, device, workspace)
    assert len(workspace) == 3

    _splitk_bufs.clear()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pointers = []
    for stream in streams:
        with torch.cuda.stream(stream):
            lse, _ = _get_splitk_bufs(1, 1, 16, 4, device)
            pointers.append(lse.data_ptr())
    assert pointers[0] != pointers[1]
    _splitk_bufs.clear()


def _scaled_fp8_ds_mla_kv(pool, seed):
    """[pool, 1, 656] fp8_ds_mla rows (fp8 NoPE, 4 fp32 scales, bf16 RoPE) and their fp32 keys."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nope = (torch.randn(pool, 512, device="cuda", generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pool, 4, device="cuda", generator=g) / 32 + 1 / 128
    rope = (torch.randn(pool, 64, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    kv = torch.empty(pool, 1, 656, dtype=torch.uint8, device="cuda")
    kv[:, 0, :512] = nope.view(torch.uint8)
    kv[:, 0, 512:528] = scales.view(torch.uint8)
    kv[:, 0, 528:] = rope.view(torch.uint8)
    keys = torch.cat(
        [nope.float() * scales.repeat_interleave(128, -1), rope.float()], -1
    )
    return kv, keys


def _sparse_mla_reference(q, keys, idx, sm_scale, lengths=None):
    if lengths is not None:
        rows, heads, _ = q.shape
        out = torch.zeros(rows, heads, 512, device=_DEVICE)
        lse = torch.full((rows, heads), -float("inf"), device=_DEVICE)
        for row in range(rows):
            n = int(lengths[row])
            if n:
                k = keys[idx[row, :n].long()]
                scores = q[row] @ k.T * sm_scale
                lse[row] = torch.logsumexp(scores, -1)
                out[row] = torch.softmax(scores, -1) @ k[:, :512]
        return out, lse
    gathered = keys[idx.clamp(min=0).long()]
    scores = torch.einsum("thd,tkd->thk", q, gathered) * sm_scale
    scores = scores.masked_fill((idx < 0)[:, None, :], float("-inf"))
    lse = torch.logsumexp(scores, -1)
    out = torch.einsum(
        "thk,tkd->thd", torch.softmax(scores, -1).nan_to_num(0.0), gathered[..., :512]
    )
    return out, lse


@pytest.mark.skipif(not _IS_GFX950, reason="gfx950 long-prefill path")
def test_long_prefill_scaled_kv_matches_reference(monkeypatch):
    """The Triton single-pass prefill dequantizes the fp8_ds_mla rows to bf16 before the
    dots; it must stay close to an fp32 reference."""
    # The Triton single-pass prefill, not the gfx950 HIP kernel it otherwise hands to.
    monkeypatch.setattr(
        sparse_mla_prefill_gfx950,
        "supports_sparse_mla_prefill_gfx950",
        lambda *a: False,
    )
    seq, heads, topk, pool = 1024, 12, 2048, 8192  # seq > CU count: single-pass prefill
    kv, keys = _scaled_fp8_ds_mla_kv(pool, seed=11)
    torch.manual_seed(11)
    q_nope = torch.randn(seq, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    q_rope = torch.randn(seq, heads, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(0, pool, (seq, topk), device="cuda", dtype=torch.int32)
    idx[:4, -300:] = -1
    sm_scale = 576**-0.5

    out = triton_sparse_mla_fwd(q_nope, q_rope, kv, idx.unsqueeze(1), sm_scale)

    q = torch.cat([q_nope.float(), q_rope.float()], -1)
    ref, _ = _sparse_mla_reference(q, keys, idx, sm_scale)
    err = ((out[0].float() - ref).norm() / ref.norm()).item()
    assert err < 4e-3, err


@pytest.mark.skipif(not _IS_GFX950, reason="topk_length path is gfx950-only")
@pytest.mark.parametrize("topk", [2048, 2050])
def test_short_prefill_index_bound_matches_full_scan(topk):
    torch.manual_seed(29)
    seq, heads, value_dim, tail_dim = 5, 16, 512, 64
    q = torch.randn(
        seq, heads, value_dim + tail_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = (
        torch.randn(256, 1, value_dim + tail_dim, device="cuda")
        .clamp_(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    indices = torch.full((seq, 1, topk), -1, device="cuda", dtype=torch.int32)
    for row in range(seq - 1):
        positions = torch.arange(row + 1, device="cuda") * 31 + 5
        indices[row, 0, positions] = torch.randint(
            0, kv.shape[0], (row + 1,), device="cuda", dtype=torch.int32
        )

    args = (
        q[:, :, :value_dim],
        q[:, :, value_dim:],
        kv,
        indices,
        value_dim**-0.5,
        value_dim,
    )
    expected = triton_sparse_mla_fwd(*args)
    actual = triton_sparse_mla_fwd(*args, max_topk_length=128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _IS_GFX950, reason="fp8 prefill path is gfx950-only")
def test_head_major_q_compiles_once_across_token_counts():
    # A head-major q ([H, T, D] storage viewed as [T, H, D]) has head stride T * D. Specializing
    # on it compiled a new kernel for every prefill length; a new length must reuse the kernel.
    heads, value_dim, tail_dim, topk = 16, 512, 64, 2048
    kv = (
        torch.randn(4096, 1, value_dim + tail_dim, device="cuda")
        .clamp_(-2, 2)
        .to(torch.float8_e4m3fn)
    )

    def run(seq):
        q = torch.randn(heads, seq, value_dim + tail_dim, device="cuda").to(
            torch.bfloat16
        )
        indices = torch.randint(
            0, kv.shape[0], (seq, 1, topk), device="cuda", dtype=torch.int32
        )
        q = q.transpose(0, 1)
        triton_sparse_mla_fwd(
            q[:, :, :value_dim],
            q[:, :, value_dim:],
            kv,
            indices,
            value_dim**-0.5,
            value_dim,
        )

    compiled = []
    previous = triton.knobs.runtime.jit_post_compile_hook
    triton.knobs.runtime.jit_post_compile_hook = lambda **kw: compiled.append(
        kw["fn"].name
    )
    try:
        run(40)
        compiled.clear()
        run(56)
        torch.cuda.synchronize()
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous
    assert not [name for name in compiled if name.startswith("_sparse_mla")], compiled


@pytest.mark.parametrize("active_splits", [2, 4])
def test_reduce_d_chunk_is_bitwise_identical(active_splits):
    torch.manual_seed(active_splits)
    batch, heads, value_dim = 96, 16, 512
    lse = torch.randn(batch, active_splits, heads, device="cuda")
    acc = torch.randn(
        batch, active_splits, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )

    def run(d_chunk):
        out = torch.empty(batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
        _sparse_mla_reduce_kernel[(batch, heads, value_dim // d_chunk)](
            lse,
            acc,
            out,
            lse,  # lse_out_ptr, unread without STORE_LSE
            H=heads,
            D_V=value_dim,
            KV_SPLITS=active_splits,
            ACTIVE_SPLITS=active_splits,
            ACTIVE_SPLITS_POW2=active_splits,
            D_CHUNK=d_chunk,
            BLOCK_K=64,
            num_warps=4,
        )
        return out

    torch.testing.assert_close(
        run(_reduce_d_chunk(active_splits, batch * heads)), run(64), rtol=0, atol=0
    )


def _pow2_scaled_kv():
    torch.manual_seed(0)
    k = (torch.randn(_N, 1, _DV + _DR, device=_DEVICE) * 2).bfloat16()
    kv = quantize_k_cache(k.unsqueeze(1), scale_ue8m0=True).view(_N, 1, -1)
    raw = kv.view(torch.uint8).view(_N, 656)
    nope = raw[:, :512].view(torch.float8_e4m3fn).float().view(_N, 4, 128)
    scale = raw[:, 512:528].contiguous().view(torch.float32).view(_N, 4, 1)
    rope = raw[:, 528:].contiguous().view(torch.bfloat16).float()
    return kv, torch.cat([(nope * scale).view(_N, 512), rope], -1)


def _decode_reference(qn, qr, idx, lens, kd, sm, mxfp8_q=True):
    """FP32 attention using the production MXFP8 Q quantizer and dequantized KV."""
    R, H, _ = qn.shape
    if not mxfp8_q:
        return _sparse_mla_reference(
            torch.cat([qn.float(), qr.float()], -1), kd, idx, sm, lens
        )
    q8 = torch.empty(R, H, 512, device=_DEVICE, dtype=torch.float8_e4m3fn)
    qs = torch.empty(R, H, 16, device=_DEVICE, dtype=torch.uint8)
    qrc = torch.empty(R, H, 64, device=_DEVICE, dtype=torch.bfloat16)
    mla_gfx950._mxfp8_quant_kernel[(R, H // 16)](
        qn,
        q8,
        qs,
        qr,
        qrc,
        qn.stride(0),
        qn.stride(1),
        qr.stride(0),
        qr.stride(1),
        H,
        HB=16,
    )
    scale = torch.exp2(qs.float() - 127).view(R, H, 16, 1)
    qd = torch.cat(
        [(q8.float().view(R, H, 16, 32) * scale).view(R, H, 512), qr.float()], -1
    )
    return _sparse_mla_reference(qd, kd, idx, sm, lens)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize(
    "H,lens,splits",
    [
        pytest.param(16, _RAGGED, 1, id="h16-unsplit"),
        pytest.param(16, _RAGGED, 2, id="h16-two-splits"),
        pytest.param(16, _RAGGED, 8, id="h16-eight-splits"),
        pytest.param(96, _RAGGED, 1, id="h96-unsplit"),
        pytest.param(96, _RAGGED, 2, id="h96-two-splits"),
        pytest.param(96, _RAGGED, 8, id="h96-eight-splits"),
        pytest.param(128, _RAGGED, 1, id="h128-unsplit"),
        pytest.param(128, _RAGGED, 2, id="h128-two-splits"),
        pytest.param(128, _RAGGED, 8, id="h128-eight-splits"),
        pytest.param(16, [256] * 8, None, id="auto-eight-splits"),
        pytest.param(96, [100] * 10, None, id="auto-four-splits"),
        pytest.param(128, [40] * 33, None, id="auto-unsplit"),
    ],
)
def test_decode_matches_reference(H, lens, splits):
    kv, kd = _pow2_scaled_kv()
    R = len(lens)
    qkv = torch.randn(R, H, _DV + _DR, device=_DEVICE).bfloat16()
    qn, qr = qkv[..., :_DV], qkv[..., _DV:]
    lengths = torch.tensor(lens, dtype=torch.int32, device=_DEVICE)
    idx = torch.full((R, _TOPK), -1, dtype=torch.int32, device=_DEVICE)
    for r, n in enumerate(lens):
        idx[r, :n] = torch.randperm(_N, device=_DEVICE)[:n].int()
    sm = (_DV + _DR) ** -0.5
    o, lse = mla_gfx950.sparse_mla_decode_gfx950(
        qn, qr, kv, idx.unsqueeze(1), sm, topk_length=lengths, num_splits=splits
    )
    o_ref, lse_ref = _decode_reference(qn, qr, idx, lengths, kd, sm)
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    torch.testing.assert_close(lse[finite], lse_ref[finite], rtol=0, atol=1e-3)
    torch.testing.assert_close(o[0].float(), o_ref, rtol=0, atol=3e-2)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
def test_decode_without_row_lengths_skips_the_padding():
    """Decode without DCP hands the backend -1-padded top-k rows and no lengths."""
    kv, kd = _pow2_scaled_kv()
    lens = [_TOPK, 300, 1, 1024]
    R, H = len(lens), 96
    qkv = torch.randn(R, H, _DV + _DR, device=_DEVICE).bfloat16()
    qn, qr = qkv[..., :_DV], qkv[..., _DV:]
    idx = torch.full((R, _TOPK), -1, dtype=torch.int32, device=_DEVICE)
    for r, n in enumerate(lens):
        idx[r, :n] = torch.randperm(_N, device=_DEVICE)[:n].int()
    sm = (_DV + _DR) ** -0.5
    backend = SimpleNamespace(_triton_sparse_mla_workspaces={}, attn_dcp_size=1)
    o = DeepseekSparseAttnBackend._forward_triton_decode(
        backend, qn, qr, kv, _DV, idx, sm
    )
    lengths = torch.tensor(lens, dtype=torch.int32, device=_DEVICE)
    # the Triton split-K kernel keeps q in BF16
    o_ref, _ = _decode_reference(qn, qr, idx, lengths, kd, sm, mxfp8_q=False)
    torch.testing.assert_close(o.float(), o_ref, rtol=0, atol=3e-2)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 only")
@pytest.mark.parametrize(
    "H,topk,q_mul,head_major",
    [
        (8, 2048, 30.0, False),
        (8, 1000, 1.0, True),
        (12, 2048, 1.0, False),
        (12, 1000, 30.0, True),
        (16, 2048, 1.0, True),
        (16, 1000, 30.0, False),
    ],
)
def test_sparse_mla_prefill_gfx950(H, topk, q_mul, head_major):
    T = 96
    kv, k = _scaled_fp8_ds_mla_kv(_PREFILL_POOL, seed=H + topk)
    torch.manual_seed(topk + H)
    q_nope = torch.randn(T, H, _DV, device=_DEVICE, dtype=torch.bfloat16) * (
        0.1 * q_mul
    )
    q_rope = torch.randn(T, H, _DR, device=_DEVICE, dtype=torch.bfloat16) * (
        0.1 * q_mul
    )
    if head_major:
        q_nope = q_nope.transpose(0, 1).contiguous().transpose(0, 1)
        q_rope = q_rope.transpose(0, 1).contiguous().transpose(0, 1)
    idx = torch.randint(0, _PREFILL_POOL, (T, topk), device=_DEVICE, dtype=torch.int32)
    idx[:8, -300:] = -1
    idx[8] = -1
    sm_scale = (_DV + _DR) ** -0.5
    assert sparse_mla_prefill_gfx950.supports_sparse_mla_prefill_gfx950(
        q_nope, q_rope, kv, _DV
    )

    out, lse = sparse_mla_prefill_gfx950.sparse_mla_prefill_gfx950(
        q_nope, q_rope, kv, idx.unsqueeze(1), sm_scale, _DV, return_lse=True
    )
    ref_out, ref_lse = _sparse_mla_reference(
        torch.cat([q_nope.float(), q_rope.float()], -1), k, idx, sm_scale
    )

    assert out.shape == (1, T, H, _DV) and lse.shape == (T, H)
    assert torch.all(out[0, 8] == 0) and torch.all(torch.isneginf(lse[8]))
    keep = torch.arange(T, device=_DEVICE) != 8
    o, r = out[0, keep].float(), ref_out[keep]
    assert ((o - r).norm() / r.norm()).item() < 4e-3
    torch.testing.assert_close(lse[keep], ref_lse[keep], rtol=0, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
