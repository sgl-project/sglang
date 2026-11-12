"""ROCm sparse MLA over packed FP8 KV: numerical references and launch tuning."""

import math
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 as mla_gfx950
from sglang.kernels.ops.attention.dsa import sparse_mla_prefill_gfx950
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _IS_FNUZ,
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
    lse_base2_for_dcp_merge,
    triton_sparse_mla_decode_splitk,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.dsa import dsa_backend_hip
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=35, stage="jit-kernel-unit", runner_config="amd")
# backend-specific: the topk_length prefill path is gfx950-only.
register_amd_ci(est_time=35, suite="stage-b-test-1-gpu-small-amd-mi35x")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")

_IS_GFX950 = is_hip() and _is_gfx950_sparse_mla_fp8(
    torch.float8_e4m3fn, 16, 512, 64, 576
)


_DEVICE = "cuda"
_DV, _DR, _TOPK, _N = 512, 64, 2048, 16384
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


def _keys_reference(q, keys, idx, sm_scale):
    """fp32 (out, natural-log lse) over each row's keys; -1 selects nothing, and a
    row with no key gives 0 and -inf."""
    gathered = keys[idx.clamp(min=0).long()]
    scores = torch.einsum("thd,tkd->thk", q, gathered) * sm_scale
    scores = scores.masked_fill((idx < 0)[:, None, :], float("-inf"))
    lse = torch.logsumexp(scores, -1)
    out = torch.einsum(
        "thk,tkd->thd", torch.softmax(scores, -1).nan_to_num(0.0), gathered[..., :512]
    )
    return out, lse


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


def _dcp_decode_table(dcp_size, topk, num_slots):
    """Rows covering a single owner, all -1, one slot, a -1-interleaved row,
    and full rows, so some ranks see empty rows."""
    rows = torch.full((6, topk), -1, dtype=torch.int32)
    owner_only = torch.randperm(num_slots // dcp_size)[:300] * dcp_size
    rows[0, : owner_only.numel()] = owner_only.to(torch.int32)
    rows[2] = torch.randperm(num_slots)[:topk].to(torch.int32)
    rows[3, 0] = 5
    rows[4, ::2][:200] = torch.randperm(num_slots)[:200].to(torch.int32)
    rows[5] = torch.randperm(num_slots)[:topk].to(torch.int32)
    return rows.cuda()


def _sparse_mla_reference(q, kv, indices, sm_scale, d_v):
    out = torch.zeros(*q.shape[:2], d_v, device=q.device)
    lse = torch.full(q.shape[:2], float("-inf"), device=q.device)
    for row in range(q.shape[0]):
        slots = indices[row][indices[row] >= 0].long()
        if slots.numel() == 0:
            continue
        keys = kv[slots, 0].float()
        scores = q[row].float() @ keys.T * sm_scale
        out[row] = torch.softmax(scores, dim=-1) @ keys[:, :d_v]
        lse[row] = torch.logsumexp(scores, dim=-1) / math.log(2)
    return out, lse


@pytest.mark.parametrize("kv_splits", [1, 4])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("fp8", [False, True])
def test_dcp_owned_count_decode_merges_to_full_kv(fp8, heads, kv_splits):
    """Replicated-index-K DCP decode: each rank's partial over its owned slots
    merges base 2 into full-KV attention, empty ranks included."""
    torch.manual_seed(heads + kv_splits)
    dcp_size, topk, num_slots, d_v, d_tail = 4, 512, 4096, 512, 64
    sm_scale = (d_v + d_tail) ** -0.5
    kv_dtype = (
        (torch.float8_e4m3fnuz if _IS_FNUZ else torch.float8_e4m3fn)
        if fp8
        else torch.bfloat16
    )
    kv = torch.randn(num_slots, 1, d_v + d_tail, device="cuda").clamp_(-2, 2)
    kv = kv.to(kv_dtype)
    q = torch.randn(6, heads, d_v + d_tail, device="cuda", dtype=torch.bfloat16)
    indices = _dcp_decode_table(dcp_size, topk, num_slots)

    outs, lses = [], []
    for rank in range(dcp_size):
        local_indices, owned_counts = transform_index_page_table_dcp(
            indices, dcp_size, rank
        )
        out, lse = triton_sparse_mla_decode_splitk(
            q[..., :d_v],
            q[..., d_v:],
            kv[rank::dcp_size].contiguous(),
            local_indices.unsqueeze(1),
            sm_scale,
            d_v,
            kv_splits=kv_splits,
            return_lse=True,
            topk_length=owned_counts,
        )
        lse = lse_base2_for_dcp_merge(lse)
        assert torch.isfinite(lse).all()
        outs.append(out[0].float())
        lses.append(lse)

    lse_all = torch.stack(lses)
    lse_max = lse_all.max(dim=0).values
    weights = torch.exp2(lse_all - lse_max)
    merged = (weights.unsqueeze(-1) * torch.stack(outs)).sum(0) / weights.sum(
        0
    ).unsqueeze(-1)
    merged_lse = lse_max + torch.log2(weights.sum(0))

    q_ref = q.to(kv_dtype) if fp8 else q
    expected, expected_lse = _sparse_mla_reference(q_ref, kv, indices, sm_scale, d_v)
    atol = 3e-2 if fp8 else 1e-2
    torch.testing.assert_close(merged, expected, rtol=0, atol=atol)
    non_empty = (indices >= 0).any(dim=-1)
    torch.testing.assert_close(
        merged_lse[non_empty], expected_lse[non_empty], rtol=0, atol=atol
    )


def _scaled_fp8_ds_mla_kv(pool, seed):
    """[pool, 1, 656] fp8_ds_mla rows from the cache writer Kimi-K3.1 uses, and the
    fp32 keys they hold (fp8 NoPE x per-128 fp32 scale, bf16 RoPE)."""
    torch.manual_seed(seed)
    k = (torch.randn(pool, 1, 576, device="cuda") * 2).bfloat16()
    kv = quantize_k_cache(k.unsqueeze(1), scale_ue8m0=True).view(pool, 1, -1)
    raw = kv.view(torch.uint8)[:, 0]
    nope = raw[:, :512].view(torch.float8_e4m3fn).float()
    scales = raw[:, 512:528].contiguous().view(torch.float32)
    rope = raw[:, 528:].contiguous().view(torch.bfloat16).float()
    keys = torch.cat([nope * scales.repeat_interleave(128, -1), rope], -1)
    return kv.view(torch.uint8), keys


def _scaled_kv_reference(q_nope, q_rope, keys, idx, sm_scale):
    q = torch.cat([q_nope.float(), q_rope.float()], -1)
    rows = keys[idx.clamp(min=0).long()]
    s = torch.einsum("thd,tkd->thk", q, rows) * sm_scale
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    return torch.einsum("thk,tkd->thd", torch.softmax(s, -1), rows[..., :512])


def _scaled_kv_case(seq, heads, topk, pool, seed):
    kv, keys = _scaled_fp8_ds_mla_kv(pool, seed)
    q_nope = torch.randn(seq, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    q_rope = torch.randn(seq, heads, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(0, pool, (seq, topk), device="cuda", dtype=torch.int32)
    idx[: max(1, seq // 4), -300:] = -1
    return kv, keys, q_nope, q_rope, idx


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("seq", [6, 1024])  # split-K, and single-pass past the CUs
def test_scaled_kv_prefill_matches_reference(seq, kv_dtype):
    """Kimi-K3.1's fp8_ds_mla rows (TP8: 12 heads) are dequantized before BF16 dots."""
    kv, keys, q_nope, q_rope, idx = _scaled_kv_case(seq, 12, 2048, 8192, seed=seq)
    sm_scale = 576**-0.5

    out = triton_sparse_mla_fwd(
        q_nope, q_rope, kv.view(kv_dtype), idx.unsqueeze(1), sm_scale
    )

    ref = _scaled_kv_reference(q_nope, q_rope, keys, idx, sm_scale)
    assert _rel_err(out[0], ref) < 4e-3


@pytest.mark.parametrize("kv_splits", [None, 1])
@pytest.mark.parametrize("bs", [1, 8, 48])
def test_scaled_kv_decode_matches_reference(bs, kv_splits):
    kv, keys, q_nope, q_rope, idx = _scaled_kv_case(bs, 12, 2048, 8192, seed=bs)
    sm_scale = 576**-0.5

    out = triton_sparse_mla_decode_splitk(
        q_nope, q_rope, kv, idx.unsqueeze(1), sm_scale, kv_splits=kv_splits
    )

    ref = _scaled_kv_reference(q_nope, q_rope, keys, idx, sm_scale)
    assert _rel_err(out[0], ref) < 4e-3


def _decode_reference(qn, qr, idx, kd, sm, mxfp8_q=True):
    """FP32 attention using the production MXFP8 Q quantizer and dequantized KV."""
    R, H, _ = qn.shape
    if not mxfp8_q:
        return _keys_reference(torch.cat([qn.float(), qr.float()], -1), kd, idx, sm)
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
    return _keys_reference(qd, kd, idx, sm)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
def test_dcp_partial_reads_the_indexer_local_slots():
    """Sharded index-K hands the backend this rank's compact slots and their counts
    (dcp_local_topk); localizing them again reads other requests' KV. DCP8 decode
    attends with the group's 96 gathered heads on the gfx950 split-K kernel."""
    kv, kd = _scaled_fp8_ds_mla_kv(_N, seed=0)
    kv = kv.view(torch.float8_e4m3fn)
    lens = [256, 0, 1, 255]  # this rank's share of a 2048 top-k under DCP8
    R, H = len(lens) + 2, 96  # two graph-padding rows select nothing
    qkv = torch.randn(R, H, _DV + _DR, device=_DEVICE).bfloat16()
    qn, qr = qkv[..., :_DV], qkv[..., _DV:]
    idx = torch.full((R, _TOPK), -1, dtype=torch.int32, device=_DEVICE)
    for r, n in enumerate(lens):
        idx[r, :n] = torch.randperm(_N, device=_DEVICE)[:n].int()
    lengths = torch.tensor(lens, dtype=torch.int32, device=_DEVICE)
    sm = (_DV + _DR) ** -0.5
    backend = SimpleNamespace(
        _triton_sparse_mla_workspaces={},
        attn_dcp_size=8,
        forward_metadata=SimpleNamespace(dcp_topk_length=lengths),
    )
    layer = SimpleNamespace(scaling=sm, v_head_dim=_DV)
    o, lse = dsa_backend_hip.forward_triton_dcp_partial(backend, qn, qr, kv, idx, layer)
    # the slots past each row's count hold -1, so the reference reads only its keys
    o_ref, lse_ref = _decode_reference(
        qn, qr, idx, kd, sm, mxfp8_q=dsa_backend_hip._gfx950_decode_enabled()
    )
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    torch.testing.assert_close(lse[finite], lse_ref[finite], rtol=0, atol=1e-3)
    torch.testing.assert_close(o.float(), o_ref, rtol=0, atol=3e-2)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 only")
def test_sparse_mla_prefill_gfx950():
    """Kimi-K3.1's TP8 shard (12 heads, head-major q) with large scores, a top-k that
    ends inside a key tile, -1 padding and a row with no key."""
    T, H, topk = 96, 12, 1000
    kv, k = _scaled_fp8_ds_mla_kv(_PREFILL_POOL, seed=H + topk)
    torch.manual_seed(topk + H)
    q_nope = torch.randn(T, H, _DV, device=_DEVICE, dtype=torch.bfloat16) * 3.0
    q_rope = torch.randn(T, H, _DR, device=_DEVICE, dtype=torch.bfloat16) * 3.0
    # [H, T, D] storage viewed as [T, H, D]
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
    ref_out, ref_lse = _keys_reference(
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
