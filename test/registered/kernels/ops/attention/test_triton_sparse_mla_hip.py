"""ROCm tests for Triton sparse MLA launch tuning."""

import math

import pytest
import torch

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
    triton_sparse_mla_decode_splitk,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")

_IS_GFX950 = is_hip() and _is_gfx950_sparse_mla_fp8(
    torch.float8_e4m3fn, 16, 512, 64, 576
)


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
            lengths=owned_counts,
        )
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
