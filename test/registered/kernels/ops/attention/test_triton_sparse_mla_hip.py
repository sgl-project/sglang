"""ROCm tests for Triton sparse MLA launch tuning."""

import pytest
import torch

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
