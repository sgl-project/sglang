"""ROCm tests for Triton sparse MLA launch tuning."""

import pytest
import torch
from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _cu_count,
    _cu_count_for_device,
    _gfx950_sparse_mla_kv_splits,
    _gfx950_sparse_mla_num_warps,
    _page_offsets_fit_i32,
    _reduce_d_chunk,
    triton_sparse_mla_fwd,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    _get_splitk_bufs,
    _gfx950_sparse_mla_decode_tile_config,
    _sparse_mla_decode_reduce_kernel,
    _splitk_bufs,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm only")


def test_cu_count_uses_tensor_device(monkeypatch):
    requested = []

    class Properties:
        multi_processor_count = 123

    def get_device_properties(device):
        requested.append(device)
        return Properties()

    _cu_count_for_device.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("current_device must not be used"),
    )
    assert _cu_count(torch.device("cuda:7")) == 123
    assert requested == [7]
    _cu_count_for_device.cache_clear()


@pytest.mark.parametrize(
    ("active_splits", "output_rows", "expected"),
    [(2, 0, 128), (4, 1535, 128), (4, 1536, 512), (8, 2048, 64)],
)
def test_reduce_d_chunk(active_splits, output_rows, expected):
    assert _reduce_d_chunk(active_splits, output_rows) == expected


@pytest.mark.parametrize(
    ("base_ctas", "topk", "initial", "expected"),
    [
        (4, 2048, 32, 32),
        (5, 2048, 32, 16),
        (65, 2048, 2, 4),
        (256, 2048, 1, 2),
        (257, 2048, 1, 1),
        (160, 1024, 1, 1),
    ],
)
def test_gfx950_kv_splits(base_ctas, topk, initial, expected):
    assert (
        _gfx950_sparse_mla_kv_splits(
            base_ctas,
            topk,
            block_k=64,
            num_cu=256,
            kv_splits=initial,
            max_kv_splits=max(1, topk // 64),
        )
        == expected
    )


def test_gfx950_launch_config():
    assert _gfx950_sparse_mla_num_warps(64, 4, 256) == 4
    assert _gfx950_sparse_mla_num_warps(65, 4, 256) == 2
    assert _gfx950_sparse_mla_decode_tile_config(1, 2048, 64, 32) == (32, 64)
    assert _gfx950_sparse_mla_decode_tile_config(3, 2048, 64, 32) == (64, 32)
    max_pages = ((1 << 31) - 1) // 576
    assert _page_offsets_fit_i32(max_pages, 576)
    assert not _page_offsets_fit_i32(max_pages + 1, 576)


def test_splitk_workspaces_are_graph_and_stream_safe():
    workspace = []
    device = torch.device("cuda")
    _get_splitk_bufs(1, 1, 16, 4, device, workspace)
    first = workspace[0]
    _get_splitk_bufs(1024, 1, 16, 4, device, workspace)
    assert len(workspace) == 2
    assert workspace[0][0] is first[0]
    assert workspace[0][1] is first[1]

    _splitk_bufs.clear()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pointers = []
    for stream in streams:
        with torch.cuda.stream(stream):
            lse, _ = _get_splitk_bufs(1, 1, 16, 4, device)
            pointers.append(lse.data_ptr())
    assert pointers[0] != pointers[1]
    _splitk_bufs.clear()


@pytest.mark.parametrize("topk", [2048, 2050])
def test_short_prefill_index_bound_matches_explicit_bound(topk):
    torch.manual_seed(29)
    seq, heads, value_dim, tail_dim = 4, 16, 512, 64
    q = torch.randn(
        seq, heads, value_dim + tail_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = (
        torch.randn(256, 1, value_dim + tail_dim, device="cuda")
        .clamp_(-2, 2)
        .to(torch.float8_e4m3fn)
    )
    indices = torch.full((seq, 1, topk), -1, device="cuda", dtype=torch.int32)
    for row in range(seq):
        positions = torch.randperm(128, device="cuda")[: row + 1]
        indices[row, 0, positions] = torch.randint(
            0, kv.shape[0], (row + 1,), device="cuda", dtype=torch.int32
        )

    ramp = torch.arange(1, topk + 1, device="cuda", dtype=torch.int32)
    topk_length = torch.where(indices.squeeze(1) >= 0, ramp, 0).amax(dim=1)
    args = (
        q[:, :, :value_dim],
        q[:, :, value_dim:],
        kv,
        indices,
        value_dim**-0.5,
        value_dim,
    )
    expected = triton_sparse_mla_fwd(
        *args, topk_length=topk_length, max_topk_length=seq
    )
    actual = triton_sparse_mla_fwd(*args, max_topk_length=seq)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("active_splits", [2, 4])
def test_reduce_d_chunk_is_bitwise_identical(active_splits):
    torch.manual_seed(active_splits)
    batch, heads, value_dim = 7, 16, 512
    lse = torch.randn(batch, active_splits, heads, device="cuda")
    acc = torch.randn(
        batch, active_splits, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )

    def run(d_chunk):
        out = torch.empty(batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
        _sparse_mla_decode_reduce_kernel[(batch, heads, value_dim // d_chunk)](
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
        run(_reduce_d_chunk(active_splits)), run(64), rtol=0, atol=0
    )
