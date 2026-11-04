"""ROCm tests for Triton sparse MLA launch tuning."""

import pytest
import torch
import triton

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
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=35, stage="jit-kernel-unit", runner_config="amd")
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
    rows = keys[idx.clamp(min=0).long()]
    s = torch.einsum("thd,tkd->thk", q, rows) * sm_scale
    s = s.masked_fill((idx < 0)[:, None, :], float("-inf"))
    ref = torch.einsum("thk,tkd->thd", torch.softmax(s, -1), rows[..., :512])
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
