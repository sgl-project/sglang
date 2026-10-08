"""Byte-exact dual stores after the stock PyTorch raw-FP8 conversion."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
    can_use_set_sharded_mla_kv_buffer,
    set_mla_kv_buffer,
    set_sharded_mla_kv_buffer,
    sharded_mla_kv_buffer_inputs_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HAS_TMA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] >= 9
)
pytestmark = pytest.mark.skipif(not _HAS_TMA, reason="requires NVIDIA SM90+ TMA")


def _converted_rows(n, dtype):
    # The value pattern covers conversion boundaries and deliberately exceeds
    # finite E4M3 range. Do not replace this with a saturating kernel conversion.
    source = torch.randn((n, 1, 576), device="cuda", dtype=dtype)
    special = torch.tensor(
        [
            0.0,
            -0.0,
            1e-9,
            -1e-9,
            448,
            -448,
            449,
            -449,
            464,
            -464,
            500,
            -500,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=dtype,
    )
    if n:
        source[:, 0, : special.numel()] = special
        source[:, 0, 512 : 512 + special.numel()] = special
    nope = source[..., :512].to(torch.float8_e4m3fn).view(torch.uint8)
    rope = source[..., 512:].to(torch.float8_e4m3fn).view(torch.uint8)
    return nope, rope


def _reference(scratch, rows, local, loc, nope, rope, cp, rank, skips=(0, 0)):
    # Oracle uses logical owner arithmetic independently of the fused kernel.
    payload = torch.cat((nope, rope), dim=-1).view(loc.numel(), 576)
    keep = rows != skips[0]
    scratch.view(scratch.shape[0], -1)[rows[keep].long(), :576] = payload[keep]
    owned = (loc // 64) % cp == rank
    local_rows = (loc // (64 * cp)) * 64 + loc % 64
    owned &= local_rows != skips[1]
    local.view(local.shape[0], -1)[local_rows[owned].long(), :576] = payload[owned]


@pytest.mark.parametrize(
    "cp,rank", [(cp, rank) for cp in (2, 4, 8) for rank in range(cp)]
)
@pytest.mark.parametrize("n", [0, 1, 63, 64, 768, 769, 1023, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_raw_fp8_dual_store(cp, rank, n, dtype, loc_dtype):
    torch.manual_seed(314 + n + rank)
    assert can_use_set_sharded_mla_kv_buffer(512, 64)
    pages = max(1, (n + 63) // 64)
    # Logical pages are fragmented and out of order; no rank is special.
    page_order = torch.randperm(pages, device="cuda") * 3 + cp
    loc = (page_order[:, None] * 64 + torch.arange(64, device="cuda")).flatten()[:n]
    loc = loc.to(loc_dtype)
    rows = (torch.randperm(n, device="cuda") + 64).to(loc_dtype)
    nope, rope = _converted_rows(n, dtype)
    # Different row strides and guard bytes catch accidentally sharing the
    # scratch destination stride with the owner-local destination.
    scratch_storage = torch.full((n + 128, 608), 0xA5, dtype=torch.uint8, device="cuda")
    local_storage = torch.full(
        (((pages * 3 + cp) // cp + 2) * 64, 624),
        0x5A,
        dtype=torch.uint8,
        device="cuda",
    )
    scratch = scratch_storage[:, :576].unsqueeze(1)
    local = local_storage[:, :576].unsqueeze(1)
    expected_scratch, expected_local = scratch_storage.clone(), local_storage.clone()
    _reference(expected_scratch, rows, expected_local, loc, nope, rope, cp, rank)
    args = (scratch, rows, local, loc, nope, rope, 64, cp, rank)
    assert sharded_mla_kv_buffer_inputs_supported(*args)
    set_sharded_mla_kv_buffer(*args)
    torch.cuda.synchronize()
    assert torch.equal(scratch_storage, expected_scratch)
    assert torch.equal(local_storage, expected_local)


@pytest.mark.parametrize("skips", [(0, 0), (-1, 0), (0, -1), (-1, -1), (19, 64)])
@pytest.mark.parametrize("rank", range(4))
def test_independent_reserved_slots(skips, rank):
    # Slot zero is writable on either side when that side disables skipping.
    loc = torch.tensor(
        [rank * 64, rank * 64 + 1, (rank + 4) * 64, ((rank + 1) % 4) * 64 + 3],
        device="cuda",
    )
    rows = torch.tensor([0, 1, 19, 20], device="cuda")
    nope, rope = _converted_rows(4, torch.bfloat16)
    scratch = torch.full((128, 576), 0xA5, device="cuda", dtype=torch.uint8)
    local = torch.full((128, 576), 0x5A, device="cuda", dtype=torch.uint8)
    expected_scratch, expected_local = scratch.clone(), local.clone()
    _reference(expected_scratch, rows, expected_local, loc, nope, rope, 4, rank, skips)
    set_sharded_mla_kv_buffer(
        scratch,
        rows,
        local,
        loc,
        nope,
        rope,
        64,
        4,
        rank,
        scratch_reserved_skip_index=skips[0],
        local_reserved_skip_index=skips[1],
    )
    torch.cuda.synchronize()
    assert torch.equal(scratch, expected_scratch)
    assert torch.equal(local, expected_local)


def test_graph_replay_reads_current_rows_and_locations():
    loc = torch.arange(128, device="cuda", dtype=torch.int64) + 256
    rows = torch.arange(128, device="cuda", dtype=torch.int64) + 64
    nope, rope = _converted_rows(128, torch.bfloat16)
    scratch = torch.full((256, 576), 0xA5, device="cuda", dtype=torch.uint8)
    local = torch.full((256, 576), 0x5A, device="cuda", dtype=torch.uint8)
    args = (scratch, rows, local, loc, nope, rope, 64, 4, 0)
    # Compile outside capture. Capture consumes pointer values, not frozen loc
    # contents, and supports batches padded with repeated reserved zeros.
    assert can_use_set_sharded_mla_kv_buffer(512, 64)
    set_sharded_mla_kv_buffer(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        set_sharded_mla_kv_buffer(*args)
    for padding in (0, 32, 128, 0):
        scratch.fill_(0xA5)
        local.fill_(0x5A)
        loc.copy_(torch.arange(128, device="cuda") + 256)
        rows.copy_(torch.arange(128, device="cuda") + 64)
        if padding:
            loc[-padding:] = 0
            rows[-padding:] = 0
        nope.random_(0, 256)
        rope.random_(0, 256)
        expected_scratch, expected_local = scratch.clone(), local.clone()
        _reference(expected_scratch, rows, expected_local, loc, nope, rope, 4, 0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(scratch, expected_scratch)
        assert torch.equal(local, expected_local)


def test_existing_single_writer_preserves_reserved_index():
    nope, rope = _converted_rows(3, torch.bfloat16)
    rows = torch.tensor([0, 1, 0], device="cuda")
    scratch = torch.full((8, 576), 0xA5, device="cuda", dtype=torch.uint8)
    reserved = scratch[0].clone()
    set_mla_kv_buffer(scratch, rows, nope, rope)
    torch.cuda.synchronize()
    assert torch.equal(scratch[0], reserved)
    assert torch.equal(scratch[1], torch.cat((nope[1, 0], rope[1, 0])))
    set_mla_kv_buffer(
        scratch,
        rows[:1],
        nope[:1],
        rope[:1],
        reserved_skip_index=-1,
    )
    torch.cuda.synchronize()
    assert torch.equal(scratch[0], torch.cat((nope[0, 0], rope[0, 0])))


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *args]))
