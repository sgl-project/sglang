"""TMA load-back of page_unified host pages, including grouped KV heads."""

import contextlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.kvcache import hicache
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 9,
    reason="HiCache TMA load-back requires SM90+",
)


@pytest.fixture(autouse=True)
def uncached_tma_gate(monkeypatch):
    # Production caches this gate for a process. These tests mock the TMA module
    # and switch the gate in one process to exercise both transfer paths.
    monkeypatch.setattr(
        hicache, "use_hicache_tma_kernel", hicache.use_hicache_tma_kernel.__wrapped__
    )


def _source(shape, dtype, device):
    size = 1
    for dim in shape:
        size *= dim
    # A nonzero aligned storage offset catches assumptions about the base
    # allocation; the CPU version is read through its pinned device mapping.
    storage = torch.empty(size + 32, dtype=dtype, device="cpu", pin_memory=True)
    values = torch.randint(
        0, 127, (size + 32,), generator=torch.Generator().manual_seed(31)
    )
    storage.copy_(values)
    if device == "cuda":
        storage = storage.cuda()
    return storage[16 : 16 + size].view(shape)


def _indices(page_size, num_pages, index_dtype, pattern):
    gen = torch.Generator().manual_seed(23)
    if pattern == "pages":
        src_pages = torch.randperm(num_pages, generator=gen)[:5]
        dst_pages = torch.randperm(num_pages, generator=gen)[:5]
        offsets = torch.arange(page_size)
        src = (src_pages[:, None] * page_size + offsets).flatten()
        dst = (dst_pages[:, None] * page_size + offsets).flatten()
    elif pattern == "crossing":
        # Consecutive token ids need not identify contiguous source bytes:
        # each unaligned run crosses page boundaries with intervening layers
        # and head groups. This must use per-row loads despite run == true.
        src = torch.arange(1, 258)
        dst = torch.arange(3, 260)
    else:
        count = min(257, num_pages * page_size - 3)
        src = torch.randperm(num_pages * page_size, generator=gen)[:count]
        dst = torch.randperm(num_pages * page_size, generator=gen)[:count]
        # Explicit boundary crossings, a repeated source, and an odd partial
        # chunk accompany the random scatter. Destinations remain unique.
        src[:5] = torch.tensor(
            [page_size - 1, page_size, 2 * page_size - 1, 2 * page_size, page_size]
        )
    return src.to(device="cuda", dtype=index_dtype), dst.to(
        device="cuda", dtype=index_dtype
    )


def _expected(src, dst, src_indices, dst_indices, layer, *, is_mla, component=0):
    result = dst.cpu().clone()
    source = src.cpu()
    src_ids, dst_ids = src_indices.cpu().long(), dst_indices.cpu().long()
    if is_mla:
        page_size = src.shape[2]
        result[dst_ids, 0] = source[src_ids // page_size, layer, src_ids % page_size]
    else:
        page_size, heads_per_group = src.shape[4:6]
        for group in range(src.shape[1]):
            result[dst_ids, group * heads_per_group : (group + 1) * heads_per_group] = (
                source[
                    src_ids // page_size, group, layer, component, src_ids % page_size
                ]
            )
    return result


def _launch(k, v, src, src_indices, dst_indices, layer, *, is_mla):
    if is_mla:
        hicache.transfer_hicache_one_layer_mla_page_unified_lf(
            k, src, src_indices, dst_indices, layer, block_quota=2
        )
    else:
        hicache.transfer_hicache_one_layer_page_unified_lf(
            k, v, src, src_indices, dst_indices, layer, block_quota=2
        )


def _tma_proxy():
    module = hicache._jit_hicache_tma_module(block_quota=2)
    if not module.fits_device(torch.cuda.current_device()):
        pytest.skip("device has insufficient shared memory for HiCache TMA stages")
    return SimpleNamespace(
        fits_device=module.fits_device,
        launch_one_page_unified=mock.Mock(wraps=module.launch_one_page_unified),
        launch_one_mla_page_unified=mock.Mock(wraps=module.launch_one_mla_page_unified),
    )


@pytest.mark.parametrize(
    "is_mla,page_size,layers,groups,heads_per_group,dim,dtype,index_dtype,source_device,pattern",
    [
        (False, 128, 8, 1, 4, 128, torch.bfloat16, torch.int64, "cpu", "pages"),
        (False, 16, 5, 2, 2, 128, torch.bfloat16, torch.int32, "cpu", "scatter"),
        (False, 1, 5, 4, 1, 64, torch.bfloat16, torch.int64, "cuda", "scatter"),
        (False, 3, 5, 3, 1, 96, torch.float16, torch.int32, "cpu", "scatter"),
        (False, 64, 3, 2, 1, 16, torch.uint8, torch.int32, "cuda", "pages"),
        (True, 128, 5, 1, 1, 576, torch.bfloat16, torch.int32, "cpu", "pages"),
        (True, 16, 5, 1, 1, 576, torch.uint8, torch.int64, "cpu", "scatter"),
        (True, 1, 3, 1, 1, 576, torch.bfloat16, torch.int32, "cuda", "scatter"),
        (True, 3, 5, 1, 1, 64, torch.float16, torch.int64, "cuda", "scatter"),
        (False, 64, 5, 2, 1, 128, torch.bfloat16, torch.int64, "cpu", "crossing"),
        (True, 64, 5, 1, 1, 576, torch.bfloat16, torch.int32, "cpu", "crossing"),
    ],
)
def test_page_unified_tma_and_register_parity(
    is_mla,
    page_size,
    layers,
    groups,
    heads_per_group,
    dim,
    dtype,
    index_dtype,
    source_device,
    pattern,
):
    num_pages = max(11, (260 + page_size - 1) // page_size)
    shape = (
        (num_pages, layers, page_size, dim)
        if is_mla
        else (num_pages, groups, layers, 2, page_size, heads_per_group, dim)
    )
    src = _source(shape, dtype, source_device)
    source_before = src.cpu().clone()
    src_indices, dst_indices = _indices(page_size, num_pages, index_dtype, pattern)
    heads = groups * heads_per_group
    k = torch.full((num_pages * page_size, heads, dim), 17, dtype=dtype, device="cuda")
    v = torch.full_like(k, 29)
    proxy = _tma_proxy()
    selected_layers = (0, layers // 2, layers - 1)
    # The odd layer counts exercise both a nonzero layer offset and the short
    # final layer window of a partitioned page object.
    with mock.patch.object(hicache, "_jit_hicache_tma_module", return_value=proxy):
        for enabled in (True, False):
            # The disabled pass forces the register kernel for the parity check.
            gate = (
                contextlib.nullcontext()
                if enabled
                else mock.patch.object(
                    hicache, "use_hicache_tma_kernel", return_value=False
                )
            )
            with gate:
                for layer in selected_layers:
                    k.fill_(17)
                    v.fill_(29)
                    expected_k = _expected(
                        src, k, src_indices, dst_indices, layer, is_mla=is_mla
                    )
                    if not is_mla:
                        expected_v = _expected(
                            src,
                            v,
                            src_indices,
                            dst_indices,
                            layer,
                            is_mla=False,
                            component=1,
                        )
                    _launch(k, v, src, src_indices, dst_indices, layer, is_mla=is_mla)
                    # Full-buffer comparison catches writes outside the selected
                    # token rows and copies into the wrong head-group columns.
                    torch.testing.assert_close(k.cpu(), expected_k, rtol=0, atol=0)
                    if not is_mla:
                        torch.testing.assert_close(v.cpu(), expected_v, rtol=0, atol=0)
    launch = (
        proxy.launch_one_mla_page_unified if is_mla else proxy.launch_one_page_unified
    )
    assert launch.call_count == len(selected_layers)
    torch.testing.assert_close(src.cpu(), source_before, rtol=0, atol=0)


def test_page_unified_gate_and_oversized_row_fallback():
    assert not hicache._use_page_unified_tma_kernel(group_bytes=0, block_quota=2)
    assert not hicache._use_page_unified_tma_kernel(group_bytes=18, block_quota=2)
    assert not hicache._use_page_unified_tma_kernel(group_bytes=32784, block_quota=2)
    # Oversized rows remain supported by the existing register copy path.
    src = _source((3, 2, 1, 16392), torch.bfloat16, "cpu")
    k = torch.zeros((4, 1, 16392), dtype=torch.bfloat16, device="cuda")
    src_indices = torch.tensor([2, 0], device="cuda")
    dst_indices = torch.tensor([1, 3], device="cuda")
    expected = _expected(src, k, src_indices, dst_indices, 1, is_mla=True)
    with mock.patch.object(
        hicache,
        "_jit_hicache_tma_module",
        side_effect=AssertionError("oversized TMA row"),
    ):
        _launch(k, k, src, src_indices, dst_indices, 1, is_mla=True)
    torch.testing.assert_close(k.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_page_unified_empty_and_cuda_graph(is_mla):
    page_size, layers, dim, num_pages = 16, 3, 128, 40
    shape = (
        (num_pages, layers, page_size, dim)
        if is_mla
        else (num_pages, 2, layers, 2, page_size, 1, dim)
    )
    src = _source(shape, torch.bfloat16, "cpu")
    heads = 1 if is_mla else 2
    k = torch.full(
        (num_pages * page_size, heads, dim), 17, dtype=torch.bfloat16, device="cuda"
    )
    v = torch.full_like(k, 29)
    empty = torch.empty(0, dtype=torch.int64, device="cuda")
    src_indices, dst_indices = _indices(page_size, num_pages, torch.int64, "scatter")
    _launch(k, v, src, empty, empty, 2, is_mla=is_mla)
    assert (k == 17).all() and (v == 29).all()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _launch(k, v, src, src_indices, dst_indices, 2, is_mla=is_mla)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(k, v, src, src_indices, dst_indices, 2, is_mla=is_mla)
    for _ in range(3):
        k.fill_(17)
        v.fill_(29)
        graph.replay()
    torch.testing.assert_close(
        k.cpu(),
        _expected(
            src, torch.full_like(k, 17), src_indices, dst_indices, 2, is_mla=is_mla
        ),
        rtol=0,
        atol=0,
    )
    if not is_mla:
        expected_v = _expected(
            src,
            torch.full_like(v, 29),
            src_indices,
            dst_indices,
            2,
            is_mla=False,
            component=1,
        )
        torch.testing.assert_close(v.cpu(), expected_v, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
