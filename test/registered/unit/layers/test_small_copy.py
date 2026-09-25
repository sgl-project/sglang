from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import sys

import pytest
import torch

from sglang.kernels.ops.memory.small_copy import _small_copy_kernel, try_small_copy


@pytest.mark.parametrize("n", [0, 1, 8, 128, 4096])
@pytest.mark.parametrize("strided", [False, True])
def test_bits(n, strided):
    dsts = []
    srcs = []
    backings = []
    for dtype in [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]:
        source = torch.randint(
            0,
            256,
            (2 * n * torch.empty((), dtype=dtype).element_size(),),
            device="cuda",
            dtype=torch.uint8,
        ).view(dtype)
        src = source[::2] if strided else source[:n]
        backing = torch.zeros(n * 3, device="cuda", dtype=dtype)
        dst = backing[::3] if strided else backing[:n]
        dsts.append(dst)
        srcs.append(src)
        backings.append(backing)
    expected = [s.clone() for s in srcs]
    assert try_small_copy(dsts, srcs)
    for d, s in zip(dsts, expected):
        assert torch.equal(
            torch.empty(d.shape, device=d.device, dtype=d.dtype)
            .copy_(d)
            .view(torch.uint8),
            torch.empty(s.shape, device=s.device, dtype=s.dtype)
            .copy_(s)
            .view(torch.uint8),
        )


@pytest.mark.parametrize("n", [1, 4, 128, 2048])
def test_cast_and_mrope_graph(n):
    s0 = torch.randint(-(2**40), 2**40, (n * 2,), device="cuda", dtype=torch.int64)[::2]
    s1 = torch.arange(n * 3, device="cuda", dtype=torch.int32).reshape(3, n)
    d0 = torch.empty(n, device="cuda", dtype=torch.int32)
    backing = torch.full((3, n + 5), -7, device="cuda", dtype=torch.int64)
    d1 = backing[:, :n]
    try_small_copy([d0, d1], [s0, s1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert try_small_copy([d0, d1], [s0, s1])
    for step in range(3):
        s0.add_(17)
        s1.add_(13)
        graph.replay()
        assert torch.equal(d0, s0.int())
        assert torch.equal(d1, s1.long())
        assert bool((backing[:, n:] == -7).all())


def test_fallbacks():
    x = torch.arange(17, device="cuda")
    y = torch.empty_like(x)
    assert not try_small_copy([x[1:], y[:16]], [x[:-1], x[:16]])
    assert not try_small_copy([x, y], [y, x])
    assert not try_small_copy([x[:8], x[4:12]], [y[:8], y[:8]])
    assert not try_small_copy(
        [torch.empty(8193, device="cuda")] * 2, [torch.empty(8193, device="cuda")] * 2
    )
    assert not try_small_copy(
        [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
        [torch.empty(2, dtype=torch.complex64, device="cuda")] * 2,
    )
    a = torch.empty(2)
    assert not try_small_copy([a, a], [a, a])
    assert try_small_copy([x, y], [x, y])


def test_broadcast_and_negative_views():
    source = torch.arange(7, device="cuda", dtype=torch.float32)
    expanded = source[None, :].expand(3, 7)
    destinations = [torch.empty_like(expanded), torch.empty_like(source)]
    assert try_small_copy(destinations, [expanded, source])
    assert torch.equal(destinations[0], expanded)
    assert torch.equal(destinations[1], source)
    assert not try_small_copy(destinations, [expanded, source._neg_view()])


def test_dynamic_metadata_has_bounded_specializations():
    kernel_cache = _small_copy_kernel.device_caches[torch.cuda.current_device()][0]
    initial_count = len(kernel_cache)
    specialization_count = None
    for n in [*range(1, 130), 255, 256, 511, 512, 1023, 1024, 2048, 2730]:
        step = 1 + n % 3
        source = torch.arange(n * step, device="cuda", dtype=torch.int64)[::step]
        backing = torch.full((n * step,), -7, device="cuda", dtype=torch.int32)
        destination = backing[::step]
        matrix = torch.arange(3 * (n + 3), device="cuda", dtype=torch.int32).reshape(
            3, n + 3
        )[:, :n]
        matrix_backing = torch.full((3, n + 5), -7, device="cuda", dtype=torch.int64)
        matrix_destination = matrix_backing[:, :n]
        assert try_small_copy([destination, matrix_destination], [source, matrix])
        assert torch.equal(destination, source.int())
        assert torch.equal(matrix_destination, matrix.long())
        assert bool((matrix_backing[:, n:] == -7).all())
        for offset in range(1, step):
            assert bool((backing[offset::step] == -7).all())
        if n == 48:
            specialization_count = len(kernel_cache)
            assert specialization_count - initial_count <= 16
        elif n > 48:
            assert len(kernel_cache) == specialization_count


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
