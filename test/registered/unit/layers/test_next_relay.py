# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("bs", [0, 1, 129])
@pytest.mark.parametrize("width", [1, 17, 2560])
@pytest.mark.parametrize("strided", [False, True])
def test_relay_scatter_graph(bs, width, strided):
    """The strided arm gives the kernel an index tensor with stride 2 and
    sources whose row stride is wider than their width (a column slice of a
    wider buffer); the contiguous arm gives it dense rows and indices."""
    from sglang.kernels.ops.speculative.gather_spec_extras import scatter_spec_extras

    gen = torch.Generator(device="cuda").manual_seed(bs * 4099 + width)
    slots = bs + 3
    indices = torch.randperm(slots, device="cuda", generator=gen)[:bs]
    if strided:
        indices = indices.repeat_interleave(2)[::2]
        assert indices.stride(0) == 2
    else:
        assert indices.stride(0) == 1
    if bs:
        indices[0] -= slots
    pad = 5 if strided else 0
    pairs = []
    for n, src_type, dst_type in [
        (1, torch.int32, torch.int64),
        (3, torch.float32, torch.float32),
        (width, torch.float32, torch.bfloat16),
        (7, torch.int64, torch.int64),
    ]:
        src = torch.arange(bs * (n + pad), device="cuda", dtype=src_type).reshape(
            bs, n + pad
        )[:, :n]
        assert src.stride() == (n + pad, 1)
        if strided and bs > 1:
            # the row stride reaches the kernel: the op's reshape keeps the view
            assert not src.is_contiguous()
            assert src.reshape(bs, -1).stride() == (n + pad, 1)
        dst = torch.full((slots, n), -7, dtype=dst_type, device="cuda")
        pairs.append((dst, src))
    scatter_spec_extras(indices, pairs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_spec_extras(indices, pairs)
    for replay in range(3):
        expected = []
        for dst, src in pairs:
            dst.fill_(-7)
            src.add_(replay)
            ref = dst.clone()
            ref[indices] = src.to(dst.dtype)
            expected.append(ref)
        graph.replay()
        for (actual, _), ref in zip(pairs, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
