import sys

import pytest
import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import _fused_norm_rope_stacked
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prefix_write_matches_materialize_then_scatter(dtype):
    # Each slot also contains host rows, other draft lanes, and alignment padding.
    layers, heads, dim, rotary_dim = 3, 2, 64, 32
    batch, width, slots = 4, 5, 31
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(42)
    args = (
        torch.randn(
            batch * width,
            layers,
            2 * heads * dim,
            dtype=dtype,
            device=device,
            generator=gen,
        ),
        torch.randn(layers, dim, dtype=dtype, device=device, generator=gen),
        torch.full((layers,), 1e-6, device=device),
        torch.randn(97, rotary_dim, dtype=dtype, device=device, generator=gen),
        torch.randperm(97, device=device, generator=gen)[: batch * width],
        heads,
        dim,
        rotary_dim,
    )
    k, v = _fused_norm_rope_stacked(*args)
    raw = torch.full(
        (slots, layers + 2, 2, heads, dim + 8), 42, dtype=dtype, device=device
    )
    expected = raw.clone()
    locs = (
        torch.randperm(slots, dtype=torch.int32, device=device, generator=gen)[
            : batch * width
        ]
        .view(width, batch)
        .T
    )
    lengths = torch.tensor([0, 1, 3, width], device=device, dtype=torch.int32)
    valid = torch.arange(width, device=device)[None, :] < lengths[:, None]
    dst = locs[valid].long()
    for idx, src in enumerate((k, v)):
        expected[dst, 1 : 1 + layers, idx, :, :dim] = src[:, valid.flatten()].transpose(
            0, 1
        )
    buffers = [raw[:, 1 : 1 + layers, idx, :, :dim].transpose(0, 1) for idx in (0, 1)]
    _fused_norm_rope_stacked(
        *args, k_out=buffers[0], v_out=buffers[1], cache_locs=locs, commit_lens=lengths
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
