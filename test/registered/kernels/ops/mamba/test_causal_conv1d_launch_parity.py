"""causal_conv1d forward with a configured launch tile is bit-identical to the
default tile: each output is computed by one thread from the same inputs.

Outputs come from an allocator block poisoned with NaN, the test records the
launch configuration, and a seeded positive control must change the output.
"""

import pytest
import torch

from sglang.kernels.ops.mamba import causal_conv1d_triton as conv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
        reason="needs an SM100+ GPU",
    ),
]

TILES = [
    {"BLOCK_M": 16, "BLOCK_N": 256, "num_warps": 1, "num_stages": 3},
    {"BLOCK_M": 32, "BLOCK_N": 1024, "num_warps": 4, "num_stages": 2},
]


def _batch(seed, dim=4096, width=4, seqs=5):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seq_lens = torch.randint(1, 3000, (seqs,), generator=gen).tolist()
    total = sum(seq_lens)
    # Channel-last x: (dim, tokens) with stride(0) == 1.
    x = torch.randn(total, dim, generator=gen).to(torch.bfloat16).cuda().t()
    weight = torch.randn(dim, width, generator=gen).to(torch.bfloat16).cuda()
    bias = torch.randn(dim, generator=gen).to(torch.bfloat16).cuda()
    states = torch.randn(seqs, width - 1, dim, generator=gen).to(torch.bfloat16)
    conv_states = states.cuda().transpose(1, 2)
    query_start_loc = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                                   dtype=torch.int32).cuda()
    cache_indices = torch.arange(seqs, dtype=torch.int32).cuda()
    has_initial_state = torch.tensor([i % 2 == 0 for i in range(seqs)]).cuda()
    return dict(
        x=x, weight=weight, bias=bias, conv_states=conv_states,
        query_start_loc=query_start_loc, seq_lens_cpu=seq_lens,
        cache_indices=cache_indices, has_initial_state=has_initial_state,
    )


def _run(batch, tile, monkeypatch):
    monkeypatch.setattr(conv, "_FWD_LAUNCH_CONFIG", tile)
    args = dict(batch, conv_states=batch["conv_states"].clone())
    poison = torch.full_like(batch["x"], float("nan"))
    del poison
    out = conv.causal_conv1d_fn(**args, activation="silu")
    torch.cuda.synchronize()
    return out, args["conv_states"]


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("seed", [0, 1])
def test_launch_tile_is_bitwise(seed, tile, monkeypatch, record_property):
    batch = _batch(seed)
    record_property("launch", tile)
    stock_out, stock_states = _run(batch, None, monkeypatch)
    tiled_out, tiled_states = _run(batch, tile, monkeypatch)
    assert conv._fwd_launch_config(None, batch["weight"].shape[1]) == tile
    assert not torch.isnan(tiled_out).any(), "an output was never written"
    assert torch.equal(tiled_out.view(torch.int16), stock_out.view(torch.int16))
    assert torch.equal(tiled_states.view(torch.int16), stock_states.view(torch.int16))

    perturbed = dict(batch, x=batch["x"].clone())
    perturbed["x"][0, 0] += 1.0
    control, _ = _run(perturbed, tile, monkeypatch)
    assert not torch.equal(control.view(torch.int16), tiled_out.view(torch.int16))
