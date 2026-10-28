import unittest

import torch
import triton

from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import _combine
from sglang.kernels.ops.attention.dsv4.swapab_gluon_hip import combine_gfx950
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


def launch(combine, b, nt, bd, rope_dim, inputs):
    part, mx, den, sink, freqs, positions = inputs
    out = torch.empty((b, 16, 512), dtype=torch.bfloat16, device="cuda")
    combine[(b, 16, 512 // bd)](
        part,
        mx,
        den,
        sink,
        out,
        freqs,
        positions,
        FREQ_STRIDE=freqs.stride(0),
        ROPE_DIM=rope_dim,
        NT=nt,
        ST=triton.next_power_of_2(nt),
        H=16,
        BD=bd,
        num_warps=4,
    )
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(), "requires gfx950"
)
class TestSwapABCombineRopeTiles(CustomTestCase):
    def check(self, b, nt, bd, rope_dim, sink_values=None, seed=0):
        self.assertEqual((512 - rope_dim) % bd, 0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        part = torch.randn(b, nt, 16, 512, device="cuda", generator=gen) * 8
        part[torch.rand(part.shape, device="cuda", generator=gen) < 0.1] = 0
        mx = torch.randn(b, nt, 16, device="cuda", generator=gen) * 3
        den = torch.rand(b, nt, 16, device="cuda", generator=gen)
        den[torch.rand(den.shape, device="cuda", generator=gen) < 0.2] = 0
        sink = torch.randn(16, device="cuda", generator=gen)
        if sink_values is not None:
            sink[: len(sink_values)] = torch.tensor(sink_values, device="cuda")
        angles = torch.randn(64, rope_dim // 2, device="cuda", generator=gen)
        freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1)
        freqs[:, :4] = torch.tensor([1.0, 0.0, 0.0, -1.0], device="cuda")
        positions = torch.randint(0, 64, (b,), device="cuda", generator=gen)
        inputs = (part, mx, den, sink, freqs, positions)
        tiled = launch(combine_gfx950, b, nt, bd, rope_dim, inputs)
        generic = launch(_combine, b, nt, bd, rope_dim, inputs)
        self.assertTrue(
            torch.equal(tiled.view(torch.int16), generic.view(torch.int16)),
            f"b={b} nt={nt} bd={bd} rope_dim={rope_dim}",
        )

    def test_bit_identical(self):
        for rope_dim in (64, 128):
            for b, nt, bd in [(1, 4, 64), (4, 20, 64), (3, 17, 64), (4, 5, 128)]:
                if (512 - rope_dim) % bd == 0:
                    self.check(b, nt, bd, rope_dim, seed=b + nt)

    def test_inf_sinks(self):
        self.check(
            2, 9, 64, 64, sink_values=[float("inf"), float("-inf"), 0.0, 1e4], seed=5
        )


if __name__ == "__main__":
    unittest.main()
