"""Validate SSD tiles against upstream, including live graphs and tracked states."""

import unittest

import torch

from sglang.kernels.ops.mamba.triton_ops.ssd_combined import (
    _mamba_chunk_scan_combined_fwd,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPrefillTiles(CustomTestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 required")
        torch.manual_seed(509)
        torch.set_num_threads(2)

    def assert_outputs_match(self, a, b, *, exact=False):
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x is None:
                self.assertIsNone(y)
            else:
                torch.testing.assert_close(
                    x,
                    y,
                    msg=lambda m, i=i: f"output {i}: {m}",
                    **({"atol": 0, "rtol": 0} if exact else {}),
                )

    def inputs(self, lengths, initial=True, track=True):
        n = sum(lengths)
        cu = (
            torch.tensor([0] + lengths, device="cuda", dtype=torch.int32)
            .cumsum(0)
            .int()
        )
        seq = torch.cat(
            [
                torch.full((v,), i, device="cuda", dtype=torch.int32)
                for i, v in enumerate(lengths)
            ]
        )[None]
        ci, co = Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(cu, 128, n)
        x = torch.randn(1, n, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1, n, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        kwargs = {
            "dt": torch.randn(1, n, 128, device="cuda", dtype=torch.bfloat16),
            "A": -torch.rand(128, device="cuda"),
            "B": b,
            "C": torch.randn_like(b),
            "chunk_size": 128,
            "D": torch.randn(128, device="cuda"),
            "dt_bias": torch.randn(128, device="cuda"),
            "dt_softplus": True,
            "initial_states": torch.randn(
                len(lengths), 128, 64, 128, device="cuda", dtype=torch.bfloat16
            )
            if initial
            else None,
            "seq_idx": seq,
            "chunk_indices": ci,
            "chunk_offsets": co,
            "cu_seqlens": cu,
            "state_dtype": torch.bfloat16,
        }
        if track:
            kwargs.update(
                track_seq_idx=torch.arange(
                    len(lengths), device="cuda", dtype=torch.int32
                ),
                track_end_locs=cu[:-1]
                + torch.tensor(
                    [max(1, v // 2) for v in lengths], device="cuda", dtype=torch.int32
                ),
            )
        return x, kwargs

    def run_ssd(self, x, kwargs, enabled, out):
        with envs.SGLANG_SM120_MAMBA_PREFILL_TILES.override(enabled):
            values = _mamba_chunk_scan_combined_fwd(x=x, out=out, **kwargs)
        return (out, *values)

    def test_boundaries_tracking_and_live_graph(self):
        for lengths, initial in [
            ([1], False),
            ([127], True),
            ([128], False),
            ([129], True),
            ([1021], True),
            ([1024], False),
            ([1024], True),
            ([137, 320, 544, 23], True),
        ]:
            with self.subTest(lengths=lengths, initial=initial):
                x, kwargs = self.inputs(lengths, initial)
                a, b = torch.empty_like(x), torch.empty_like(x)
                expected = self.run_ssd(x, kwargs, False, a)
                actual = self.run_ssd(x, kwargs, True, b)
                self.assert_outputs_match(actual, expected)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = self.run_ssd(x, kwargs, True, b)
                for _ in range(3):
                    x.normal_()
                    kwargs["B"].normal_()
                    kwargs["C"].normal_()
                    if initial:
                        kwargs["initial_states"].normal_()
                    expected = self.run_ssd(x, kwargs, False, a)
                    eager = self.run_ssd(x, kwargs, True, torch.empty_like(x))
                    # Different tile choices can change floating-point rounding.
                    self.assert_outputs_match(eager, expected)
                    graph.replay()
                    self.assert_outputs_match(actual, eager, exact=True)

    def test_eight_continuations(self):
        x, kwargs = self.inputs([1024])
        a, b = torch.empty_like(x), torch.empty_like(x)
        ka, kb = dict(kwargs), dict(kwargs)
        kb["initial_states"] = ka["initial_states"].clone()
        for _ in range(8):
            x.normal_()
            expected = self.run_ssd(x, ka, False, a)
            actual = self.run_ssd(x, kb, True, b)
            self.assert_outputs_match(actual, expected)
            # The production Mamba caller commits the varlen state to its pool.
            ka["initial_states"].copy_(expected[6])
            kb["initial_states"].copy_(actual[6])


if __name__ == "__main__":
    unittest.main()
