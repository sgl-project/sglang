"""CUDA coverage for device-side text MRoPE construction; no model weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _make_batch(prefixes, lengths, dtype=torch.int64, spec_info=None):
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.seq_lens_cpu = torch.tensor(
        [prefix + length for prefix, length in zip(prefixes, lengths)],
        dtype=torch.int64,
    )
    fb.positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, dtype=dtype, device="cuda")
            for prefix, length in zip(prefixes, lengths)
        ]
    )
    fb.spec_info = spec_info
    batch = SimpleNamespace(
        multimodal_inputs=[None] * len(prefixes),
        extend_lens=lengths,
        prefix_lens=prefixes,
        dllm_config=None,
        reqs=[SimpleNamespace(session=None) for _ in prefixes],
    )
    return fb, batch


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMRopePositionsCUDA(CustomTestCase):
    def setUp(self):
        self.runner = SimpleNamespace(device="cuda")
        context = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )
        patcher = patch.object(fbi, "get_exec", return_value=context)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_text_positions_without_cpu_rebuild_or_host_sync(self):
        for dtype in (torch.int32, torch.int64):
            for bs in (1, 4):
                for spec_info in (None, SimpleNamespace(positions=None)):
                    with self.subTest(dtype=dtype, bs=bs, spec_info=spec_info):
                        fb, batch = _make_batch(
                            [17 * i for i in range(bs)],
                            [4096] * bs,
                            dtype,
                            spec_info,
                        )
                        expected = fb.positions.to(torch.int64).expand(3, -1).clone()
                        before = fb.positions.clone()
                        torch.cuda.synchronize()
                        previous = torch.cuda.get_sync_debug_mode()
                        try:
                            torch.cuda.set_sync_debug_mode("error")
                            with patch.object(
                                torch,
                                "arange",
                                side_effect=AssertionError("CPU rebuild"),
                            ):
                                fb._compute_mrope_positions_extend(self.runner, batch)
                        finally:
                            torch.cuda.set_sync_debug_mode(previous)
                        self.assertEqual(fb.mrope_positions.device, fb.positions.device)
                        self.assertEqual(fb.mrope_positions.dtype, torch.int64)
                        torch.testing.assert_close(
                            fb.mrope_positions, expected, rtol=0, atol=0
                        )
                        torch.testing.assert_close(fb.positions, before, rtol=0, atol=0)

    def test_non_default_stream_observes_prior_position_updates(self):
        fb, batch = _make_batch([0], [2048], torch.int32)
        expected = (fb.positions.to(torch.int64) + 8).expand(3, -1).clone()
        batch.prefix_lens = [8]
        fb.seq_lens_cpu += 8
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fb.positions.add_(8)
            fb._compute_mrope_positions_extend(self.runner, batch)
        stream.synchronize()
        torch.testing.assert_close(fb.mrope_positions, expected, rtol=0, atol=0)

    def test_mixed_multimodal_fallback_preserves_spatial_axes(self):
        fb, batch = _make_batch([1, 8], [2, 2])
        batch.multimodal_inputs[0] = SimpleNamespace(
            mrope_positions=torch.tensor(
                [[0, 1, 2, 3], [100, 101, 102, 103], [200, 201, 202, 203]],
                dtype=torch.int64,
            )
        )
        expected = torch.tensor(
            [[1, 2, 8, 9], [101, 102, 8, 9], [201, 202, 8, 9]],
            dtype=torch.int64,
            device="cuda",
        )
        fb._compute_mrope_positions_extend(self.runner, batch)
        torch.testing.assert_close(fb.mrope_positions, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
