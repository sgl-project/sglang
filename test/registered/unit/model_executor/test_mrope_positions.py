"""CPU-only regression coverage for text-only MRoPE EXTEND positions."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_batch(prefixes, lengths, dtype=torch.int64, spec_info=None):
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.seq_lens_cpu = torch.tensor(
        [prefix + length for prefix, length in zip(prefixes, lengths)],
        dtype=torch.int64,
    )
    fb.positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, dtype=dtype)
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


class TestMRopeExtendPositions(CustomTestCase):
    def setUp(self):
        self.runner = SimpleNamespace(device="cpu")
        self.exec_ctx = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )
        self.exec_patch = patch.object(fbi, "get_exec", return_value=self.exec_ctx)
        self.exec_patch.start()
        self.addCleanup(self.exec_patch.stop)

    def test_text_positions_match_prefix_and_chunk_lengths(self):
        rng = random.Random(901103)
        for bs in (1, 2, 4, 8):
            for dtype in (torch.int32, torch.int64):
                for trial in range(16):
                    prefixes = [rng.randrange(200001) for _ in range(bs)]
                    lengths = [rng.randrange(33) for _ in range(bs)]
                    with self.subTest(bs=bs, dtype=dtype, trial=trial):
                        fb, batch = _make_batch(prefixes, lengths, dtype)
                        before = fb.positions.clone()
                        expected = torch.tensor(
                            [
                                pos
                                for prefix, length in zip(prefixes, lengths)
                                for pos in range(prefix, prefix + length)
                            ],
                            dtype=torch.int64,
                        ).expand(3, -1)
                        fb._compute_mrope_positions_extend(self.runner, batch)
                        self.assertTrue(torch.equal(fb.mrope_positions, expected))
                        self.assertEqual(fb.mrope_positions.dtype, torch.int64)
                        self.assertEqual(fb.mrope_positions.device.type, "cpu")
                        self.assertTrue(torch.equal(fb.positions, before))

    def test_empty_and_long_chunks(self):
        for lengths in ([0, 0], [0, 4096], [4096, 4096]):
            with self.subTest(lengths=lengths):
                fb, batch = _make_batch([0, 200000], lengths)
                expected = fb.positions.expand(3, -1).clone()
                fb._compute_mrope_positions_extend(self.runner, batch)
                self.assertEqual(tuple(fb.mrope_positions.shape), (3, sum(lengths)))
                self.assertTrue(torch.equal(fb.mrope_positions, expected))

    def test_text_and_nextn_metadata_do_not_rebuild_cpu_positions(self):
        for spec_info in (None, SimpleNamespace(positions=None)):
            with self.subTest(spec_info=spec_info):
                fb, batch = _make_batch([8], [4], spec_info=spec_info)
                expected = torch.tensor([[8, 9, 10, 11]] * 3, dtype=torch.int64)
                with patch.object(
                    torch, "arange", side_effect=AssertionError("CPU rebuild")
                ):
                    fb._compute_mrope_positions_extend(self.runner, batch)
                self.assertTrue(torch.equal(fb.mrope_positions, expected))

    def test_dllm_and_deterministic_modes_keep_original_coordinates(self):
        expected = torch.tensor([[8, 9]] * 3, dtype=torch.int64)
        for mode in ("dllm", "deterministic"):
            with self.subTest(mode=mode):
                fb, batch = _make_batch([8], [2])
                fb.positions = torch.tensor([100, 101], dtype=torch.int64)
                batch.dllm_config = object() if mode == "dllm" else None
                self.exec_ctx.deterministic.rl_on_policy_target = (
                    object() if mode == "deterministic" else None
                )
                fb._compute_mrope_positions_extend(self.runner, batch)
                self.assertTrue(torch.equal(fb.mrope_positions, expected))

    def test_explicit_speculative_positions_keep_original_fallback(self):
        fb, batch = _make_batch(
            [8], [2], spec_info=SimpleNamespace(positions=torch.tensor([100, 101]))
        )
        fb.positions = torch.tensor([100, 101], dtype=torch.int64)
        fb._compute_mrope_positions_extend(self.runner, batch)
        self.assertTrue(torch.equal(fb.mrope_positions, torch.tensor([[8, 9]] * 3)))

    def test_mixed_multimodal_batch_preserves_axis_positions(self):
        fb, batch = _make_batch([1, 8], [2, 2])
        mm_positions = torch.tensor(
            [[0, 1, 2, 3], [100, 101, 102, 103], [200, 201, 202, 203]],
            dtype=torch.int64,
        )
        batch.multimodal_inputs[0] = SimpleNamespace(mrope_positions=mm_positions)
        expected = torch.tensor(
            [[1, 2, 8, 9], [101, 102, 8, 9], [201, 202, 8, 9]],
            dtype=torch.int64,
        )
        fb._compute_mrope_positions_extend(self.runner, batch)
        self.assertTrue(torch.equal(fb.mrope_positions, expected))


if __name__ == "__main__":
    unittest.main()
