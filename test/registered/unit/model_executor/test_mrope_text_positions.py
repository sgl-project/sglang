"""Text-only extend batches take their mrope rows from self.positions on the
device; the rows equal the per-request host loop's."""

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    compute_position_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _kernel_positions(prefix, extend):
    """compute_position_kernel: request i writes prefix_i + j at cumsum(extend[:i]) + j."""
    out = torch.empty(sum(extend), dtype=torch.int64)
    start = 0
    for p, e in zip(prefix, extend):
        out[start : start + e] = p + torch.arange(e, dtype=torch.int32).to(torch.int64)
        start += e
    return out


class _MM:
    def __init__(self, n, delta):
        self.mrope_positions = torch.randint(0, 1000, (3, n), dtype=torch.int64)
        self.mrope_position_delta = torch.tensor([[delta]])
        self.mrope_position_delta_repeated_cache = None


def _make(prefix, extend, *, mm=None, torch_positions=False, pad=0, mirror=True):
    fb = object.__new__(ForwardBatch)
    if torch_positions:
        pos = compute_position_torch(
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(extend, dtype=torch.int32),
        )[0]
    else:
        pos = _kernel_positions(prefix, extend)
    if pad:
        pos = torch.cat([pos, torch.full((pad,), -7, dtype=torch.int64)])
    fb.positions = pos
    fb.extend_seq_lens_cpu = list(extend) if mirror else None
    fb.extend_prefix_lens_cpu = list(prefix) if mirror else None
    fb.seq_lens_cpu = torch.tensor([p + e for p, e in zip(prefix, extend)])
    fb.mrope_positions = None
    batch = SimpleNamespace(
        extend_lens=list(extend),
        prefix_lens=list(prefix),
        multimodal_inputs=mm if mm is not None else [None] * len(extend),
        dllm_config=None,
        reqs=[SimpleNamespace(session=None) for _ in extend],
    )
    return fb, batch


def _exec(rl_on_policy_target=None):
    return mock.patch.object(
        fbi,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=rl_on_policy_target)
        ),
    )


class TestTextOnlyMropePositions(CustomTestCase):
    def setUp(self):
        super().setUp()
        gate = envs.SGLANG_ENABLE_MROPE_TEXT_DEVICE_POSITIONS.override(True)
        gate.__enter__()
        self.addCleanup(gate.__exit__, None, None, None)

    def _both(self, *args, rl=None, **kwargs):
        """(device-path rows or None, rows the host loop builds)."""
        runner = SimpleNamespace(device="cpu")
        with _exec(rl):
            fb, batch = _make(*args, **kwargs)
            device_rows = fb._text_only_mrope_positions(batch)
            fb, batch = _make(*args, **kwargs)
            with mock.patch.object(
                ForwardBatch, "_text_only_mrope_positions", lambda self, b: None
            ):
                fb._compute_mrope_positions_extend(runner, batch)
            return device_rows, fb.mrope_positions

    def _assert_same(self, rows, loop_rows):
        self.assertIsNotNone(rows)
        self.assertEqual(rows.dtype, torch.int64)
        self.assertEqual(rows.shape, loop_rows.shape)
        self.assertTrue(rows.is_contiguous())
        self.assertTrue(torch.equal(rows, loop_rows))

    def test_named_layouts_equal_the_host_loop(self):
        for prefix, extend in (
            ([0, 0, 0], [5, 3, 7]),
            ([64, 0, 128], [10, 20, 1]),
            ([16384], [3616]),
            ([32768, 0, 4096], [1000, 2000, 500]),
            ([0, 4096, 511, 77], [300, 200, 1, 1]),
            ([5, 9, 0], [0, 4, 3]),
            ([262143 - 16384], [16384]),
        ):
            self._assert_same(*self._both(prefix, extend))
        self._assert_same(*self._both([3, 0], [2, 9], torch_positions=True))

    def test_random_layouts_equal_the_host_loop(self):
        rng = random.Random(20261101)
        for i in range(500):
            bs = rng.randint(1, 128)
            prefix = [rng.choice([0, 0, rng.randint(1, 262000)]) for _ in range(bs)]
            budget = rng.choice([64, 512, 4096, 16384])
            extend = [rng.randint(0, max(1, budget // bs)) for _ in range(bs)]
            if sum(extend) == 0:
                extend[0] = 1
            self._assert_same(*self._both(prefix, extend, torch_positions=i % 2 == 0))

    def test_rl_on_policy_target_treats_multimodal_rows_as_text(self):
        mm = [None, _MM(40, -3), None]
        self._assert_same(*self._both([0, 7, 0], [5, 40, 6], mm=mm, rl="fsdp"))

    def test_other_batches_keep_the_host_loop(self):
        mm = [None, _MM(40, -3), None]
        for kwargs in (
            dict(prefix=[0, 0, 0], extend=[5, 40, 6], mm=mm),
            dict(prefix=[0, 0], extend=[4, 4], pad=8),
            dict(prefix=[0, 2], extend=[4, 4], mirror=False),
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), _exec():
                fb, batch = _make(**kwargs)
                self.assertIsNone(fb._text_only_mrope_positions(batch))
        with _exec():
            fb, batch = _make([0], [4])
            batch.dllm_config = object()
            self.assertIsNone(fb._text_only_mrope_positions(batch))

    def test_positions_not_built_from_these_lens_are_caught(self):
        """Positive control: rows are taken from self.positions, so positions that
        are not compute_position's must make the comparison fail."""
        runner = SimpleNamespace(device="cpu")
        with _exec():
            fb, batch = _make([3, 0], [4, 2])
            fb._compute_mrope_positions_extend(runner, batch)
            shifted, batch = _make([3, 0], [4, 2])
            shifted.positions = shifted.positions + 1
            shifted._compute_mrope_positions_extend(runner, batch)
        self.assertFalse(torch.equal(fb.mrope_positions, shifted.mrope_positions))


class TestMropeDevicePositionsSwitch(CustomTestCase):
    """SGLANG_ENABLE_MROPE_TEXT_DEVICE_POSITIONS: off by default; off keeps the
    host loop for every batch."""

    def test_defaults_off(self):
        self.assertIs(envs.SGLANG_ENABLE_MROPE_TEXT_DEVICE_POSITIONS.default, False)

    def test_off_keeps_the_host_loop(self):
        runner = SimpleNamespace(device="cpu")
        with _exec(), envs.SGLANG_ENABLE_MROPE_TEXT_DEVICE_POSITIONS.override(False):
            fb, batch = _make([64, 0, 128], [10, 20, 1])
            self.assertIsNone(fb._text_only_mrope_positions(batch))
            fb._compute_mrope_positions_extend(runner, batch)
        with _exec():
            want, batch = _make([64, 0, 128], [10, 20, 1])
            with mock.patch.object(
                ForwardBatch, "_text_only_mrope_positions", lambda self, b: None
            ):
                want._compute_mrope_positions_extend(runner, batch)
        self.assertTrue(torch.equal(fb.mrope_positions, want.mrope_positions))


if __name__ == "__main__":
    unittest.main()
