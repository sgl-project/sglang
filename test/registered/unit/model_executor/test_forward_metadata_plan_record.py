"""Unit tests for the ForwardBatch attention plan marker / plan record.

Covers the contract behind ``skip_attn_backend_init`` deprecation:
  * fresh batches need planning; marked batches don't
  * the plan record (planned bs / num tokens) snapshots mark-time shapes
  * reshape after marking triggers a re-plan only for sites that opted
    into ``replan_equivalent``; re-marking re-records the new shapes
  * the deprecated kwarg shim maps explicit values onto the marker
    (mapped, not ignored) and warns once per process

Pure dataclass logic — CPU only.
"""

import random
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.model_executor.forward_batch_info as fbi
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_batch(bs: int = 2, num_tokens: int = 2) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.zeros(bs, dtype=torch.long),
        seq_lens=torch.ones(bs, dtype=torch.long),
        out_cache_loc=torch.zeros(bs, dtype=torch.long),
        seq_lens_sum=bs,
    )


class TestForwardMetadataPlanRecord(CustomTestCase):
    def test_fresh_batch_needs_planning(self):
        fb = _make_batch()
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertFalse(fb.forward_metadata_ready)

    def test_mark_records_shapes_and_skips_planning(self):
        fb = _make_batch(bs=3, num_tokens=7)
        fb.mark_forward_metadata_ready()
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 3)
        self.assertEqual(fb.forward_metadata_planned_num_tokens, 7)

    def test_reshape_without_opt_in_keeps_skipping(self):
        # Wrapper regimes must never auto-re-plan (would clobber per-step metadata).
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready()
        fb.batch_size = 4  # DP padding reshapes the batch
        self.assertFalse(fb.needs_forward_metadata_init())

    def test_reshape_with_opt_in_replans(self):
        fb = _make_batch(bs=2, num_tokens=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())

        fb.batch_size = 4  # bs drift (prepare_mlp_sync_batch decode pad)
        self.assertTrue(fb.needs_forward_metadata_init())

        fb.batch_size = 2
        fb.input_ids = torch.zeros(6, dtype=torch.long)  # token drift
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_remark_re_records_padded_shapes(self):
        # Per-step loops re-mark each plan; the re-mark must snapshot padded shapes.
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        fb.batch_size = 4
        self.assertTrue(fb.needs_forward_metadata_init())
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 4)


def _mrope_batch(prefix, extend, *, mm=None, mirror=True, positions=None):
    """A text-only extend ForwardBatch as compute_position leaves it."""
    fb = object.__new__(ForwardBatch)
    if positions is None:
        positions = fbi.compute_position_torch(
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(extend, dtype=torch.int32),
        )[0]
    fb.positions = positions
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


class TestMropeTextDevicePositions(CustomTestCase):
    """SGLANG_OPT_MROPE_TEXT_DEVICE_POSITIONS: a text-only extend batch's mrope
    rows repeat self.positions; they equal the per-request host loop's."""

    def _rows(self, *args, enabled=True, rl=None, **kwargs):
        runner = SimpleNamespace(device="cpu")
        exec_ = SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target=rl))
        with (
            mock.patch.object(fbi, "get_exec", lambda: exec_),
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            envs.SGLANG_OPT_MROPE_TEXT_DEVICE_POSITIONS.override(enabled),
        ):
            fb, batch = _mrope_batch(*args, **kwargs)
            device_rows = fb._text_only_mrope_positions(batch)
            fb._compute_mrope_positions_extend(runner, batch)
            loop, batch = _mrope_batch(*args, **kwargs)
            with mock.patch.object(
                ForwardBatch, "_text_only_mrope_positions", lambda self, b: None
            ):
                loop._compute_mrope_positions_extend(runner, batch)
        return device_rows, fb.mrope_positions, loop.mrope_positions

    def test_text_only_rows_equal_the_host_loop(self):
        rng = random.Random(20261111)
        layouts = [([0, 0, 0], [5, 3, 7]), ([64, 0, 128], [10, 20, 1])]
        layouts.append(([5, 9, 0], [0, 4, 3]))
        for _ in range(100):
            bs = rng.randint(1, 64)
            prefix = [rng.choice([0, rng.randint(1, 262000)]) for _ in range(bs)]
            extend = [rng.randint(0, 64) for _ in range(bs)]
            extend[0] = max(extend[0], 1)
            layouts.append((prefix, extend))
        for prefix, extend in layouts:
            device_rows, rows, loop = self._rows(prefix, extend)
            self.assertIsNotNone(device_rows)
            self.assertEqual((rows.dtype, rows.shape), (loop.dtype, loop.shape))
            self.assertTrue(torch.equal(rows, loop))

    def test_rl_on_policy_target_treats_multimodal_rows_as_text(self):
        mm = SimpleNamespace(mrope_positions=torch.randint(0, 99, (3, 40)))
        device_rows, rows, loop = self._rows(
            [0, 7, 0], [5, 40, 6], mm=[None, mm, None], rl="fsdp"
        )
        self.assertIsNotNone(device_rows)
        self.assertTrue(torch.equal(rows, loop))

    def test_other_batches_and_switch_off_keep_the_host_loop(self):
        mm = SimpleNamespace(mrope_positions=torch.randint(0, 99, (3, 40)))
        padded = torch.cat([torch.arange(8), torch.full((8,), -7)])
        for kwargs in (
            dict(prefix=[0, 0, 0], extend=[5, 40, 6], mm=[None, mm, None]),
            dict(prefix=[0, 0], extend=[4, 4], positions=padded),
            dict(prefix=[0, 2], extend=[4, 4], mirror=False),
            dict(prefix=[64, 0], extend=[10, 20], enabled=False),
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                device_rows, rows, loop = self._rows(**kwargs)
                self.assertIsNone(device_rows)
                self.assertTrue(torch.equal(rows, loop))

    def test_positions_not_built_from_these_lens_are_caught(self):
        shifted = fbi.compute_position_torch(
            torch.tensor([3, 0], dtype=torch.int32),
            torch.tensor([4, 2], dtype=torch.int32),
        )[0]
        _, rows, loop = self._rows([3, 0], [4, 2], positions=shifted + 1)
        self.assertFalse(torch.equal(rows, loop))


class TestDeprecatedSkipKwargShim(CustomTestCase):
    def setUp(self):
        self._saved_warned = fbi._skip_attn_backend_init_warned
        fbi._skip_attn_backend_init_warned = False

    def tearDown(self):
        fbi._skip_attn_backend_init_warned = self._saved_warned

    def test_none_is_a_silent_no_op(self):
        fb = _make_batch()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fb.apply_deprecated_skip_attn_backend_init(None)
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_true_maps_onto_marker_and_warns(self):
        # Mapped, not ignored: a no-op would silently re-plan multi-step metadata.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, DeprecationWarning))

    def test_false_warns_but_does_not_mark(self):
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(False)
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)

    def test_warns_once_per_process(self):
        # Hot-loop guard: per-forward callers must not pay warnings.warn repeatedly.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
            fb.apply_deprecated_skip_attn_backend_init(True)
            _make_batch().apply_deprecated_skip_attn_backend_init(False)
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
