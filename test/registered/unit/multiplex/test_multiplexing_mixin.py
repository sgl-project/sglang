# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.multiplex.pdmux_context as pdmux
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.multiplex.pdmux_context import PDMuxConfig


class FakeRunningBatch:
    """Minimal stand-in for ScheduleBatch on the multiplexing paths."""

    def __init__(self, batch_size=0, empty=False):
        self._batch_size = batch_size
        self._empty = empty

    def is_empty(self):
        return self._empty

    def batch_size(self):
        return self._batch_size


def _make_scheduler(pdmux_config, real_sm_group_num, split_prefill_batch=None):
    stream_groups = [(f"p{i}", f"d{i}") for i in range(real_sm_group_num)]
    update_decode_attn_backend = mock.Mock()
    scheduler = SimpleNamespace(
        pdmux_config=pdmux_config,
        real_sm_group_num=real_sm_group_num,
        split_prefill_batch=split_prefill_batch,
        stream_groups=stream_groups,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                update_decode_attn_backend=update_decode_attn_backend
            )
        ),
    )
    return scheduler, update_decode_attn_backend, stream_groups


class TestAdjustStreamGroups(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: getattr(pdmux, name)
            for name in ("STREAM_GROUPS", "CURRENT_STREAM_IDX", "CURRENT_STREAM_GROUP")
        }

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(pdmux, name, value)

    def _adjust(self, scheduler, running_batch):
        with mock.patch.object(pdmux, "STREAM_GROUPS", scheduler.stream_groups):
            return SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch
            )

    def test_both_active_formula_selects_middle_group(self):
        cfg = PDMuxConfig()
        sched, recorder, groups = _make_scheduler(
            cfg, real_sm_group_num=8, split_prefill_batch=object()
        )
        idx, group = self._adjust(sched, FakeRunningBatch(batch_size=18))
        # stream_idx = max(1, min(6, 18 * 6 // 36)) == 3
        self.assertEqual(idx, 3)
        self.assertIs(group, groups[3])
        recorder.assert_called_once_with(3)

    def test_formula_saturates_and_floors(self):
        cfg = PDMuxConfig()
        for batch_size, expected in [(1, 1), (300, 6)]:
            sched, _, _ = _make_scheduler(
                cfg, real_sm_group_num=8, split_prefill_batch=object()
            )
            idx, _ = self._adjust(sched, FakeRunningBatch(batch_size=batch_size))
            self.assertEqual(idx, expected)

    def test_manual_divisions_pick_last_matching_tier(self):
        cfg = PDMuxConfig(
            manual_divisions=[
                [112, 20, 10],
                [96, 36, 50],
                [80, 52, 200],
            ]
        )
        sched, _, groups = _make_scheduler(
            cfg, real_sm_group_num=8, split_prefill_batch=object()
        )
        # decode_bs=60 matches tier 0 (>= 10) and tier 1 (>= 50) but not
        # tier 2 (>= 200). The loop has no break, so the last matching tier
        # wins: stream_idx = last matching i + 1 = 2. This pins the current
        # last-match semantics.
        idx, group = self._adjust(sched, FakeRunningBatch(batch_size=60))
        self.assertEqual(idx, 2)
        self.assertIs(group, groups[2])

        # With decode_bs=300 every tier matches, so the last tier wins.
        sched_all, _, _ = _make_scheduler(
            cfg, real_sm_group_num=8, split_prefill_batch=object()
        )
        idx_all, _ = self._adjust(sched_all, FakeRunningBatch(batch_size=300))
        self.assertEqual(idx_all, 3)

    def test_decode_only_uses_last_group(self):
        cfg = PDMuxConfig()
        sched, _, groups = _make_scheduler(cfg, real_sm_group_num=8)
        idx, group = self._adjust(sched, FakeRunningBatch(batch_size=4))
        self.assertEqual(idx, 7)
        self.assertIs(group, groups[7])

    def test_idle_uses_first_group(self):
        cfg = PDMuxConfig()
        sched, _, _ = _make_scheduler(cfg, real_sm_group_num=8)
        idx, _ = self._adjust(sched, FakeRunningBatch(batch_size=0, empty=True))
        self.assertEqual(idx, 0)


class TestUpdateSplitPrefillBatch(unittest.TestCase):
    def test_existing_split_batch_short_circuits(self):
        existing = object()
        sched = SimpleNamespace(
            split_prefill_batch=existing, get_new_batch_prefill=mock.Mock()
        )
        running = FakeRunningBatch()
        created, out = SchedulerMultiplexMixin.update_split_prefill_batch(
            sched, 32, running
        )
        self.assertFalse(created)
        self.assertIs(out, running)
        self.assertIs(sched.split_prefill_batch, existing)
        sched.get_new_batch_prefill.assert_not_called()

    def test_new_batch_marked_split_prefill(self):
        batch = SimpleNamespace(is_empty=lambda: False, forward_mode=None)
        new_running = object()
        sched = SimpleNamespace(
            split_prefill_batch=None,
            get_new_batch_prefill=mock.Mock(
                return_value=SimpleNamespace(
                    batch_to_run=batch, running_batch=new_running
                )
            ),
        )
        created, out = SchedulerMultiplexMixin.update_split_prefill_batch(
            sched, 32, object()
        )
        self.assertTrue(created)
        self.assertIs(out, new_running)
        self.assertIs(sched.split_prefill_batch, batch)
        self.assertEqual(batch.forward_mode, ForwardMode.SPLIT_PREFILL)

    def test_empty_batch_not_stored(self):
        sched = SimpleNamespace(
            split_prefill_batch=None,
            get_new_batch_prefill=mock.Mock(
                return_value=SimpleNamespace(batch_to_run=None, running_batch=object())
            ),
        )
        created, _ = SchedulerMultiplexMixin.update_split_prefill_batch(
            sched, 32, object()
        )
        self.assertFalse(created)
        self.assertIsNone(sched.split_prefill_batch)


if __name__ == "__main__":
    unittest.main()
