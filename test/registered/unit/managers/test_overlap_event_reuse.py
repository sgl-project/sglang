"""CPU control-flow tests; CUDA ordering is validated separately on device."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import overlap_events, round_outputs, scheduler_affinity
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestOverlapEventReuse(unittest.TestCase):
    def test_join_reuses_only_matching_stream_pair(self):
        dst = SimpleNamespace(cuda_stream=1, wait_stream=Mock(), wait_event=Mock())
        src = SimpleNamespace(cuda_stream=2)
        other = SimpleNamespace(cuda_stream=3)
        with (
            patch.object(overlap_events, "WAIT_EVENTS", True),
            patch.object(overlap_events, "_join_events", {}),
            patch("torch.cuda.Event", side_effect=lambda: Mock()) as create,
        ):
            overlap_events.wait_stream(dst, src)
            first = dst.wait_event.call_args.args[0]
            overlap_events.wait_stream(dst, src)
            self.assertIs(dst.wait_event.call_args.args[0], first)
            self.assertEqual(first.record.call_count, 2)
            first.record.assert_called_with(src)
            overlap_events.wait_stream(dst, other)
            self.assertIsNot(dst.wait_event.call_args.args[0], first)
            self.assertEqual(create.call_count, 2)
            dst.wait_stream.assert_not_called()

    def test_disabled_path_uses_native_wait_and_fresh_event(self):
        dst = Mock()
        src = Mock()
        device = SimpleNamespace(Event=Mock(side_effect=lambda: object()))
        with patch.object(overlap_events, "WAIT_EVENTS", False):
            overlap_events.wait_stream(dst, src)
            dst.wait_stream.assert_called_once_with(src)
            first = overlap_events.copy_done_event(device)
            second = overlap_events.copy_done_event(device)
            self.assertIsNot(first, second)
            self.assertEqual(device.Event.call_count, 2)

    def test_completion_ring_wraps_after_four_events(self):
        device = SimpleNamespace(Event=Mock(side_effect=lambda: object()))
        with (
            patch.object(overlap_events, "WAIT_EVENTS", True),
            patch.object(overlap_events, "_copy_done_ring", []),
            patch.object(overlap_events, "_copy_done_pos", 0),
        ):
            events = [overlap_events.copy_done_event(device) for _ in range(12)]
            self.assertEqual(device.Event.call_count, 4)
            self.assertEqual(len({id(event) for event in events[:4]}), 4)
            for index, event in enumerate(events):
                self.assertIs(event, events[index % 4])


class TestFutureMapIndexOps(unittest.TestCase):
    def _future_map(self):
        future_map = overlap_utils.FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.NONE,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(8, 4)),
            needs_cpu_seq_lens=False,
        )
        future_map.new_seq_lens_buf.fill_(-1)
        return future_map

    def test_publish_and_gather_match_advanced_indexing(self):
        indices = torch.tensor([5, 1, 6], dtype=torch.int64)
        lens = torch.tensor([17, 3, 9], dtype=torch.int32)
        results = []
        for index_ops in (False, True):
            with patch.object(overlap_utils, "_INDEX_OPS", index_ops):
                future_map = self._future_map()
                future_map.publish(indices, lens)
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace(future_indices=indices.flip(0))
                )
                future_map.resolve_seq_lens_cpu(batch)
                results.append((future_map.new_seq_lens_buf.clone(), batch.seq_lens))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[1][1].tolist(), [9, 3, 17])
        self.assertEqual(results[1][1].dtype, torch.int64)

    def test_staged_lengths_skip_the_publish_wait_and_the_gather(self):
        """SGLANG_OPT_SPEC_STAGED_SEQ_LENS: a round given the previous round's published
        lengths takes them as they are: no wait on the publish, no gather, and none of
        the gathered rows' poisoning; without them it gathers as before."""
        indices = torch.tensor([5, 1, 6], dtype=torch.int64)
        staged = torch.tensor([17, 3, 9], dtype=torch.int64)
        for given in (False, True):
            future_map = self._future_map()
            future_map.publish(indices, staged)
            future_map.publish_ready = Mock()
            future_map._publish_fresh = True
            poison = Mock()
            batch = SimpleNamespace(spec_info=SimpleNamespace(future_indices=indices))
            with (
                self.subTest(given=given),
                patch.object(overlap_utils, "_DEBUG_ASSERT", True),
                patch.object(overlap_utils, "_assert_nonneg_and_invalidate", poison),
            ):
                future_map.resolve_seq_lens_cpu(
                    batch, staged_seq_lens=staged if given else None
                )
                self.assertEqual(future_map.publish_ready.wait.called, not given)
                self.assertEqual(poison.called, not given)
                self.assertFalse(future_map._publish_fresh)
                if given:
                    self.assertIs(batch.seq_lens, staged)
                else:
                    self.assertTrue(torch.equal(batch.seq_lens, staged))
                    self.assertIsNot(batch.seq_lens, staged)


class TestRoundOutputRing(unittest.TestCase):
    def test_ring_wraps_after_eight_tensors_of_a_shape(self):
        with patch.object(round_outputs, "_rings", {}):
            first = [
                round_outputs.round_empty((4,), torch.int64, "cpu") for _ in range(8)
            ]
            self.assertEqual(len({id(t) for t in first}), 8)
            self.assertIs(round_outputs.round_empty((4,), torch.int64, "cpu"), first[0])
            other = round_outputs.round_empty((4,), torch.int32, "cpu")
            self.assertNotIn(id(other), {id(t) for t in first})
            self.assertEqual(other.dtype, torch.int32)

    def test_index_kernels_take_their_outputs_from_the_ring_only_when_enabled(self):
        from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_layout

        seq_lens = torch.empty((0,), dtype=torch.int64)
        for enabled in (False, True):
            with (
                patch.object(round_outputs, "ENABLED", enabled),
                patch.object(round_outputs, "_rings", {}),
            ):
                prefix, *_ = prepare_draft_extend_layout(seq_lens, 4, False)
                pooled = round_outputs._rings.get(((0,), torch.int32, seq_lens.device))
                self.assertEqual(pooled is not None, enabled)
                if enabled:
                    self.assertIs(prefix, pooled[0][0])


class TestSchedulerMainThreadAffinity(unittest.TestCase):
    """Affinity syscalls are mocked: the test process keeps its own CPU set."""

    def setUp(self):
        for name, value in (
            ("_main_cores", frozenset()),
            ("_other_cores", frozenset()),
            ("_last_scan", 0.0),
        ):
            patcher = patch.object(scheduler_affinity, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_main_thread_takes_its_gpu_pair_and_others_leave_the_reserve(self):
        affinity = scheduler_affinity
        for cpus, gpu, main, reserved in (
            (64, 0, {62, 63}, 4),
            (64, 1, {60, 61}, 4),
            (128, 3, {120, 121}, 8),
        ):
            with (
                self.subTest(cpus=cpus, gpu=gpu),
                patch.object(
                    affinity.os, "sched_getaffinity", return_value=set(range(cpus))
                ),
                patch.object(affinity.os, "sched_setaffinity") as apply,
                patch.object(affinity.os, "listdir", return_value=["10", "20"]),
                patch.object(affinity.threading, "get_native_id", return_value=10),
            ):
                affinity.pin_main_thread(gpu)
                others = frozenset(range(cpus - reserved))
                self.assertEqual(apply.call_args_list[0].args, (0, frozenset(main)))
                self.assertEqual(apply.call_args_list[1].args, (20, others))

    def test_small_cpu_set_and_idle_rescan(self):
        affinity = scheduler_affinity
        with (
            patch.object(affinity.os, "sched_getaffinity", return_value=set(range(8))),
            patch.object(affinity.os, "sched_setaffinity") as apply,
        ):
            affinity.pin_main_thread(0)
            apply.assert_not_called()
        with patch.object(affinity, "_move_other_threads") as scan:
            affinity.maybe_repin()
            scan.assert_not_called()
            affinity._main_cores = frozenset({2, 3})
            with patch.object(
                affinity.time, "monotonic", side_effect=[6.0, 10.0, 11.5]
            ):
                for _ in range(3):
                    affinity.maybe_repin()
            self.assertEqual(scan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
