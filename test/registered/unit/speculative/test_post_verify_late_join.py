"""Post-verify late join: the slot allocator's guard, the switches, the refusals and the pool presize (CPU)."""

import random
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as gdn_commit
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.speculative import post_verify_late_join as late_join
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Gpu:
    """A fake GPU timeline: commit ``n`` is the n-th side-stream replay; commits <= ``completed`` have finished."""

    def __init__(self):
        self.pending = 0
        self.completed = 0
        self.waits = []  # the commit sequence each stream wait covers


class _Event:
    gpu = None

    def __init__(self):
        self.seq = None

    def record(self, stream=None):
        self.seq = _Event.gpu.pending

    def query(self):
        return self.seq is None or self.seq <= _Event.gpu.completed

    def synchronize(self):
        _Event.gpu.completed = max(_Event.gpu.completed, self.seq)


class _Stream:
    def wait_event(self, event):
        _Event.gpu.waits.append(event.seq)


def _fake_cuda(gpu):
    _Event.gpu = gpu
    stack = ExitStack()
    stack.enter_context(mock.patch.object(torch.cuda, "Event", _Event))
    stream = _Stream()
    stack.enter_context(
        mock.patch.object(torch.cuda, "current_stream", lambda *a: stream)
    )
    return stack


def _schedule(seed, *, active, size=64, steps=400):
    """A serving-like schedule of commits, frees of finished rows, group and direct allocations and GPU progress.

    Returns (hand-outs of a slot a running commit still writes without a covering wait, hand-outs that waited
    needlessly, total waits, the slot order)."""
    rng = random.Random(seed)
    gpu = _Gpu()
    graphs = SimpleNamespace(
        _commit_pending=False, _commit_done=_Event(), _side_stream=None
    )
    misses = needless = 0
    order = []
    with (
        _fake_cuda(gpu),
        mock.patch.object(late_join, "_graphs", graphs if active else None),
    ):
        alloc = MambaSlotAllocator(size=size, device="cpu")
        live, written_by, freed_under = set(), {}, {}

        def hand_out(slot, waits_before):
            nonlocal misses, needless
            order.append(slot)
            seq = freed_under.pop(slot, 0)
            running = seq > gpu.completed
            covered = any(w is not None and w >= seq for w in gpu.waits[waits_before:])
            misses += running and not covered
            needless += not running and len(gpu.waits) > waits_before
            live.add(slot)

        for _ in range(steps):
            if live:  # a decode pass: the commit replays over the batch's slots
                gpu.pending += 1
                graphs._commit_pending = True
                graphs._commit_done.seq = gpu.pending
                for slot in live:
                    written_by[slot] = gpu.pending
            for slot in [
                s for s in sorted(live) if rng.random() < 0.08
            ]:  # finished rows
                live.discard(slot)
                freed_under[slot] = written_by.get(slot, 0)
                alloc.free(torch.tensor([slot]))
            if rng.random() < 0.5:
                gpu.completed = max(gpu.completed, gpu.pending - rng.choice([0, 0, 1]))
            if rng.random() < 0.6:
                n = rng.randint(1, 6)
                alloc.alloc_group_begin(n)
                for _ in range(rng.randint(0, n + 1)):
                    waits_before = len(gpu.waits)
                    got = alloc.alloc(1)
                    if got is None:
                        break
                    hand_out(int(got[0]), waits_before)
                alloc.alloc_group_end()
            if rng.random() < 0.4 and alloc.available_size() >= 2:
                waits_before = len(gpu.waits)
                for slot in alloc.alloc(rng.choice([1, 2])).tolist():
                    hand_out(slot, waits_before)
            if rng.random() < 0.3:  # the next verify joined
                graphs._commit_pending = False
            if rng.random() < 0.05:
                gpu.completed = gpu.pending
    return misses, needless, len(gpu.waits), order


class TestSlotAllocatorGuard(CustomTestCase):
    def test_no_slot_reaches_a_new_owner_before_its_commit(self):
        """A finished row's slot is freed while its commit may still run; whoever takes it next must wait first.
        Inactive, the allocator hands out the same slots in the same order and never waits."""
        total_waits = 0
        for seed in range(150):
            misses, needless, waits, order_on = _schedule(seed, active=True)
            self.assertEqual(misses, 0, seed)
            total_waits += waits
            _, _, waits_off, order_off = _schedule(seed, active=False)
            self.assertEqual(waits_off, 0, seed)
            self.assertEqual(order_on, order_off, seed)
        self.assertGreater(total_waits, 0)  # the schedules do exercise the guard


class TestSwitches(CustomTestCase):
    def test_defaults_keep_the_early_join(self):
        self.assertFalse(late_join.LATE_JOIN)
        self.assertEqual(late_join.max_graph_keys(), 64)
        self.assertIsNone(envs.SGLANG_FORWARD_HELPER_STREAM_PRIORITY.get())
        with late_join.side_commit_tiling():
            self.assertIsNone(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE)
        with envs.SGLANG_POST_VERIFY_MAX_GRAPH_KEYS.override(0):
            with self.assertRaises(ValueError):
                late_join.max_graph_keys()

    def test_side_commit_warps_restore_on_exit_and_error(self):
        with mock.patch.object(late_join, "_graphs", SimpleNamespace()):
            with late_join.side_commit_tiling():
                self.assertEqual(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE, 2)
                with gdn_commit.tiled_commit_warps(4):
                    self.assertEqual(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE, 4)
                self.assertEqual(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE, 2)
            with self.assertRaises(RuntimeError):
                with late_join.side_commit_tiling():
                    raise RuntimeError("capture failed")
        self.assertIsNone(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE)

    def test_riders_wait_for_activation(self):
        """With the switch set but the late join refused (never activated), the side commit keeps its warps."""
        with mock.patch.object(late_join, "LATE_JOIN", True):
            with late_join.side_commit_tiling():
                self.assertIsNone(gdn_commit._TILED_COMMIT_WARPS_OVERRIDE)


class TestRefusal(CustomTestCase):
    def _refusal(self, **overrides):
        shape = dict(
            mode="null",
            tp=1,
            pp=1,
            no_overlap=False,
            unified=False,
            int8=False,
            extra=True,
        )
        shape.update(overrides)
        ns = SimpleNamespace
        getters = dict(
            get_disagg=lambda: ns(disaggregation_mode=shape["mode"]),
            get_parallel=lambda: ns(tp_size=shape["tp"], pp_size=shape["pp"]),
            get_schedule=lambda: ns(disable_overlap_schedule=shape["no_overlap"]),
            get_memory=lambda: ns(enable_unified_memory=shape["unified"]),
            get_exec=lambda: ns(
                mamba=ns(
                    enable_int8_mamba_checkpoint=shape["int8"],
                    enable_mamba_extra_buffer=shape["extra"],
                )
            ),
        )
        with ExitStack() as stack:
            for name, fn in getters.items():
                stack.enter_context(
                    mock.patch(f"sglang.srt.runtime_context.{name}", fn)
                )
            return late_join.late_join_refusal()

    def test_only_the_guarded_shape_is_accepted(self):
        self.assertIsNone(self._refusal())
        for overrides in (
            dict(mode="decode"),
            dict(mode="prefill"),
            dict(tp=2),
            dict(pp=2),
            dict(no_overlap=True),
            dict(unified=True),
            dict(int8=True),
            dict(extra=False),
        ):
            self.assertIsNotNone(self._refusal(**overrides), overrides)


class TestWaitAndSync(CustomTestCase):
    def test_wait_orders_and_keeps_the_join_sync_finishes_it(self):
        gpu = _Gpu()
        with _fake_cuda(gpu):
            graphs = SimpleNamespace(
                _commit_pending=False, _commit_done=_Event(), _side_stream=None
            )
            with mock.patch.object(late_join, "_graphs", None):
                self.assertFalse(late_join.wait("x") or late_join.sync("x"))
            with mock.patch.object(late_join, "_graphs", graphs):
                self.assertFalse(
                    late_join.wait("x") or late_join.sync("x")
                )  # nothing pending
                gpu.pending = 5
                graphs._commit_done.seq = 5
                graphs._commit_pending = True
                d2h = _Stream()
                self.assertTrue(late_join.wait("hicache_write", stream=d2h))
                self.assertEqual(gpu.waits[-1], 5)
                self.assertTrue(
                    graphs._commit_pending
                )  # the forward stream still joins
                self.assertTrue(late_join.sync("idle_pass"))
                self.assertEqual(gpu.completed, 5)
                self.assertFalse(graphs._commit_pending)
                self.assertFalse(late_join.sync("idle_pass"))


class TestPresize(CustomTestCase):
    def test_two_sampling_buffers_of_the_largest_key_or_nothing(self):
        vocab, tokens, requests = 248320, 4, 128
        want = 2 * requests * tokens * vocab * 4 + (64 << 20)
        self.assertEqual(
            late_join.presize_bytes(
                max_requests=requests,
                tokens_per_request=tokens,
                vocab=vocab,
                free_bytes=20 << 30,
            ),
            want,
        )
        self.assertEqual(
            late_join.presize_bytes(
                max_requests=requests,
                tokens_per_request=tokens,
                vocab=vocab,
                free_bytes=want + (1 << 30) - 1,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
