"""The queue trigger must hold a prefill delay until its own deadline.

The trigger delays prefill until the waiting queue reaches
``running_req * queue_min_ratio``, bounded by ``max_delay_ms``. Three things
used to end that delay early: a threshold capped by the observed
max_prefill_bs high-watermark (which a delayed pass feeds with the waiting
queue), the one-shot ``skip_first_delayer`` bypass that belongs to the slot
trigger, and the ``max_delay_passes`` bound that also belongs to it.

Once the deadline is the normal end of a delay, every rank must reach it on the
same pass: each rank starts its own timer, and ranks whose clocks straddle the
deadline would otherwise build different batches.
"""

import dataclasses
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.distributed
import torch.multiprocessing

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


def _gather_on_one_rank(output, local, group=None):
    output.copy_(local)


class TestPrefillDelayerQueueTrigger(CustomTestCase):
    def make_delayer(
        self,
        *,
        queue_min_ratio=0.3,
        max_delay_passes=30,
        max_delay_ms=5000.0,
        prefill_max_requests=None,
    ):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            published_topology(
                enable_prefill_delayer=True,
                prefill_delayer_queue_min_ratio=queue_min_ratio,
                prefill_delayer_max_delay_ms=max_delay_ms,
                prefill_max_requests=prefill_max_requests,
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.managers.prefill_delayer.all_gather_single",
                side_effect=_gather_on_one_rank,
            )
        )
        return PrefillDelayer(
            cpu_group=None,
            max_delay_passes=max_delay_passes,
            token_usage_low_watermark=None,
        )

    def negotiate(self, delayer, **kwargs):
        kwargs.setdefault("local_prefillable", True)
        kwargs.setdefault("token_usage", 0.5)
        kwargs.setdefault("max_running_requests", 1024)
        return delayer._negotiate_should_allow_prefill(**kwargs)

    def test_threshold_ignores_collapsing_max_prefill_bs(self):
        # 50 running requests at ratio 0.3 ask for 15 queued requests. The
        # recent-attempt high-watermark drops to 11 while the delay runs,
        # because a delayed pass observes the waiting queue instead of a real
        # batch size. The delay must still hold at a queue of 11.
        delayer = self.make_delayer()
        delayer.skip_first_delayer = False
        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=16, waiting_queue_len=2
        )
        self.assertFalse(out.output_allow)

        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=11, waiting_queue_len=11
        )
        self.assertEqual((out.output_allow, out.output_reason), (False, "delay"))

        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=11, waiting_queue_len=15
        )
        self.assertEqual((out.output_allow, out.output_reason), (True, "wait_success"))

    def test_prefill_max_requests_still_caps_the_threshold(self):
        # A configured request limit is static, so it may cap the threshold:
        # waiting for more than one batch can admit would never be satisfied.
        delayer = self.make_delayer(prefill_max_requests=8)
        delayer.skip_first_delayer = False
        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=16, waiting_queue_len=8
        )
        self.assertTrue(out.output_allow)

    def test_queue_trigger_delays_on_the_first_pass(self):
        # skip_first_delayer exists for the slot trigger's first merge_batch.
        # A queue-only trigger must not consume it, and must not be skipped.
        delayer = self.make_delayer()
        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=16, waiting_queue_len=2
        )
        self.assertFalse(out.output_allow)
        self.assertTrue(delayer.skip_first_delayer)

        # The bypass is still available to the slot condition that owns it.
        delayer._curr_state = None
        out = self.negotiate(
            delayer,
            running_batch=50,
            max_prefill_bs=16,
            max_running_requests=60,
            waiting_queue_len=99,
        )
        self.assertTrue(out.output_allow)
        self.assertFalse(delayer.skip_first_delayer)

    def test_queue_delay_is_bounded_by_wall_clock_not_passes(self):
        # Forward passes take milliseconds, so max_delay_passes would end a
        # queue delay long before max_delay_ms does.
        delayer = self.make_delayer(max_delay_passes=3)
        delayer.skip_first_delayer = False
        for _ in range(8):
            out = self.negotiate(
                delayer, running_batch=50, max_prefill_bs=16, waiting_queue_len=2
            )
            self.assertFalse(out.output_allow)
        self.assertEqual(delayer._curr_state.delayed_count, 8)

    def test_queue_delay_releases_once_max_delay_ms_elapsed(self):
        delayer = self.make_delayer(max_delay_ms=0.0)
        delayer.skip_first_delayer = False
        delayer._curr_state = _State()
        out = self.negotiate(
            delayer, running_batch=50, max_prefill_bs=16, waiting_queue_len=2
        )
        self.assertTrue(out.output_allow)

    def test_slot_only_delay_still_capped_by_passes(self):
        # Nothing changes for an engine that never enabled the queue trigger.
        delayer = self.make_delayer(queue_min_ratio=None, max_delay_passes=3)
        delayer.skip_first_delayer = False
        reasons = [
            self.negotiate(
                delayer,
                running_batch=50,
                max_prefill_bs=16,
                max_running_requests=60,
                waiting_queue_len=99,
            ).output_reason
            for _ in range(3)
        ]
        self.assertEqual(reasons, ["delay", "delay", "wait_timeout"])


WORLD_SIZE = 4
MAX_DELAY_MS = 1000.0

# 100 running requests at ratio 0.5 wait for 50 queued; 10 are waiting and the
# slot trigger stays off, so only the queue trigger can delay.
RANK_NEGOTIATE_KWARGS = dict(
    local_prefillable=True,
    token_usage=0.9,
    running_batch=100,
    max_prefill_bs=80,
    max_running_requests=1024,
    waiting_queue_len=10,
)
DELAY = (False, "delay")
RELEASE = (True, "wait_success")


def _run_rank(rank, init_file, topology, scenarios):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        with published_topology(
            ranks={"world_rank": rank},
            enable_prefill_delayer=True,
            prefill_delayer_queue_min_ratio=0.5,
            prefill_delayer_max_delay_ms=MAX_DELAY_MS,
            **topology,
        ):
            for expired_ranks, expected in scenarios:
                delayer = PrefillDelayer(
                    cpu_group=torch.distributed.group.WORLD,
                    max_delay_passes=100,
                    token_usage_low_watermark=None,
                )
                delayer.skip_first_delayer = False
                first = delayer._negotiate_should_allow_prefill(**RANK_NEGOTIATE_KWARGS)
                assert (first.output_allow, first.output_reason) == DELAY, first

                # Age this rank's timer past the deadline, or keep it fresh.
                age_s = 2 * MAX_DELAY_MS / 1000.0 if rank in expired_ranks else 0.0
                delayer._curr_state = dataclasses.replace(
                    delayer._curr_state, start_time=time.perf_counter() - age_s
                )
                out = delayer._negotiate_should_allow_prefill(**RANK_NEGOTIATE_KWARGS)

                outcomes = [None] * WORLD_SIZE
                torch.distributed.all_gather_object(
                    outcomes, (out.output_allow, out.output_reason)
                )
                assert outcomes == [expected] * WORLD_SIZE, (
                    f"expired ranks {sorted(expired_ranks)}: {outcomes}"
                )
    finally:
        torch.distributed.destroy_process_group()


class TestPrefillDelayerQueueTimeoutRanks(CustomTestCase):
    def _spawn(self, topology, scenarios):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_rank,
                args=(str(Path(directory) / "gloo-init"), topology, scenarios),
                nprocs=WORLD_SIZE,
                join=True,
            )

    def test_tp_ranks_follow_tp0_deadline(self):
        self._spawn(
            dict(tp_size=WORLD_SIZE),
            [
                ({WORLD_SIZE - 1}, DELAY),
                ({0}, RELEASE),
                (set(range(WORLD_SIZE)), RELEASE),
            ],
        )

    def test_dp_attention_ranks_follow_each_group_leader(self):
        # Two DP groups of two ranks; ranks 0 and 2 lead them.
        self._spawn(
            dict(tp_size=WORLD_SIZE, attn_dp_size=2),
            [
                ({1, 3}, DELAY),
                ({2}, RELEASE),
            ],
        )


if __name__ == "__main__":
    unittest.main()
