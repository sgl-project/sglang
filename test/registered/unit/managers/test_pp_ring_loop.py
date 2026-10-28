"""The PP ring driver: when each slot's step commits, and which requests a
slot's step decodes."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.managers.scheduler_assembly import init_mode_stages  # noqa: E402
from sglang.srt.managers.scheduler_components.bundles import Streams  # noqa: E402
from sglang.srt.managers.scheduler_components.engine_state import (  # noqa: E402
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.srt.managers.scheduler_components.mode_stages import (  # noqa: E402
    ModeStages,
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.pending_steps import (  # noqa: E402
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.step_planner import (  # noqa: E402
    StepPlanner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopRing(Exception):
    pass


def _step(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        reqs=[],
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        return_logprob=False,
    )


class TestPPRingCommit(CustomTestCase):
    def run_ring(self, depth: int, schedule):
        """Run the plain PP ring on the last rank over a script of steps (None:
        the slot schedules nothing); return per iteration what it launched,
        the relayed result it received, and what it committed."""
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        enter_scope(
            self,
            get_parallel().override(
                pp_size=2,
                pp_async_batch_depth=depth,
                pp_group=SimpleNamespace(is_last_rank=True),
                world_group=None,
                attn_tp_group=None,
            ),
        )
        s = partial_scheduler()
        s.parts.mode_owners = ModeOwners()
        s.parts.counters = LaunchCounters()
        s.parts.flags = EngineFlags()
        s.parts.lists = CoreLists()
        s.parts.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        s.parts.streams = Streams(
            device="cpu",
            device_module=SimpleNamespace(
                Event=Mock, current_stream=Mock(), stream=lambda stream: nullcontext()
            ),
            schedule_stream=None,
            forward_stream=None,
            copy_stream=None,
            async_d2h_worker=None,
        )
        s.parts.kv = SimpleNamespace(token_to_kv_pool_allocator=None)
        s.parts.future_map = s.parts.workers.tp_worker = None
        s.parts.pending = PendingSteps()
        s.parts.recv_skipper = None
        s.parts.pp_spec_forward = None
        s._sched_idled = False
        s.ingest_requests = Mock(return_value=[])
        s.mode.step_launcher = SimpleNamespace(
            run_batch=lambda step, pp_proxy_tensors: SimpleNamespace(launched=step.name)
        )
        s.join_launched_step = Mock()
        s.on_idle = Mock()

        trace = []
        script = iter(schedule)

        def schedule_slot(slot):
            try:
                step = next(script)
            except StopIteration:
                raise _StopRing
            trace.append(
                {"slot": slot.index, "launched": step and step.name, "committed": None}
            )
            return step

        def exchange_outputs(next_first_rank_mb_id, next_mb_id):
            trace[-1]["relayed"] = relayed = object()
            return None, relayed, Mock()

        def commit_step(record):
            trace[-1]["committed"] = record.step.name
            assert record.result is trace[-1]["relayed"]

        s.mode.stages = ModeStages(commit_extend=Mock())
        s.schedule_step = schedule_slot
        s.commit_step = commit_step
        s.init_pp_ring()
        ring = s.pp_ring
        ring.channel.commit = Mock()
        ring.channel.commit_proxy_send = Mock()
        ring.channel.recv_proxy = Mock(return_value=None)
        ring.channel.wait_forward_dependencies = Mock()
        ring._prepare_tensor_dict = Mock(return_value={})
        ring._commit_send_output_work_and_preprocess_output_tensors = exchange_outputs
        with patch("sglang.srt.managers.scheduler_components.pp_ring.set_time_batch"):
            with self.assertRaises(_StopRing):
                ring.run_ring_loop()
        return s, trace

    def test_slot_step_commits_with_its_relayed_outputs_one_round_later(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                loop_size = 2 + depth
                schedule = [_step("a"), _step("b"), None, _step("c")]
                schedule += [None] * (2 * loop_size)
                s, trace = self.run_ring(depth, schedule)

                # Each iteration schedules the slot the ring is at.
                self.assertEqual(
                    [it["slot"] for it in trace],
                    [i % loop_size for i in range(len(trace))],
                )
                # A step commits loop_size - 1 iterations after its launch,
                # with the outputs relayed in that iteration.
                launched_at = {
                    it["launched"]: i for i, it in enumerate(trace) if it["launched"]
                }
                committed_at = {
                    it["committed"]: i for i, it in enumerate(trace) if it["committed"]
                }
                self.assertEqual(
                    committed_at,
                    {name: i + loop_size - 1 for name, i in launched_at.items()},
                )
                self.assertEqual(len(s.parts.pending), 0)
                s.on_idle.assert_called()


class _Req:
    is_prefill_only = False
    beam_group = None
    ring_slot = None


class TestPPSlotTag(CustomTestCase):
    def test_slot_step_decodes_only_the_requests_its_slot_launched(self):
        """A request joins decoding tagged with the slot that launched its last
        prefill chunk; another slot's step never decodes it while that slot's
        step may still be in flight."""
        s = partial_scheduler()
        s.parts.mode_owners = ModeOwners()
        s.parts.counters = LaunchCounters()
        s.parts.flags = EngineFlags()
        s.parts.lists = CoreLists()
        s.mode.disaggregation_mode = DisaggregationMode.NULL
        s.parts.mode_owners.disagg_prefill = None
        s.parts.mode_owners.disagg_decode = None
        s.parts.mode_owners.dllm = None
        s.parts.mode_owners.hisparse_coordinator = None
        s.parts.lists.decoding = []
        s.parts.lists.waiting_queue = []
        s.parts.pending = PendingSteps()
        s.parts.flags.enable_skip_finishing_decode = False
        s.parts.batch_result_processor = Mock()
        s.mode.stages = init_mode_stages(
            s.parts, s.loop_schedules(), lora_ids_in_use=set
        )
        reqs = [_Req() for _ in range(3)]
        slots = [ScheduleSlot(index=0), ScheduleSlot(index=1)]
        for slot, launched in ((slots[0], reqs[:2]), (slots[1], reqs[2:])):
            s.join_launched_step(
                SimpleNamespace(
                    prefill_reqs=lambda launched=launched: launched, chunked_req=None
                ),
                slot,
            )

        planner = StepPlanner.__new__(StepPlanner)
        planner.core_lists, planner.pending, planner.flags = (
            s.parts.lists,
            s.parts.pending,
            s.parts.flags,
        )
        self.assertEqual(planner._select_decode_reqs(slots[0]), reqs[:2])
        self.assertEqual(planner._select_decode_reqs(slots[1]), reqs[2:])
        # Outside PP every decoding request runs.
        self.assertEqual(planner._select_decode_reqs(ScheduleSlot()), reqs)

    def test_pd_decode_admits_up_to_the_slot_cap(self):
        """PD decode admission fills the scheduled slot up to the per-slot cap,
        counting only that slot's decoding requests."""
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        s = partial_scheduler()
        s.parts.mode_owners = ModeOwners()
        s.parts.counters = LaunchCounters()
        s.parts.flags = EngineFlags()
        s.parts.lists = CoreLists()
        s.parts.lists.decoding = []
        for slot, num_reqs in ((0, 2), (1, 1)):
            for _ in range(num_reqs):
                req = _Req()
                req.ring_slot = slot
                s.parts.lists.decoding.append(req)
        ready = [Mock() for _ in range(3)]
        owner = decode_owner_shell(
            ready=list(ready),
            enable_priority_scheduling=False,
            req_to_token_pool=SimpleNamespace(size=16),
            max_running_requests=3,
            tree_cache=Mock(),
            core_lists=s.parts.lists,
        )

        self.assertEqual(owner.pop_reqs_to_admit(ScheduleSlot(index=1)), ready[:2])
        self.assertEqual(owner.ready, ready[2:])

    def test_pd_decode_consensus_budgets_by_the_slot_scheduled_next(self):
        """Between schedules the ring applies the next slot's consensus; its
        resume, preallocation and host staging budget against that slot's
        decoding requests, not those of the slot scheduled last."""
        enter_scope(self, published_topology(role="scheduler", pp_size=2))
        decoding = []
        for slot, num_reqs in ((0, 2), (1, 1)):
            for _ in range(num_reqs):
                req = _Req()
                req.ring_slot = slot
                decoding.append(req)
        budgeted = []

        def record(decoding, *args, **kwargs):
            budgeted.append(len(decoding))
            return [], []

        owner = decode_owner_shell(
            enable_priority_scheduling=False,
            enable_hisparse=False,
            core_lists=CoreLists(),
        )
        owner.core_lists.decoding = decoding
        owner.prealloc_queue.resume_retracted_reqs = lambda decoding, rids: record(
            decoding
        )[0]
        owner.prealloc_queue.pop_preallocated = record
        owner.transfer_queue.resolve_deferred_releases = Mock(return_value=[])
        owner.transfer_queue.pop_transferred = lambda decoding, rids: record(decoding)[
            0
        ]
        owner.transfer_queue.extend = Mock()

        # Slot 0 is scheduled last; slot 1 schedules next.
        owner.pop_reqs_to_admit(ScheduleSlot(index=0))
        next_slot = ScheduleSlot(index=1)
        owner.apply_retract_consensus([], next_slot)
        owner.apply_prealloc_consensus([[], []], next_slot, lora_ids_in_use=set)
        owner.apply_transfer_consensus([], next_slot)

        self.assertEqual(budgeted, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
