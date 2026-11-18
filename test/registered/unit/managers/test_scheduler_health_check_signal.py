"""A busy scheduler acknowledges the health probes it skipped after each step,
and the probes already waiting once other work exists.

The Python frontend reads the acknowledgement from the zmq tokenizer socket.
The embedded Rust server does not consume that socket, so in Rust mode the
acknowledgement goes on its response ring instead.
"""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # noqa: E402
from sglang.srt.disaggregation.fake.conn import FakeKVSender  # noqa: E402
from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator  # noqa: E402
from sglang.srt.managers.io_struct import HealthCheckOutput  # noqa: E402
from sglang.srt.managers.req_types import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.request_list import (  # noqa: E402
    move,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.test.scheduler_test_utils import (  # noqa: E402
    empty_request_lists,
    release_only_retirer,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHealthCheckSignal(CustomTestCase):
    def _scheduler(self, rust_server, pending) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.rust_server = rust_server
        s.ipc_channels = MagicMock()
        s.return_health_check_ipcs = deque(pending)
        return s

    def test_rust_mode_acknowledges_all_skipped_probes_on_the_ring(self):
        server = MagicMock()
        s = self._scheduler(RustServer(server, http_port=0), [None, None])

        s.maybe_send_health_check_signal()
        s.maybe_send_health_check_signal()

        server.push_health_ack.assert_called_once_with()
        self.assertFalse(s.return_health_check_ipcs)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_python_mode_acknowledges_one_probe_per_step_on_zmq(self):
        s = self._scheduler(None, ["ipc-a", "ipc-b"])

        s.maybe_send_health_check_signal()

        (output,), _ = s.ipc_channels.send_to_tokenizer.send_output.call_args
        self.assertIsInstance(output, HealthCheckOutput)
        self.assertEqual(output.http_worker_ipc, "ipc-a")
        self.assertEqual(list(s.return_health_check_ipcs), ["ipc-b"])


class _PriorityPassReached(Exception):
    pass


_ADMISSION_ENTRY_POINTS = {
    "prefill": lambda s, batch: s._get_new_batch_prefill_raw(
        prefill_delayer_single_pass=None, running_batch=batch
    ),
    "dllm": lambda s, batch: s.get_new_batch_dllm(batch, []),
}


class TestWaitingHealthCheckAck(CustomTestCase):
    def _req(self, rid):
        return Req(
            rid, "", array("q", [1]), SamplingParams(), http_worker_ipc=f"ipc-{rid}"
        )

    def _prompt_and_probe(self):
        return self._req("prompt"), self._req(f"{HEALTH_CHECK_RID_PREFIX}_probe")

    def _scheduler(self, waiting, disaggregation_mode=DisaggregationMode.NULL):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = disaggregation_mode
        empty_request_lists(s)
        move(waiting, s.waiting_queue)
        s.return_health_check_ipcs = deque()
        s.tree_cache = MagicMock()
        # Releases only: an abort sent to the tokenizer for a probe would raise.
        s.request_retirer = release_only_retirer(tree_cache=s.tree_cache)
        return s

    def test_probe_behind_a_waiting_request_is_acknowledged(self):
        """A probe queued behind another request is acknowledged, not left queued.

        Run as a request, a probe is answered only once scheduled; a long prompt
        ahead of it fills every prefill pass until its last chunk, so the
        caller's health check times out.
        """
        prompt, probe = self._prompt_and_probe()
        s = self._scheduler([prompt, probe])

        s._ack_waiting_health_checks()

        self.assertEqual(list(s.waiting_queue), [prompt])
        self.assertEqual(list(s.return_health_check_ipcs), [probe.http_worker_ipc])
        self.assertIsInstance(probe.finished_reason, FINISH_ABORT)
        s.tree_cache.finish.assert_called_once_with(
            probe.cache_request_handle, CacheRequestOutcome.ABORT
        )
        self.assertIsNone(prompt.finished_reason)

    def test_lone_probe_runs_unless_a_chunked_prefill_is_in_flight(self):
        """A probe alone in the queue runs, unless a chunked prefill is in flight.

        Its forward pass is what shows the engine can still run one. Behind a
        chunked prefill it would wait out the remaining chunks and the caller's
        health check times out; a running batch alone does not fill the chunk budget.
        """
        # (chunked prefill in flight, running batch, probe acknowledged)
        cases = {
            "idle": (False, False, False),
            "chunked": (True, False, True),
            "running": (False, True, False),
        }
        for name, (chunked, running, acknowledged) in cases.items():
            with self.subTest(name):
                _, probe = self._prompt_and_probe()
                s = self._scheduler([probe])
                if chunked:
                    move([self._req("chunk")], s.chunked_reqs)
                s.running_batch = ScheduleBatch(
                    reqs=[self._req("running")] if running else []
                )

                s._ack_waiting_health_checks()

                self.assertEqual(list(s.waiting_queue), [] if acknowledged else [probe])
                self.assertEqual(
                    list(s.return_health_check_ipcs),
                    [probe.http_worker_ipc] if acknowledged else [],
                )
                self.assertEqual(
                    isinstance(probe.finished_reason, FINISH_ABORT), acknowledged
                )

    def test_pd_prefill_probe_returns_its_metadata_buffer(self):
        """A PD prefill probe taken out of the queue frees its metadata buffer slot.

        A probe takes a slot when it leaves the bootstrap queue; a slot never
        returned shrinks the pool until no request can leave bootstrap.
        """
        prompt, probe = self._prompt_and_probe()
        s = self._scheduler([prompt, probe], DisaggregationMode.PREFILL)
        s.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(2)
        s.disagg_prefill_pending_chunk_rids = set()
        for req in (prompt, probe):
            req.metadata_buffer_index = s.req_to_metadata_buffer_idx_allocator.alloc()
            req.disagg_kv_sender = FakeKVSender(
                mgr=None, bootstrap_addr="fake:0", bootstrap_room=0
            )
        prompt_slot = prompt.metadata_buffer_index

        s._ack_waiting_health_checks()

        self.assertEqual(probe.metadata_buffer_index, -1)
        self.assertEqual(s.req_to_metadata_buffer_idx_allocator.available_size(), 1)
        self.assertEqual(prompt.metadata_buffer_index, prompt_slot)

    def _admission(self, waiting):
        # Admission passes stop at calc_priority; `ranked` records their queues.
        s = self._scheduler(waiting)
        s.grammar_manager.has_waiting_grammars = lambda: False
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 1
        s.processed_tokens_counter = 0
        s.dllm_manager = SimpleNamespace(is_empty=lambda: True)
        ranked = []

        def calc_priority(waiting, *args, **kwargs):
            ranked.append(list(waiting))
            raise _PriorityPassReached

        s.policy = SimpleNamespace(
            poll_storage_prefetches=lambda waiting: None, calc_priority=calc_priority
        )
        return s, ranked

    def test_each_admission_entry_point_acknowledges_waiting_probes(self):
        """Prefill and dLLM admission both acknowledge a probe before ranking.

        An entry point that skips this leaves the probe queued behind a long
        prompt. dLLM branches off before prefill admission, and some callers run
        _get_new_batch_prefill_raw without get_new_batch_prefill.
        """
        for name, schedule in _ADMISSION_ENTRY_POINTS.items():
            with self.subTest(entry_point=name):
                prompt, probe = self._prompt_and_probe()
                s, ranked = self._admission([prompt, probe])

                with self.assertRaises(_PriorityPassReached):
                    schedule(s, ScheduleBatch(reqs=[]))

                self.assertEqual(ranked, [[prompt]])
                self.assertEqual(
                    list(s.return_health_check_ipcs), [probe.http_worker_ipc]
                )

    def test_full_batch_pass_does_not_scan_the_waiting_queue(self):
        """A pass that stops at a full running batch leaves waiting probes alone.

        Decode runs such a pass every step; scanning a long backlog for probes
        there slows every decode step, and the pass admits nothing anyway.
        """
        for name, schedule in _ADMISSION_ENTRY_POINTS.items():
            with self.subTest(entry_point=name):
                prompt, probe = self._prompt_and_probe()
                s, ranked = self._admission([prompt, probe])
                batch = ScheduleBatch(reqs=[self._req("running")], batch_is_full=True)

                schedule(s, batch)

                self.assertEqual(ranked, [])
                self.assertEqual(list(s.waiting_queue), [prompt, probe])
                self.assertFalse(s.return_health_check_ipcs)


if __name__ == "__main__":
    unittest.main()
