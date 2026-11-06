"""/health_generate passes only on a completed-inference acknowledgement.

The handler registers its probe with the tokenizer manager and waits for a
HealthCheckOutput that names the probe rid and carries the scheduler's
incarnation and a completion sequence newer than the baseline recorded when
the probe was registered. Any other message, including a legacy empty ack, a
stale replay or an ack for another probe, leaves the probe failing.

It also must not leave its abandoned probe's exception unretrieved: the cancel
can lose against _stream_one_response's wait_for timeout, and nothing awaits
the task (#1327).
"""

import asyncio
import gc
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

DISCONNECT_ERROR = "Request is disconnected from the client side (type 1)."
INCARNATION = "scheduler-a"


def _ack(rid, seq=1, incarnation=INCARNATION, dp_rank=0):
    return HealthCheckOutput(
        rid=rid, incarnation=incarnation, completion_seq=seq, dp_rank=dp_rank
    )


def _probe_that_acks(make_ack, *, lose_cancel=False, bump_tstamp=False):
    """generate_request stand-in: delivers make_ack(rid) like handle_loop would."""

    async def generate_request(obj, request=None):
        tokenizer_manager = http_server.get_global_state().tokenizer_manager
        await asyncio.sleep(0)
        if bump_tstamp:
            tokenizer_manager.last_receive_tstamp = time.time() + 60
        ack = make_ack(obj.rid)
        if ack is not None:
            tokenizer_manager._handle_health_check_output(ack)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if lose_cancel:
                # The cancel arrives as _stream_one_response's wait_for times
                # out: the task ends with the disconnect error instead.
                raise ValueError(DISCONNECT_ERROR)
            raise
        yield

    return generate_request


class TestHealthGenerateCompletionAck(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="tokenizer"))

    def call_health_generate(
        self, generate_request, *, health_progress=None, timeout=0.5
    ):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        tokenizer_manager.gracefully_exit = False
        tokenizer_manager.server_status = ServerStatus.Up
        tokenizer_manager.is_generation = True
        tokenizer_manager.last_receive_tstamp = 0.0
        tokenizer_manager.health_progress = dict(health_progress or {})
        tokenizer_manager.pending_health_probes = {}
        tokenizer_manager.rid_to_state = {}
        tokenizer_manager.generate_request = generate_request
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        unretrieved = []

        async def run():
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(
                lambda _loop, context: unretrieved.append(context["message"])
            )
            response = await http_server.health_generate(request)
            # Let the abandoned probe finish, then collect it: asyncio reports an
            # unretrieved exception when the finished task is destroyed.
            for _ in range(5):
                await asyncio.sleep(0)
            gc.collect()
            return response

        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        try:
            with patch.object(http_server, "HEALTH_CHECK_TIMEOUT", timeout):
                response = asyncio.run(run())
        finally:
            http_server._global_state = prior_state
        self.assertEqual(tokenizer_manager.pending_health_probes, {})
        self.assertEqual(tokenizer_manager.rid_to_state, {})
        return response, tokenizer_manager, unretrieved

    def test_completion_ack_for_the_probe_passes(self):
        response, tokenizer_manager, unretrieved = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(rid, seq=7))
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.Up)
        self.assertEqual(tokenizer_manager.health_progress, {0: (INCARNATION, 7)})
        self.assertEqual(unretrieved, [])

    def test_any_received_message_alone_fails(self):
        # The old predicate: abort/control/error traffic bumps the receive
        # timestamp while the model is stalled.
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: None, bump_tstamp=True)
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.UnHealthy)

    def test_legacy_empty_ack_fails(self):
        response, _, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: HealthCheckOutput(rid=rid))
        )
        self.assertEqual(response.status_code, 503)

    def test_ack_for_another_probe_fails(self):
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(f"{rid}_sibling", seq=9))
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tokenizer_manager.health_progress, {})

    def test_stale_ack_from_same_incarnation_fails(self):
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(rid, seq=5)),
            health_progress={0: (INCARNATION, 5)},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tokenizer_manager.health_progress, {0: (INCARNATION, 5)})

    def test_newer_seq_from_same_incarnation_passes(self):
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(rid, seq=6)),
            health_progress={0: (INCARNATION, 5)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.health_progress, {0: (INCARNATION, 6)})

    def test_restarted_scheduler_with_low_seq_passes(self):
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(rid, seq=1, incarnation="scheduler-b")),
            health_progress={0: (INCARNATION, 500)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.health_progress, {0: ("scheduler-b", 1)})

    def test_other_worker_progress_does_not_stale_this_ack(self):
        response, tokenizer_manager, _ = self.call_health_generate(
            _probe_that_acks(lambda rid: _ack(rid, seq=1, dp_rank=1)),
            health_progress={0: (INCARNATION, 500)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            tokenizer_manager.health_progress,
            {0: (INCARNATION, 500), 1: (INCARNATION, 1)},
        )

    def test_decode_probe_forces_a_real_decode_step(self):
        seen = []

        async def generate_request(obj, request=None):
            seen.append(obj)
            http_server.get_global_state().tokenizer_manager._handle_health_check_output(
                _ack(obj.rid)
            )
            await asyncio.Event().wait()
            yield

        with patch.object(
            http_server,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_mode=DisaggregationMode.DECODE.value
            ),
        ):
            response, _, _ = self.call_health_generate(generate_request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen[0].sampling_params["max_new_tokens"], 2)
        self.assertEqual(seen[0].bootstrap_room, 0)

    def test_non_disagg_probe_generates_one_token(self):
        seen = []

        async def generate_request(obj, request=None):
            seen.append(obj)
            http_server.get_global_state().tokenizer_manager._handle_health_check_output(
                _ack(obj.rid)
            )
            await asyncio.Event().wait()
            yield

        self.call_health_generate(generate_request)
        self.assertEqual(seen[0].sampling_params["max_new_tokens"], 1)
        self.assertIsNone(seen[0].bootstrap_host)

    def test_healthy_probe_whose_cancel_is_lost_logs_nothing(self):
        with self.assertLogs(http_server.logger, level="DEBUG") as logs:
            # assertLogs requires one record; keep the evidence below in order.
            http_server.logger.debug("start")
            response, tokenizer_manager, unretrieved = self.call_health_generate(
                _probe_that_acks(lambda rid: _ack(rid), lose_cancel=True)
            )

        self.assertEqual(unretrieved, [])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.Up)
        self.assertTrue(
            any(
                "Abandoned health check probe ended with ValueError" in line
                and DISCONNECT_ERROR in line
                for line in logs.output
            ),
            logs.output,
        )

    def test_timed_out_probe_whose_cancel_is_lost_logs_nothing(self):
        response, tokenizer_manager, unretrieved = self.call_health_generate(
            _probe_that_acks(lambda rid: None, lose_cancel=True)
        )

        self.assertEqual(unretrieved, [])
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.UnHealthy)

    def test_cancelled_probe_is_silent(self):
        with self.assertNoLogs(http_server.logger, level="DEBUG"):
            response, _, unretrieved = self.call_health_generate(
                _probe_that_acks(lambda rid: _ack(rid))
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(unretrieved, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
