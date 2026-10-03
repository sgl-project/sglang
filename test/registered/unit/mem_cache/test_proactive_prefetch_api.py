"""HTTP control transport is independent of generation admission."""

import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import ProactivePrefetchReqOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProactiveAPI(unittest.TestCase):
    def test_exact_prefix_and_cancel_use_only_control_transport(self):
        received = []

        async def control(obj):
            received.append(obj)
            return ProactivePrefetchReqOutput(success=True, result={"state": "RUNNING"})

        state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(proactive_prefetch=control)
        )
        with mock.patch.object(http_server, "_global_state", state):
            client = TestClient(http_server.app)
            response = client.post(
                "/hicache/prefetch",
                json={
                    "operation_id": "tool-gap",
                    "input_ids": [1, 2, 3, 4],
                    "cache_salt": "session",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(received[0].input_ids, [1, 2, 3, 4])
            self.assertEqual(received[0].cache_salt, "session")
            response = client.post(
                "/hicache/prefetch",
                json={"operation_id": "tool-gap", "action": "cancel"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(received[1].action, "cancel")
            self.assertIsNone(received[1].input_ids)

    def test_unsupported_control_is_reported_as_bad_request(self):
        async def control(obj):
            return ProactivePrefetchReqOutput(
                success=False, message="Single worker required"
            )

        state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(proactive_prefetch=control)
        )
        with mock.patch.object(http_server, "_global_state", state):
            response = TestClient(http_server.app).post(
                "/hicache/prefetch", json={"operation_id": "p", "input_ids": [1]}
            )
            self.assertEqual(response.status_code, 400)
            self.assertFalse(response.json()["success"])

    def test_multiworker_rejects_before_control_fanout(self):
        import asyncio

        from sglang.srt.managers.io_struct import ProactivePrefetchReqInput
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

        parallel = SimpleNamespace(
            tp_size=2, pp_size=1, dp_size=1, nnodes=1, attn_cp_size=1, attn_dp_size=1
        )
        with mock.patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_parallel",
            return_value=parallel,
        ):
            result = asyncio.run(
                TokenizerControlMixin.proactive_prefetch(
                    SimpleNamespace(),
                    ProactivePrefetchReqInput(operation_id="p", input_ids=[1]),
                )
            )
            self.assertFalse(result.success)

    def test_scheduler_control_creates_no_generation_request(self):
        import test_prefetch_finite_io as finite

        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.io_struct import ProactivePrefetchReqInput
        from sglang.srt.managers.scheduler import Scheduler

        fixture = finite.TestFiniteIO(
            methodName="test_finite_read_exception_worker_survives"
        )
        fixture.setUp()
        try:
            scheduler = SimpleNamespace(
                tree_cache=fixture.cache,
                _engine_paused=False,
                max_req_input_len=64,
                model_config=SimpleNamespace(
                    vocab_size=128, is_multimodal=False, is_generation=True
                ),
                disaggregation_mode=DisaggregationMode.NULL,
            )
            parallel = SimpleNamespace(
                tp_size=1,
                pp_size=1,
                dp_size=1,
                nnodes=1,
                attn_cp_size=1,
                attn_dp_size=1,
            )
            with (
                mock.patch(
                    "sglang.srt.managers.scheduler.get_parallel", return_value=parallel
                ),
                mock.patch(
                    "sglang.srt.managers.scheduler.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=None),
                ),
                mock.patch(
                    "sglang.srt.managers.scheduler.get_lora",
                    return_value=SimpleNamespace(enable_lora=False),
                ),
                mock.patch(
                    "sglang.srt.managers.scheduler.Req",
                    side_effect=AssertionError("control must not construct Req"),
                ),
            ):
                result = Scheduler.handle_proactive_prefetch(
                    scheduler,
                    ProactivePrefetchReqInput(
                        operation_id="p", input_ids=list(fixture.tokens)
                    ),
                )
                self.assertTrue(result.success, result.message)
                fixture.pump_until(lambda: not fixture.cache.ongoing_prefetch)
                scheduler.proactive_prefetch.tick()
                self.assertEqual(
                    scheduler.proactive_prefetch.status("p")["restored_tokens"], 12
                )
                fixture.conservation(
                    scheduler.proactive_prefetch.records["p"].handle, resident=12
                )
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
