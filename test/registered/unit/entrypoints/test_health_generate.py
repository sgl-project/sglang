import asyncio
import time
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []
        self.gracefully_exit = False
        self.server_status = ServerStatus.Up
        self.is_generation = True
        self.elastic_worker_count = 4
        self.rid_to_state = {}
        self.last_receive_tstamp = time.time()
        self.server_args = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL.value
        )

    async def generate_request(self, request, raw_request):
        del raw_request
        self.requests.append(request)
        for output in self.outputs(request):
            yield output


def _request(dp_rank=None, path="/health_generate"):
    query_params = {} if dp_rank is None else {"dp_rank": str(dp_rank)}
    return SimpleNamespace(
        url=SimpleNamespace(path=path),
        query_params=query_params,
    )


def _call_health_generate(manager, request):
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        return asyncio.run(http_server.health_generate(request))
    finally:
        http_server._global_state = prior_state


class TestHealthGenerate(unittest.TestCase):
    def test_targets_requested_dp_rank_and_waits_for_exact_rid(self):
        def outputs(request):
            yield {
                "meta_info": {
                    "id": request.rid,
                    "dp_rank": request.routed_dp_rank,
                }
            }

        manager = _FakeTokenizerManager(outputs)
        response = _call_health_generate(manager, _request(dp_rank=2))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(manager.requests), 1)
        probe = manager.requests[0]
        self.assertIsInstance(probe, GenerateReqInput)
        self.assertEqual(probe.routed_dp_rank, 2)
        self.assertFalse(probe.log_metrics)
        self.assertTrue(probe.no_logs)

    def test_rank_target_forces_generation_on_lightweight_health_path(self):
        def outputs(request):
            yield {"meta_info": {"id": request.rid, "dp_rank": 1}}

        manager = _FakeTokenizerManager(outputs)
        response = _call_health_generate(manager, _request(dp_rank=1, path="/health"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(manager.requests[0].routed_dp_rank, 1)

    def test_unrelated_output_cannot_mask_failed_probe(self):
        def outputs(request):
            yield {"meta_info": {"id": f"other-{request.rid}", "dp_rank": 1}}

        manager = _FakeTokenizerManager(outputs)
        response = _call_health_generate(manager, _request(dp_rank=3))

        self.assertEqual(response.status_code, 503)
        # A rank-targeted probe must not flip the whole-server status:
        # concurrent per-rank probes would race on the shared field.
        self.assertEqual(manager.server_status, ServerStatus.Up)

    def test_invalid_dp_rank_is_rejected_without_generation(self):
        manager = _FakeTokenizerManager(lambda request: ())

        for value in ("not-an-int", -1, 4):
            with self.subTest(value=value):
                response = _call_health_generate(manager, _request(dp_rank=value))
                self.assertEqual(response.status_code, 400)

        self.assertEqual(manager.requests, [])

    def test_default_path_keeps_heartbeat_semantics(self):
        # Without dp_rank, any detokenizer activity within the timeout marks
        # the server healthy, even if this probe's own rid never completes.
        def outputs(request):
            yield {"meta_info": {"id": f"other-{request.rid}"}}

        manager = _FakeTokenizerManager(outputs)
        manager.last_receive_tstamp = time.time() + 3600
        response = _call_health_generate(manager, _request())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(manager.server_status, ServerStatus.Up)
        probe = manager.requests[0]
        self.assertIsNone(probe.routed_dp_rank)
        self.assertFalse(probe.no_logs)

    def test_concurrent_rank_probes_do_not_interfere(self):
        def outputs(request):
            yield {"meta_info": {"id": request.rid, "dp_rank": request.routed_dp_rank}}

        manager = _FakeTokenizerManager(outputs)

        async def run_all():
            prior_state = http_server.get_global_state()
            http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
            try:
                return await asyncio.gather(
                    *(
                        http_server.health_generate(_request(dp_rank=rank % 4))
                        for rank in range(8)
                    )
                )
            finally:
                http_server._global_state = prior_state

        responses = asyncio.run(run_all())

        self.assertEqual([r.status_code for r in responses], [200] * 8)
        self.assertEqual(len(manager.requests), 8)
        rids = {probe.rid for probe in manager.requests}
        self.assertEqual(len(rids), 8)
        self.assertEqual(
            sorted(probe.routed_dp_rank for probe in manager.requests),
            [0, 0, 1, 1, 2, 2, 3, 3],
        )


if __name__ == "__main__":
    unittest.main()
