"""CPU regressions for POST /vertex_generate request bounds (#52345).

One oversized request fanned every instance out to its own scheduler
forward, each holding a ZMQ REQ socket open, until the HTTP process hit
"Too many open files" and degraded /health and the other endpoints. The
endpoint now rejects oversized batches with HTTP 413 and bounds in-flight
forwards with a semaphore.
"""

import asyncio
import itertools
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VertexGenerateReqInput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ConcurrencyTracker:
    """Fake scheduler forward that records peak in-flight concurrency."""

    def __init__(self):
        self.current = 0
        self.max_seen = 0

    async def __call__(self, req, sp):
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        try:
            await asyncio.sleep(0)
            return f"pred-{req.sampling_params.seq}"
        finally:
            self.current -= 1


class TestVertexGenerateBounds(CustomTestCase):
    def setUp(self):
        # The semaphore is cached at module level; drop it so each test reads
        # the env var it sets.
        http_server._vertex_forward_semaphore = None

    def tearDown(self):
        http_server._vertex_forward_semaphore = None

    def test_instances_at_default_limit_pass(self):
        http_server._validate_vertex_instances([{}] * 64)
        http_server._validate_vertex_instances([])

    def test_instances_over_default_limit_rejected_with_413(self):
        with self.assertRaises(HTTPException) as ctx:
            http_server._validate_vertex_instances([{}] * 65)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("65", ctx.exception.detail)
        self.assertIn("64", ctx.exception.detail)

    def test_env_var_controls_instance_limit(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_DIFFUSION_MAX_VERTEX_INSTANCES": "2"}
        ):
            http_server._validate_vertex_instances([{}] * 2)
            with self.assertRaises(HTTPException) as ctx:
                http_server._validate_vertex_instances([{}] * 3)
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertIn("2", ctx.exception.detail)

    def test_non_positive_limits_clamp_to_one(self):
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_DIFFUSION_MAX_VERTEX_INSTANCES": "0",
                "SGLANG_DIFFUSION_VERTEX_MAX_CONCURRENCY": "-5",
            },
        ):
            self.assertEqual(http_server._vertex_instance_limit(), 1)
            self.assertEqual(http_server._vertex_concurrency_limit(), 1)
            http_server._validate_vertex_instances([{}])
            with self.assertRaises(HTTPException):
                http_server._validate_vertex_instances([{}, {}])

    def test_semaphore_bounds_concurrent_forwards(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_DIFFUSION_VERTEX_MAX_CONCURRENCY": "3"}
        ):
            semaphore = http_server._get_vertex_forward_semaphore()
            self.assertIs(semaphore, http_server._get_vertex_forward_semaphore())

            counter = itertools.count()
            tracker = _ConcurrencyTracker()

            async def run():
                reqs = [
                    SimpleNamespace(sampling_params=SimpleNamespace(seq=next(counter)))
                    for _ in range(10)
                ]
                with mock.patch.object(http_server, "forward_to_scheduler", tracker):
                    return await asyncio.gather(
                        *(
                            http_server._forward_vertex_request(
                                semaphore, req, req.sampling_params
                            )
                            for req in reqs
                        )
                    )

            results = asyncio.run(run())
            self.assertEqual(tracker.max_seen, 3)
            self.assertEqual(results, [f"pred-{i}" for i in range(10)])

    def test_handler_rejects_oversized_batch_before_dispatch(self):
        req = VertexGenerateReqInput(instances=[{"prompt": "a cat"}] * 65)
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(http_server.vertex_generate(req))
        self.assertEqual(ctx.exception.status_code, 413)

    def test_handler_returns_empty_predictions_for_empty_instances(self):
        resp = asyncio.run(
            http_server.vertex_generate(VertexGenerateReqInput(instances=[]))
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(resp.body), {"predictions": []})

    def test_handler_dispatches_bounded(self):
        counter = itertools.count()
        tracker = _ConcurrencyTracker()

        def fake_build_sampling_params(rid, **kwargs):
            return SimpleNamespace(seq=next(counter))

        def fake_prepare_request(server_args, sampling_params=None):
            return SimpleNamespace(sampling_params=sampling_params)

        req = VertexGenerateReqInput(
            instances=[{"prompt": "a cat"}] * 5, parameters={"width": 256}
        )
        with (
            mock.patch.dict(
                os.environ, {"SGLANG_DIFFUSION_VERTEX_MAX_CONCURRENCY": "2"}
            ),
            mock.patch.object(
                http_server,
                "get_global_server_args",
                lambda: SimpleNamespace(),
            ),
            mock.patch.object(
                http_server, "build_sampling_params", fake_build_sampling_params
            ),
            mock.patch.object(http_server, "prepare_request", fake_prepare_request),
            mock.patch.object(http_server, "forward_to_scheduler", tracker),
        ):
            resp = asyncio.run(http_server.vertex_generate(req))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            json.loads(resp.body)["predictions"],
            [f"pred-{i}" for i in range(5)],
        )
        self.assertEqual(tracker.max_seen, 2)


if __name__ == "__main__":
    unittest.main()
