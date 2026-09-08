"""The opt-in JSON warmup must drain responses and surface startup failures."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.warmup import execute_warmups
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestStructuredOutputWarmup(CustomTestCase):
    def test_request_and_generator_completion(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                requests = []
                completed = []

                async def generate(req, request):
                    requests.append(req)
                    yield {"text": "partial"}
                    yield {"text": "complete"}
                    completed.append(True)

                manager = SimpleNamespace(generate_request=generate)
                asyncio.run(execute_warmups(mode, ["structured_output"], manager))
                self.assertEqual(completed, [True])
                self.assertEqual(len(requests), 1)
                req = requests[0]
                self.assertEqual(
                    json.loads(req.sampling_params["json_schema"])["type"], "object"
                )
                self.assertGreater(req.sampling_params["max_new_tokens"], 0)
                if mode != "null":
                    self.assertEqual(req.bootstrap_room, 0)
                    self.assertEqual(req.bootstrap_host, FAKE_BOOTSTRAP_HOST)

    def test_failure_after_first_response_propagates(self):
        async def generate(req, request):
            yield {"text": "partial"}
            raise RuntimeError("warmup failed")

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            asyncio.run(
                execute_warmups(
                    "null",
                    ["structured_output"],
                    SimpleNamespace(generate_request=generate),
                )
            )


if __name__ == "__main__":
    unittest.main()
