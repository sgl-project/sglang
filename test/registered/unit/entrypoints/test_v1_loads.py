import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sglang.srt.entrypoints.v1_loads import _format_loads_prometheus, get_loads
from sglang.srt.managers.io_struct import GetLoadsReqOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestV1Loads(CustomTestCase):
    def _run_async(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _make_load(self, **kwargs) -> GetLoadsReqOutput:
        defaults = dict(
            dp_rank=0,
            timestamp=1710000000.0,
            num_running_reqs=2,
            num_waiting_reqs=1,
            num_used_tokens=256,
            max_total_num_tokens=1024,
            token_usage=0.25,
            gen_throughput=12.34,
            prefill_throughput=56.78,
            cache_hit_rate=0.5,
            utilization=0.75,
            max_running_requests=128,
        )
        defaults.update(kwargs)
        return GetLoadsReqOutput(**defaults)

    def test_prometheus_format_includes_prefill_throughput(self):
        response = _format_loads_prometheus([self._make_load()])
        content = response.body.decode()

        self.assertIn("# HELP sglang_prefill_throughput", content)
        self.assertIn("# TYPE sglang_prefill_throughput gauge", content)
        self.assertIn('sglang_prefill_throughput{dp_rank="0"} 56.78', content)

    def test_json_response_includes_prefill_throughput(self):
        tokenizer_manager = SimpleNamespace(
            get_loads=AsyncMock(
                return_value=[
                    self._make_load(dp_rank=0, prefill_throughput=78.9),
                    self._make_load(dp_rank=1, prefill_throughput=12.3),
                ]
            )
        )

        response = self._run_async(get_loads(tokenizer_manager=tokenizer_manager))

        self.assertEqual(response["dp_rank_count"], 2)
        self.assertEqual(response["loads"][0]["prefill_throughput"], 78.9)
        self.assertEqual(response["loads"][1]["prefill_throughput"], 12.3)
        self.assertEqual(response["loads"][0]["num_total_reqs"], 3)
        tokenizer_manager.get_loads.assert_awaited_once_with(
            include=None,
            dp_rank=None,
        )

    def test_prometheus_endpoint_uses_prefill_field(self):
        tokenizer_manager = SimpleNamespace(
            get_loads=AsyncMock(return_value=[self._make_load(prefill_throughput=91.2)])
        )

        response = self._run_async(
            get_loads(format="prometheus", tokenizer_manager=tokenizer_manager)
        )
        content = response.body.decode()

        self.assertIn('sglang_prefill_throughput{dp_rank="0"} 91.2', content)


if __name__ == "__main__":
    unittest.main()
