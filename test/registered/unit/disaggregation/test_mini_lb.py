"""Unit tests for srt/disaggregation/mini_lb"""

import asyncio
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation import mini_lb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DECODE_ARGS = ["--decode", "http://d:30001"]


def _make_role_switch_load_balancer(worker_url):
    load_balancer = mini_lb.MiniLoadBalancer.__new__(mini_lb.MiniLoadBalancer)
    load_balancer.timeout = 1
    load_balancer.prefill_urls = [worker_url]
    load_balancer.prefill_bootstrap_ports = [8998]
    load_balancer.decode_urls = ["http://decode:8000"]
    return load_balancer


class TestMiniLBArgs(CustomTestCase):
    def test_prefill_bootstrap_port(self):
        cases = [
            (["--prefill", "http://p:30000"], [("http://p:30000", None)]),
            (["--prefill", "http://p:30000", "none"], [("http://p:30000", None)]),
            (["--prefill", "http://p:30000", "8998"], [("http://p:30000", 8998)]),
        ]
        for prefill_args, expected in cases:
            with self.subTest(prefill_args=prefill_args):
                args = mini_lb.parse_args(prefill_args + DECODE_ARGS)
                load_balancer = mini_lb.MiniLoadBalancer(args)
                self.assertEqual(
                    list(
                        zip(
                            load_balancer.prefill_urls,
                            load_balancer.prefill_bootstrap_ports,
                        )
                    ),
                    expected,
                )
                self.assertEqual(load_balancer.decode_urls, ["http://d:30001"])

    def test_bad_arguments_exit(self):
        cases = [
            ["--prefill", "http://p:30000", "port"] + DECODE_ARGS,
            ["--prefill", "http://p:30000", "1", "2"] + DECODE_ARGS,
            ["--prefill", "http://p:30000"],
            DECODE_ARGS,
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                mini_lb.parse_args(argv)


class TestMiniLBRoleSwitch(CustomTestCase):
    def test_failed_role_switch_restores_only_healthy_worker(self):
        cases = [
            (
                {
                    "success": False,
                    "message": "instance is not idle",
                    "safe_to_restore": True,
                },
                True,
            ),
            (
                {
                    "success": False,
                    "message": "instance unhealthy, restart required",
                    "safe_to_restore": False,
                },
                False,
            ),
            ({"success": False, "message": "connection lost"}, False),
        ]
        for result, restored in cases:
            with self.subTest(result=result):
                worker_url = "http://prefill:8000"
                load_balancer = _make_role_switch_load_balancer(worker_url)

                async def post_role_switch(*_args, _result=result, **_kwargs):
                    return 400, _result

                with (
                    patch.object(mini_lb, "lb", load_balancer),
                    patch.object(mini_lb, "_post_role_switch", post_role_switch),
                ):
                    response = asyncio.run(
                        mini_lb.pd_role_switch(
                            {
                                "worker_url": worker_url,
                                "new_role": "decode",
                                "drain_timeout_secs": 0,
                            }
                        )
                    )

                self.assertEqual(response.status_code, 400)
                self.assertIs(worker_url in load_balancer.prefill_urls, restored)

    def test_role_switch_forwards_decode_graph_requirements(self):
        worker_url = "http://prefill:8000"
        load_balancer = _make_role_switch_load_balancer(worker_url)
        sent_body = {}

        async def post_role_switch(_worker_url, body):
            sent_body.update(body)
            return 200, {"success": True, "message": "ok"}

        with (
            patch.object(mini_lb, "lb", load_balancer),
            patch.object(mini_lb, "_post_role_switch", post_role_switch),
        ):
            response = asyncio.run(
                mini_lb.pd_role_switch(
                    {
                        "worker_url": worker_url,
                        "new_role": "decode",
                        "decode_cuda_graph_bs": [1, 2, 4],
                        "decode_cuda_graph_memory_gb": 1.25,
                    }
                )
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(sent_body["decode_cuda_graph_bs"], [1, 2, 4])
        self.assertEqual(sent_body["decode_cuda_graph_memory_gb"], 1.25)


if __name__ == "__main__":
    unittest.main()
