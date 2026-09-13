import unittest

from sglang.srt.managers.scheduler_components.memory_usage import (
    build_memory_usage,
    combine_graph_memory_usage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerMemoryUsage(CustomTestCase):
    def test_combine_graph_memory_usage_sums_target_and_draft(self):
        target = {"prefill": 1.25, "decode": 2.0}
        draft = {"decode": 0.5, "draft_decode": 0.75}

        combined = combine_graph_memory_usage(target, draft)

        self.assertEqual(
            combined,
            {
                "prefill": 1.25,
                "decode": 2.5,
                "target_verify": 0.0,
                "draft_prefill": 0.0,
                "draft_decode": 0.75,
                "draft_extend": 0.0,
            },
        )
        self.assertEqual(target, {"prefill": 1.25, "decode": 2.0})
        self.assertEqual(draft, {"decode": 0.5, "draft_decode": 0.75})

    def test_combine_graph_memory_usage_handles_missing_inputs(self):
        self.assertEqual(
            combine_graph_memory_usage(None, None),
            {
                "prefill": 0.0,
                "decode": 0.0,
                "target_verify": 0.0,
                "draft_prefill": 0.0,
                "draft_decode": 0.0,
                "draft_extend": 0.0,
            },
        )

    def test_build_memory_usage_normalizes_public_payload(self):
        for token_capacity_swa, expected_swa in ((None, None), (2048, 2048)):
            with self.subTest(token_capacity_swa=token_capacity_swa):
                usage = build_memory_usage(
                    weight_gb=10.1236,
                    kv_cache_gb=2.3454,
                    startup_available_gb=16.9999,
                    token_capacity=4096,
                    token_capacity_swa=token_capacity_swa,
                    target_graph_memory_usage={
                        "prefill": 1.2349,
                        "decode": 0.5,
                    },
                    draft_graph_memory_usage={
                        "decode": 0.2504,
                        "draft_decode": 0.3336,
                    },
                )

                self.assertEqual(
                    usage,
                    {
                        "weight": 10.124,
                        "kvcache": 2.345,
                        "startup_available": 17.0,
                        "token_capacity": 4096,
                        "token_capacity_swa": expected_swa,
                        "graph": {
                            "prefill": 1.235,
                            "decode": 0.75,
                            "target_verify": 0.0,
                            "draft_prefill": 0.0,
                            "draft_decode": 0.334,
                            "draft_extend": 0.0,
                        },
                    },
                )


if __name__ == "__main__":
    unittest.main()
