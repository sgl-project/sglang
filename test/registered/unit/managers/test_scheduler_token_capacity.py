import unittest

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.memory_usage import build_memory_usage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerTokenCapacity(unittest.TestCase):
    def test_init_info_exposes_logical_and_per_dcp_rank_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.logical_max_total_num_tokens = 4096
        scheduler.max_total_num_tokens_per_dcp_rank = 1024
        scheduler.max_req_input_len = 8192
        scheduler.startup_time = {"scheduler_e2e": 1.25}

        info = scheduler.get_init_info()

        self.assertEqual(info["max_total_num_tokens"], 4096)
        self.assertEqual(info["max_total_num_tokens_per_dcp_rank"], 1024)

    def test_memory_usage_exposes_logical_and_per_dcp_rank_capacity(self):
        memory_usage = build_memory_usage(
            weight_gb=10.0,
            kv_cache_gb=20.0,
            startup_available_gb=30.0,
            token_capacity=4096,
            token_capacity_per_dcp_rank=1024,
            token_capacity_swa=None,
            target_graph_memory_usage=None,
            draft_graph_memory_usage=None,
        )

        self.assertEqual(memory_usage["token_capacity"], 4096)
        self.assertEqual(memory_usage["token_capacity_per_dcp_rank"], 1024)


if __name__ == "__main__":
    unittest.main()
