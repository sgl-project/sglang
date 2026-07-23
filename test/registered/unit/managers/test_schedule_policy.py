import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class TestSchedulePolicy(unittest.TestCase):
    def test_compute_prefix_matches_refreshes_cache_protected_len(self):
        tree_cache = RadixCache.create_simulated(page_size=1)
        policy = SchedulePolicy(
            policy="lpm",
            tree_cache=tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        req = Req(
            rid="req-1",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.init_next_round_input(tree_cache=tree_cache, cow_mamba=False)
        self.assertEqual(req.cache_protected_len, 0)

        tree_cache.insert(InsertParams(key=RadixKey([1, 2, 3, 4])))

        policy._compute_prefix_matches([req], CacheAwarePolicy.LPM)

        self.assertEqual(len(req.prefix_indices), 4)
        self.assertEqual(req.cache_protected_len, 4)


if __name__ == "__main__":
    unittest.main()
