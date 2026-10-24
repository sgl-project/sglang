"""Automatic speculative limits must not consume the memory reserved for KV.

With radix disabled, a default request limit must leave the configured KV
share even when its state pool fits within the total cache budget.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

# isort: split
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaSpeculativeBudget(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _planner(self, request_limit=None, *, state_limit=None):
        reset_context()
        args = ServerArgs(
            model_path="dummy",
            max_running_requests=request_limit,
            max_mamba_cache_size=state_limit,
            disable_radix_cache=True,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
            mamba_full_memory_ratio=0.9,
        )
        if request_limit is None:
            # The same declaration used by speculation; raw input stays unset.
            declare_resolution(args, "_handle_eagle_family", max_running_requests=48)
        publish(args, role="scheduler")
        params = Mamba2CacheParams(
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=list(range(36)),
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            ),
        )
        planner = object.__new__(KVCacheConfigurator)
        planner.pp_size = planner.attn_dp_size = 1
        planner.mambaish_config = SimpleNamespace(mamba2_cache_params=params)
        planner.hybrid_gdn_config = planner.mambaish_config
        planner.hybrid_kda_config = None
        planner.extra_mamba_cache_bytes_per_req = 0
        planner.spec_algorithm = SpeculativeAlgorithm.EAGLE
        return planner

    def test_automatic_speculative_limit_leaves_the_kv_memory_budget(self):
        # The fixed pool needs about 24.68 GiB. Crossing that threshold must
        # not let it consume the KV share of the budget.
        for budget in (10.0, 24.68, 24.70, 30.0):
            with self.subTest(budget=budget):
                remaining = self._planner()._handle_max_mamba_cache(budget)
                self.assertGreaterEqual(remaining, budget / (1 + 0.9))
                self.assertGreater(get_schedule().max_mamba_cache_size, 0)

    def test_fitting_automatic_limit_keeps_the_request_sized_pool(self):
        remaining = self._planner()._handle_max_mamba_cache(64.0)
        self.assertEqual(get_schedule().max_mamba_cache_size, 48)
        self.assertGreater(remaining, 0)

    def test_explicit_request_limits_keep_fixed_sizing_including_48(self):
        for limit in (10, 48):
            with self.subTest(request_limit=limit):
                remaining = self._planner(limit)._handle_max_mamba_cache(10.0)
                self.assertEqual(get_schedule().max_mamba_cache_size, limit)
                if limit == 10:
                    self.assertGreater(remaining, 0)
                else:
                    # Supplying the default's numeric value is still explicit.
                    self.assertLess(remaining, 0)

    def test_explicit_state_limit_takes_priority_over_automatic_requests(self):
        remaining = self._planner(state_limit=7)._handle_max_mamba_cache(10.0)
        self.assertEqual(get_schedule().max_mamba_cache_size, 7)
        self.assertGreater(remaining, 0)


if __name__ == "__main__":
    unittest.main()
