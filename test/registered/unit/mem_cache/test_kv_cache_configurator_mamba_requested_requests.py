"""An explicit --max-running-requests must not be silently capped by the
auto-fit Mamba state cache while the KV pool has memory to spare.

With the fallback --mamba-full-memory-ratio the auto-fit split gives Mamba
state about half the cache budget. Several state slots per request (radix
cache, speculative scratch) can then hold far fewer requests than the caller
asked for. When the caller set the request limit and nobody chose the ratio,
the planner moves memory from KV to Mamba state, up to
MAMBA_AUTO_FIT_MAX_SHARE of the budget.
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
from sglang.srt.mem_cache.kv_cache_configurator import (
    MAMBA_AUTO_FIT_MAX_SHARE,
    KVCacheConfigurator,
)
from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1 << 30


class TestMambaAutoFitHonorsRequestedRequests(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _planner(self, request_limit=None, *, ratio=None, spec=True):
        reset_context()
        kwargs = dict(
            model_path="dummy",
            max_running_requests=request_limit,
            mamba_full_memory_ratio=ratio,
        )
        if spec:
            kwargs.update(
                speculative_algorithm="EAGLE",
                speculative_num_steps=3,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=4,
            )
        args = ServerArgs(**kwargs)
        if spec and request_limit is None:
            # The same declaration used by speculation; raw input stays unset.
            declare_resolution(args, "_handle_eagle_family", max_running_requests=48)
        publish(args, role="scheduler")
        params = Mamba2CacheParams(
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=list(range(40)),
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=8192,
                n_groups=8,
                num_heads=128,
                head_dim=64,
                state_size=128,
                conv_kernel=4,
            ),
        )
        planner = object.__new__(KVCacheConfigurator)
        planner.pp_size = planner.attn_dp_size = 1
        planner.spec_aux_config = None
        planner.is_hybrid_swa = False
        planner.is_draft_worker = False
        planner.mambaish_config = SimpleNamespace(mamba2_cache_params=params)
        planner.hybrid_gdn_config = planner.mambaish_config
        planner.hybrid_kda_config = None
        planner.extra_mamba_cache_bytes_per_req = 0
        planner.spec_algorithm = (
            SpeculativeAlgorithm.EAGLE if spec else SpeculativeAlgorithm.NONE
        )
        return planner

    def _size(self, budget_gb, request_limit=None, **kw):
        planner = self._planner(request_limit, **kw)
        remaining = planner._handle_max_mamba_cache(budget_gb)
        return (
            get_schedule().max_mamba_cache_size,
            remaining,
            planner._calculate_mamba_ratio(),
        )

    def test_explicit_limit_grows_the_state_cache_to_fit(self):
        for spec in (True, False):
            with self.subTest(spec=spec):
                planner = self._planner(128, spec=spec)
                ratio = planner._calculate_mamba_ratio()
                per_req = (
                    planner.mambaish_config.mamba2_cache_params.mamba_cache_per_req
                )
                # 128 requests' state (plus speculative scratch) at 80% of the
                # budget: above the fallback split, below the share cap.
                slots_per_req = ratio + (4 if spec else 0)
                budget = 128 * slots_per_req * per_req / GIB / 0.8
                baseline, _, _ = self._size(budget, 128, ratio=0.9, spec=spec)
                self.assertLess(baseline // ratio, 128)
                size, remaining, _ = self._size(budget, 128, spec=spec)
                self.assertEqual(size, 128 * ratio)
                self.assertGreaterEqual(
                    remaining, budget * (1 - MAMBA_AUTO_FIT_MAX_SHARE)
                )

    def test_growth_stops_at_the_share_cap(self):
        budget = 60.0
        baseline, _, ratio = self._size(budget, 128, ratio=0.9)
        size, remaining, _ = self._size(budget, 128)
        self.assertGreater(size, baseline)
        self.assertLess(size // ratio, 128)
        # The KV pool keeps its floor (allow one slot of rounding).
        self.assertGreater(remaining, budget * (1 - MAMBA_AUTO_FIT_MAX_SHARE) - 1.0)

    def test_limit_that_already_fits_is_unchanged(self):
        baseline, base_rest, _ = self._size(200.0, 4, ratio=0.9)
        size, remaining, _ = self._size(200.0, 4)
        self.assertEqual(size, baseline)
        self.assertAlmostEqual(remaining, base_rest)

    def test_default_limit_keeps_the_fallback_split(self):
        baseline, base_rest, _ = self._size(60.0, 48, ratio=0.9)
        size, remaining, _ = self._size(60.0)  # spec default 48, not explicit
        self.assertEqual(size, baseline)
        self.assertAlmostEqual(remaining, base_rest)

    def test_explicit_ratio_is_respected(self):
        for ratio in (0.9, 0.5):
            with self.subTest(ratio=ratio):
                size, remaining, r = self._size(60.0, 128, ratio=ratio)
                self.assertLess(size // r, 128)
                self.assertGreater(remaining, 60.0 / (1 + ratio) - 1.0)


if __name__ == "__main__":
    unittest.main()
