"""A unified pool without a radix cache sizes state for the requests it
runs, within the state share of the arena: the rest is FULL's."""

import unittest

import msgspec
from test_declared_parts import _build

from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUncachedStateShare(CustomTestCase):
    def test_estimated_concurrency_without_a_radix_cache_leaves_full_its_share(self):
        """REGRESSION: without a radix cache the state target was the slots
        of the estimated concurrency, so sizing grew state until FULL held
        only twice the running requests: a decode node booted 539 requests on
        1815 FULL tokens and refused every longer prompt."""
        _, plan, _ = _build(state=True)
        state_bytes = plan.sub_pool("mamba").spec.entry_bytes()
        owner_bytes = plan.owner().spec.entry_bytes()
        inp = KVSizingInputs(
            budget_bytes=4096 * state_bytes,
            context_len=16,
            radix_cache=False,
            state_share=0.5,
            alignment=1,
        )
        sized = size_plan(plan, inp)
        state = sized.units("mamba") * state_bytes
        self.assertGreater(sized.max_running_requests, 0)
        self.assertLessEqual(state, sized.total_bytes * 0.5)
        # FULL takes the rest of the arena.
        self.assertGreater(
            sized.full_tokens * owner_bytes, (sized.total_bytes - state) * 0.9
        )
        # A concurrency the user asks for still gets its slots.
        asked = size_plan(plan, msgspec.structs.replace(inp, max_running_requests=8))
        self.assertGreaterEqual(asked.units("mamba"), 8)


if __name__ == "__main__":
    unittest.main()
