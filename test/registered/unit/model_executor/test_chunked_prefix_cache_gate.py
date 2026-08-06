"""The load-time chunked-prefix gate must land on the published config bag.

Regression: the gate wrote the ServerArgs instance while every attention
backend reads ``get_schedule().disable_chunked_prefix_cache`` — the bag never
saw the flip, so an unsupported backend kept chunked prefix enabled.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest

from sglang.srt.model_executor.model_runner_components.misc_utils import (
    maybe_disable_chunked_prefix_cache,
)
from sglang.srt.runtime_context import get_context, get_schedule, get_server_args
from sglang.test.test_utils import CustomTestCase


class TestChunkedPrefixCacheGate(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_unsupported_backend_disables_on_the_bag(self):
        self._seed(attention_backend="triton")
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertTrue(get_schedule().disable_chunked_prefix_cache)
        self.assertFalse(get_server_args().disable_chunked_prefix_cache)

    def test_supported_backend_keeps_chunked_prefix(self):
        self._seed(attention_backend="fa3")
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertFalse(get_schedule().disable_chunked_prefix_cache)

    def test_draft_worker_never_writes(self):
        self._seed(attention_backend="triton")
        maybe_disable_chunked_prefix_cache(use_mla_backend=False, is_draft_worker=True)
        self.assertFalse(get_schedule().disable_chunked_prefix_cache)

    def test_republish_discards_the_gate_so_it_must_run_after_publish(self):
        # Pins the ordering contract in ModelRunner.__init__: publishing
        # rebuilds the bags from the pristine instance, so the gate runs after
        # the target-worker publish.
        self._seed(attention_backend="triton")
        sa = get_server_args()
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertTrue(get_schedule().disable_chunked_prefix_cache)
        get_context().set_server_args(sa)  # what a later republish would do
        self.assertFalse(get_schedule().disable_chunked_prefix_cache)

    def test_the_gate_survives_a_draft_build(self):
        # The draft build no longer publishes a config of its own, so the gate
        # the target resolved stays in the bags for the rest of the process.
        self._seed(attention_backend="triton")
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertTrue(get_schedule().disable_chunked_prefix_cache)

        maybe_disable_chunked_prefix_cache(use_mla_backend=False, is_draft_worker=True)
        self.assertTrue(get_schedule().disable_chunked_prefix_cache)


if __name__ == "__main__":
    unittest.main()
