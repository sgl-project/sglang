"""Warn at startup when the KV pool cannot hold one full-length request.

[Test Category] Correctness
[Test Target] mem_cache/kv_cache_configurator.py (_warn_if_pool_cannot_hold_context)

Without the warning, an undersized pool shows up only as refused or truncated
requests at serving time. Under DCP the pool's per-rank rows are not its
capacity: each row widens into attn_dcp_size request tokens, so comparing rows
to --context-length would warn on a pool that holds 1M tokens comfortably.
"""

import types
import unittest

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LOGGER = "sglang.srt.mem_cache.kv_cache_configurator"
CONTEXT_LEN = 1_048_576


def _check(*, max_total_num_tokens):
    # Only the attributes the check and logical_token_capacity read.
    stub = types.SimpleNamespace(
        model_config=types.SimpleNamespace(context_len=CONTEXT_LEN),
        is_draft_worker=False,
        is_hybrid_swa=False,
    )
    stub.loc_space_scale = KVCacheConfigurator.loc_space_scale.fget(stub)
    stub.logical_token_capacity = types.MethodType(
        KVCacheConfigurator.logical_token_capacity, stub
    )
    KVCacheConfigurator._warn_if_pool_cannot_hold_context(
        stub, max_total_num_tokens=max_total_num_tokens
    )


class TestPoolContextWarning(CustomTestCase):
    def test_warns_when_the_pool_is_short(self):
        with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            with self.assertLogs(LOGGER, "WARNING") as cm:
                _check(max_total_num_tokens=210_688)
        self.assertIn("210688", cm.output[0])
        self.assertIn(str(CONTEXT_LEN), cm.output[0])

    def test_dcp_counts_request_tokens_not_rows(self):
        # A3 dcp16 at 0.76: 114,048 rows per rank serve 1,824,768 tokens.
        with get_parallel().override(attn_dcp_size=16, attn_dcp_rank=0):
            with self.assertNoLogs(LOGGER, "WARNING"):
                _check(max_total_num_tokens=114_048)


if __name__ == "__main__":
    unittest.main()
