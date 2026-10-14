"""Warn at startup when the KV pool cannot hold one full-length request.

[Test Category] Correctness
[Test Target] mem_cache/kv_cache_configurator.py (_warn_if_pool_cannot_hold_context)

Without the warning, an undersized pool shows up only as refused or truncated
requests at serving time. Under DCP the pool's per-rank rows are not its
capacity: each row widens into attn_dcp_size request tokens, so comparing rows
to --context-length would warn on a pool that holds 1M tokens comfortably.
KV sharding widens rows the same way, by kv_shard_size. Under hybrid SWA with
unified memory the allocator, not the sizer, sets what ModelRunner serves.
"""

import types
import unittest

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import (
    get_memory,
    get_parallel,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LOGGER = "sglang.srt.mem_cache.kv_cache_configurator"
CONTEXT_LEN = 1_048_576


def _check(
    *, max_total_num_tokens, kv_shard_size=1, is_hybrid_swa=False, allocator=None
):
    # Only the attributes the check and the capacity helpers read.
    stub = types.SimpleNamespace(
        model_config=types.SimpleNamespace(context_len=CONTEXT_LEN),
        is_draft_worker=False,
        is_hybrid_swa=is_hybrid_swa,
        kv_shard_size=kv_shard_size,
    )
    stub.loc_space_scale = KVCacheConfigurator.loc_space_scale.fget(stub)
    for name in ("logical_token_capacity", "hybrid_swa_token_capacity"):
        setattr(stub, name, types.MethodType(getattr(KVCacheConfigurator, name), stub))
    KVCacheConfigurator._warn_if_pool_cannot_hold_context(
        stub,
        max_total_num_tokens=max_total_num_tokens,
        full_max_total_num_tokens=max_total_num_tokens if is_hybrid_swa else None,
        swa_max_total_num_tokens=None,
        allocator=allocator,
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

    def test_kv_sharding_counts_request_tokens_not_rows(self):
        # 131,072 rows per rank x an 8-rank shard group = exactly CONTEXT_LEN.
        with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            with self.assertNoLogs(LOGGER, "WARNING"):
                _check(max_total_num_tokens=131_072, kv_shard_size=8)

    def test_unified_hybrid_swa_counts_the_allocator_not_the_sizer(self):
        # The sizer estimated 800K, but the unified allocator gave the full side
        # 1.3M; ModelRunner serves the latter.
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        allocator = types.SimpleNamespace(size_full=1_300_000)
        with (
            get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0),
            get_memory().override(enable_unified_memory=True),
        ):
            with self.assertNoLogs(LOGGER, "WARNING"):
                _check(
                    max_total_num_tokens=800_000,
                    is_hybrid_swa=True,
                    allocator=allocator,
                )


if __name__ == "__main__":
    unittest.main()
