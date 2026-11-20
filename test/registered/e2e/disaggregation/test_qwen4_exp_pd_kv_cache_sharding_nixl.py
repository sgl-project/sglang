"""test_qwen4_exp_pd_kv_cache_sharding.py over NIXL instead of Mooncake.

NIXL gates the per-request state and aux on its own send path: every prefill
rank sends its KV pages to every decode rank, and only the rank-matched one
also sends the GDN/QSA/PLE state and the aux, then all report completion.
"""

import unittest

import test_qwen4_exp_pd_kv_cache_sharding as mooncake_case

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    configure_nixl_pd_backend,
)

register_cuda_ci(est_time=600, stage="nightly", runner_config="8-gpu-b200")


class TestQwen4ExpPDKVCacheShardingNixl(mooncake_case.TestQwen4ExpPDKVCacheSharding):
    @classmethod
    def configure_transfer_backend(cls):
        configure_nixl_pd_backend(cls)


if __name__ == "__main__":
    unittest.main()
