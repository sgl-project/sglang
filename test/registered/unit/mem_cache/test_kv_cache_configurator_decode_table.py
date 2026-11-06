"""Decode and role-switching prefill servers initialize stateless request tables."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache import kv_cache_configurator as kcc  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ReachedTokenPool(Exception):
    def __init__(self, req_to_token_pool, state):
        super().__init__()
        self.req_to_token_pool = req_to_token_pool
        self.state = state


def _reach_token_pool(self, *, req_to_token_pool, state, **kwargs):
    raise _ReachedTokenPool(req_to_token_pool, state)


class TestDecodeTableInitPools(CustomTestCase):
    def _init_pools(self, *, disagg):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = False
        cfg.mambaish_config = None
        cfg.model_config = SimpleNamespace(context_len=32, hf_config=None)
        cfg.device = "cpu"
        sizes = kcc._PoolSizes(
            max_total_num_tokens=64,
            max_running_requests=4,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
            c4_max_total_num_tokens=0,
            c128_max_total_num_tokens=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            c4_state_dtype=None,
            c128_state_dtype=None,
        )
        with (
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=False),
            ),
            patch.object(kcc, "get_disagg", return_value=disagg),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(kcc, "get_req_to_token_extra_context_len", return_value=0),
            patch.object(kcc, "is_deepseek_dsa", return_value=False),
            patch.object(kcc, "is_deepseek_v4", return_value=False),
            patch.object(
                kcc.KVCacheConfigurator,
                "_validate_prefill_only_disable_kv_cache_pool_family",
                lambda self, *a, **kw: None,
            ),
            patch.object(
                kcc.KVCacheConfigurator, "_build_token_to_kv_pool", _reach_token_pool
            ),
        ):
            with self.assertRaises(_ReachedTokenPool) as reached:
                cfg._init_pools(
                    sizes=sizes, req_to_token_pool=None, token_to_kv_pool_allocator=None
                )
        return reached.exception

    def _assert_decode_table_without_state(self, reached):
        self.assertIsInstance(reached.req_to_token_pool, DecodeReqToTokenPool)
        self.assertIsNone(reached.state)

    def test_a_decode_server_reaches_the_token_pool(self):
        """A decode table must expose its absent recurrent state so pool initialization
        can continue to the token pool."""
        reached = self._init_pools(
            disagg=SimpleNamespace(
                disaggregation_mode="decode",
                enable_pd_role_switch=False,
                disaggregation_decode_extra_slots=2,
            )
        )
        self._assert_decode_table_without_state(reached)

    def test_a_role_switching_prefill_reaches_the_token_pool(self):
        """A prefill server that can switch to decode builds the decode table
        too, and must reach the token pool the same way."""
        reached = self._init_pools(
            disagg=SimpleNamespace(
                disaggregation_mode="prefill",
                enable_pd_role_switch=True,
                disaggregation_decode_extra_slots=None,
            )
        )
        self._assert_decode_table_without_state(reached)


if __name__ == "__main__":
    unittest.main()
