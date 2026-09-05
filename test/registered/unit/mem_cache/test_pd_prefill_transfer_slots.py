import unittest
from types import SimpleNamespace

from sglang.srt import runtime_context as rc
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_schedule
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1 << 30
STATE_BYTES = 64 << 20


def make_configurator(
    mode, *, max_running_requests=16, attn_dp_size=1, draft_tokens=None
):
    configurator = SimpleNamespace(
        ps=SimpleNamespace(attn_dp_size=attn_dp_size),
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                layers=[], mamba_cache_per_req=STATE_BYTES
            )
        ),
        spec_algorithm=SimpleNamespace(is_none=lambda: draft_tokens is None),
        hybrid_gdn_config=None,
        model_config=SimpleNamespace(),
    )
    configurator._prefill_transfer_slots = lambda count: (
        KVCacheConfigurator._prefill_transfer_slots(configurator, count)
    )
    return configurator, dict(
        disaggregation_mode=mode,
        disable_radix_cache=True,
        enable_linear_replayssm_spec=False,
        max_mamba_cache_size=None,
        max_running_requests=max_running_requests,
        speculative_num_draft_tokens=draft_tokens,
    )


def handle_max_mamba_cache(mode, **kwargs):
    configurator, server_args = make_configurator(mode, **kwargs)
    with rc.get_context().override_server_args(**server_args):
        remaining = KVCacheConfigurator._handle_max_mamba_cache(configurator, 100.0)
        return get_schedule().max_mamba_cache_size, remaining


class TestPrefillTransferSlots(unittest.TestCase):
    def test_only_prefill_reserves_transfer_slots(self):
        for mode, expected in (("prefill", 16), ("decode", 0), ("null", 0)):
            configurator, server_args = make_configurator(mode)
            with rc.get_context().override_server_args(**server_args):
                self.assertEqual(
                    KVCacheConfigurator._prefill_transfer_slots(configurator, 16),
                    expected,
                )

    def test_state_pool_and_memory_include_prefill_transfers(self):
        pool_size, remaining = handle_max_mamba_cache("prefill")
        self.assertEqual(pool_size, 32)
        self.assertAlmostEqual(remaining, 100.0 - 33 * STATE_BYTES / GB)

    def test_draft_memory_is_reserved_only_for_running_requests(self):
        pool_size, remaining = handle_max_mamba_cache("prefill", draft_tokens=3)
        self.assertEqual(pool_size, 32)
        expected = (33 + 17 * 3) * STATE_BYTES / GB
        self.assertAlmostEqual(remaining, 100.0 - expected)

    def test_headroom_is_per_attention_worker(self):
        pool_size, _ = handle_max_mamba_cache("prefill", attn_dp_size=2)
        self.assertEqual(pool_size, 16)


if __name__ == "__main__":
    unittest.main()
