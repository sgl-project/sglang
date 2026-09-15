import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaCacheRebalance(CustomTestCase):
    GiB = 1 << 30
    STATE_BYTES_PER_SLOT = 36 << 20

    @staticmethod
    def _fake_configurator(*, attn_dp_size=1, pp_size=1):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.ps = SimpleNamespace(attn_dp_size=attn_dp_size, pp_size=pp_size)
        return configurator

    def _target_size(
        self,
        *,
        total_budget_gib=8,
        required_kv_gib=2,
        max_running_requests=32,
        max_total_tokens=65_000,
        attn_dp_size=1,
        pp_size=1,
        has_spec_dec=False,
        enable_unified_memory=False,
        estimator_available=True,
        default_cache_size=101,
    ):
        from sglang.srt import runtime_context as rc

        fake = self._fake_configurator(attn_dp_size=attn_dp_size, pp_size=pp_size)
        pool_configurator = SimpleNamespace(
            required_memory_bytes_for_max_tokens=lambda *_: (
                required_kv_gib * self.GiB if estimator_available else None
            )
        )
        with (
            envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False),
            rc.get_context().override_server_args(
                disable_radix_cache=False,
                disable_overlap_schedule=False,
                mamba_radix_cache_strategy="extra_buffer",
                max_running_requests=max_running_requests,
                max_total_tokens=max_total_tokens,
                page_size=1,
                enable_unified_memory=enable_unified_memory,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator."
                "create_memory_pool_configurator",
                return_value=pool_configurator,
            ),
        ):
            return KVCacheConfigurator._target_mamba_cache_size_for_workload_caps(
                fake,
                default_cache_size=default_cache_size,
                total_budget_bytes=total_budget_gib * self.GiB,
                state_bytes_per_slot=self.STATE_BYTES_PER_SLOT,
                has_spec_dec=has_spec_dec,
            )

    def test_donates_capped_kv_budget_to_reach_target_concurrency(self):
        # extra_buffer needs five state slots per request, so 32 requests need 160.
        self.assertEqual(self._target_size(), 160)

    def test_accounts_for_attention_dp_sharding(self):
        # max_running_requests is global; each of two attention-DP workers serves 32.
        self.assertEqual(
            self._target_size(max_running_requests=64, attn_dp_size=2), 160
        )

    def test_retains_ratio_pool_when_joint_targets_do_not_fit(self):
        with self.assertLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="INFO"
        ):
            self.assertEqual(self._target_size(total_budget_gib=7), 101)

    def test_requires_both_explicit_caps(self):
        self.assertEqual(self._target_size(max_total_tokens=None), 101)
        self.assertEqual(self._target_size(max_running_requests=None), 101)

    def test_unsupported_modes_retain_legacy_sizing(self):
        self.assertEqual(self._target_size(has_spec_dec=True), 101)
        self.assertEqual(self._target_size(pp_size=2), 101)
        self.assertEqual(self._target_size(enable_unified_memory=True), 101)

    def test_unsupported_pool_retains_legacy_sizing(self):
        with self.assertLogs(
            "sglang.srt.mem_cache.kv_cache_configurator", level="INFO"
        ):
            self.assertEqual(self._target_size(estimator_available=False), 101)

    def test_never_shrinks_ratio_sized_pool(self):
        self.assertEqual(self._target_size(default_cache_size=200), 200)

    def test_handle_max_mamba_cache_applies_joint_sizing(self):
        from sglang.srt import runtime_context as rc

        cache_params = SimpleNamespace(
            layers=[0],
            mamba_cache_per_req=self.STATE_BYTES_PER_SLOT,
        )
        fake = object.__new__(KVCacheConfigurator)
        fake.mambaish_config = SimpleNamespace(mamba2_cache_params=cache_params)
        fake.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        fake.ps = SimpleNamespace(attn_dp_size=1, pp_size=1)
        fake.hybrid_gdn_config = None
        fake.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), num_hidden_layers=1
        )
        pool_configurator = SimpleNamespace(
            required_memory_bytes_for_max_tokens=lambda *_: 2 * self.GiB
        )
        with (
            envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(False),
            rc.get_context().override_server_args(
                disable_radix_cache=False,
                disable_overlap_schedule=False,
                mamba_radix_cache_strategy="extra_buffer",
                max_mamba_cache_size=None,
                max_running_requests=32,
                max_total_tokens=65_000,
                mamba_full_memory_ratio=0.9,
                page_size=1,
                enable_linear_replayssm_spec=False,
                enable_unified_memory=False,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator."
                "create_memory_pool_configurator",
                return_value=pool_configurator,
            ),
        ):
            remaining_gib = KVCacheConfigurator._handle_max_mamba_cache(fake, 8.0)
            self.assertEqual(rc.get_schedule().max_mamba_cache_size, 160)
            self.assertAlmostEqual(
                remaining_gib,
                8.0 - 161 * self.STATE_BYTES_PER_SLOT / self.GiB,
            )

    def test_replayssm_preserves_fixed_scratch_budget(self):
        from sglang.srt import runtime_context as rc

        ring_per_req = 8 << 20
        fake = self._fake_configurator()
        fake.mambaish_config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                layers=[0],
                mamba_cache_per_req=self.STATE_BYTES_PER_SLOT,
                replayssm_ring_bytes_per_req=lambda **_: ring_per_req,
            )
        )
        fake.spec_algorithm = SimpleNamespace(is_none=lambda: False)
        fake.hybrid_gdn_config = object()
        fake.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), num_hidden_layers=1
        )
        with (
            rc.get_context().override_server_args(
                max_mamba_cache_size=None,
                max_running_requests=32,
                max_total_tokens=65_000,
                mamba_full_memory_ratio=0.9,
                disable_radix_cache=False,
                enable_linear_replayssm_spec=True,
                linear_replayssm_cache_len=4,
                speculative_num_draft_tokens=4,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator."
                "create_memory_pool_configurator"
            ) as create_configurator,
        ):
            remaining_gib = fake._handle_max_mamba_cache(8.0)
            fixed_bytes = ring_per_req * 33
            ratio_budget_bytes = 8.0 * 0.9 / 1.9 * self.GiB
            expected_slots = int(
                (ratio_budget_bytes - fixed_bytes - self.STATE_BYTES_PER_SLOT)
                // self.STATE_BYTES_PER_SLOT
            )
            self.assertEqual(rc.get_schedule().max_mamba_cache_size, expected_slots)
            self.assertAlmostEqual(
                remaining_gib,
                8.0
                - ((expected_slots + 1) * self.STATE_BYTES_PER_SLOT + fixed_bytes)
                / self.GiB,
            )
        create_configurator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
