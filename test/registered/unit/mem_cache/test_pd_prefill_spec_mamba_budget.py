"""CPU regressions for role-aware speculative Mamba-state sizing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache import kv_cache_configurator as kvc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

GB = 1 << 30
PERSISTENT_BYTES_PER_SLOT = 1024
REPLAY_RING_BYTES_PER_SLOT = 64
DRAFT_TOKENS = 8
MAX_RUNNING = 4


class TestPDWorkerSpecMambaBudget(unittest.TestCase):
    def _budgeted_verify_bytes(self, *, mode, draft_tokens, replay):
        params = SimpleNamespace(
            layers=[0],
            mamba_cache_per_req=PERSISTENT_BYTES_PER_SLOT,
            replayssm_ring_bytes_per_req=lambda *, record_len: (
                REPLAY_RING_BYTES_PER_SLOT
            ),
        )
        configurator = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=params),
            ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
            hybrid_gdn_config=object(),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), num_hidden_layers=1
            ),
            _calculate_mamba_ratio=lambda: 1,
        )
        schedule = SimpleNamespace(
            max_mamba_cache_size=MAX_RUNNING,
            max_running_requests=MAX_RUNNING,
        )
        exec_config = SimpleNamespace(
            mamba=SimpleNamespace(
                enable_linear_replayssm_spec=replay,
                linear_replayssm_cache_len=16,
            )
        )

        with (
            patch.object(
                kvc,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=mode),
            ),
            patch.object(
                kvc, "max_speculative_num_draft_tokens", return_value=draft_tokens
            ),
            patch.object(kvc, "get_schedule", return_value=schedule),
            patch.object(kvc, "get_exec", return_value=exec_config),
            patch.object(
                kvc,
                "get_context",
                return_value=SimpleNamespace(override=Mock()),
            ),
            patch.object(kvc, "kimi_linear_config", return_value=None),
        ):
            remaining_gb = kvc.KVCacheConfigurator._handle_max_mamba_cache(
                configurator, 1.0
            )

        charged_bytes = round((1.0 - remaining_gb) * GB)
        persistent_bytes = (MAX_RUNNING + 1) * PERSISTENT_BYTES_PER_SLOT
        return charged_bytes - persistent_bytes

    def test_role_and_replay_matrix(self):
        snapshot_bytes = (MAX_RUNNING + 1) * DRAFT_TOKENS * PERSISTENT_BYTES_PER_SLOT
        replay_ring_bytes = (MAX_RUNNING + 1) * REPLAY_RING_BYTES_PER_SLOT
        cases = (
            ("null", None, False, 0, "standalone without speculation"),
            ("null", DRAFT_TOKENS, False, snapshot_bytes, "standalone snapshot"),
            ("null", DRAFT_TOKENS, True, replay_ring_bytes, "standalone ReplaySSM"),
            ("decode", DRAFT_TOKENS, False, snapshot_bytes, "PD decode snapshot"),
            ("decode", DRAFT_TOKENS, True, replay_ring_bytes, "PD decode ReplaySSM"),
            ("prefill", DRAFT_TOKENS, False, 0, "PD prefill snapshot config"),
            ("prefill", DRAFT_TOKENS, True, 0, "PD prefill ReplaySSM config"),
        )

        for mode, draft_tokens, replay, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    self._budgeted_verify_bytes(
                        mode=mode, draft_tokens=draft_tokens, replay=replay
                    ),
                    expected,
                )

    def test_pdmux_keeps_target_verify_width(self):
        with (
            patch.object(
                kvc,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode="null", enable_pdmux=True
                ),
            ),
            patch.object(
                kvc,
                "max_speculative_num_draft_tokens",
                return_value=DRAFT_TOKENS,
            ),
        ):
            self.assertEqual(kvc._target_verify_num_draft_tokens(), DRAFT_TOKENS)

    def test_pool_allocation_uses_the_same_worker_role_gate(self):
        params = SimpleNamespace(layers=[0])
        configurator = SimpleNamespace(
            model_config=SimpleNamespace(context_len=8192),
            device="cuda",
            mambaish_config=SimpleNamespace(mamba2_cache_params=params),
            layer_info=SimpleNamespace(start_layer=0, end_layer=1),
            hybrid_gdn_config=object(),
        )
        schedule = SimpleNamespace(
            max_mamba_cache_size=MAX_RUNNING, disable_overlap_schedule=False
        )
        exec_config = SimpleNamespace(
            features=SimpleNamespace(enable_memory_saver=False),
            mamba=SimpleNamespace(
                enable_linear_replayssm=False,
                linear_replayssm_cache_len=16,
                enable_linear_replayssm_spec=False,
            ),
        )
        memory = SimpleNamespace(enable_page_major_kv_layout=False)
        spec = SimpleNamespace(speculative_algorithm="DFLASH", speculative_eagle_topk=1)

        for mode, expected in (
            ("prefill", None),
            ("decode", DRAFT_TOKENS),
            ("null", DRAFT_TOKENS),
        ):
            pool = Mock()
            with (
                self.subTest(mode=mode),
                patch.object(
                    kvc,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=mode),
                ),
                patch.object(
                    kvc,
                    "max_speculative_num_draft_tokens",
                    return_value=DRAFT_TOKENS,
                ),
                patch.object(kvc, "get_schedule", return_value=schedule),
                patch.object(kvc, "get_exec", return_value=exec_config),
                patch.object(kvc, "get_memory", return_value=memory),
                patch.object(kvc, "get_spec", return_value=spec),
                patch.object(kvc, "mamba_extra_buffer_enabled", return_value=False),
                patch.object(
                    kvc, "mamba_extra_buffer_lazy_enabled", return_value=False
                ),
                patch.object(kvc, "HybridReqToTokenPool", return_value=pool) as ctor,
            ):
                result = kvc.KVCacheConfigurator._build_hybrid_req_pool(
                    configurator,
                    max_num_reqs=MAX_RUNNING,
                    extra_max_context_len=0,
                )

            self.assertIs(result, pool)
            self.assertEqual(
                ctor.call_args.kwargs["speculative_num_draft_tokens"], expected
            )


if __name__ == "__main__":
    unittest.main()
