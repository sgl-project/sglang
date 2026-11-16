"""The KV budget leaves room for the aiter decode workspace once the request
count, speculative settings and hybrid-SWA decision are final."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import AttentionArch  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache import kv_cache_configurator as kcc  # noqa: E402
from sglang.srt.model_executor import pool_configurator  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1 << 30
MB = 1 << 20
# Llama-3.1-8B: 32 layers x 8 KV heads x 128 x (K, V) x bf16 per token.
KV_BYTES_PER_TOKEN = 32 * 8 * 128 * 2 * 2
# MI300X
GPU_MEM_MB = 192 * 1024
PRE_MODEL_LOAD_GB = 190.0
ACTIVATION_RESERVE_MB = 10 * 1024


def _llama_8b(**overrides):
    config = dict(
        num_attention_heads=32,
        head_dim=128,
        context_len=131072,
        attention_arch=AttentionArch.MHA,
        is_hybrid_swa=False,
        hf_config=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _gemma_3_12b(**overrides):
    return _llama_8b(num_attention_heads=16, head_dim=256, **overrides)


def _per_row(model):
    return kcc.aiter_decode_workspace_bytes_per_row(
        num_heads=model.num_attention_heads,
        head_dim=model.head_dim,
        context_len=model.context_len,
    )


class TestAiterDecodeWorkspaceReserve(CustomTestCase):
    def _configurator(
        self,
        model=None,
        *,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        draft_model=None,
        is_hybrid_swa=False,
        use_mla_backend=False,
        explicit_mem_fraction=None,
    ):
        kvc = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        kvc.model_config = model or _llama_8b()
        kvc.is_draft_worker = False
        kvc.is_hybrid_swa = is_hybrid_swa
        kvc.use_mla_backend = use_mla_backend
        kvc.mambaish_config = None
        kvc.attn_dp_size = 1
        kvc.device = "cuda"
        kvc.post_capture_kv_active = False
        kvc.spec_algorithm = spec_algorithm
        kvc.spec_aux_config = SimpleNamespace(draft_model_config=draft_model)
        kvc.draft_parallel_layout = SimpleNamespace(attn_tp_size=1)
        kvc.server_args = SimpleNamespace(
            _raw_input={"mem_fraction_static": explicit_mem_fraction}
        )
        return kvc

    def _run(
        self,
        kvc,
        *,
        budget_gb=140,
        mem_fraction_static=0.816,
        max_running_requests=None,
        backends=("aiter", "aiter"),
        draft_backend=None,
        num_steps=None,
        is_hip=True,
    ):
        """Run _resolve_memory_pool_config with a KV pool of KV_BYTES_PER_TOKEN
        per token; returns the config and the KV budgets it was sized from."""
        budgets = []

        def config_from_budget(self, budget_bytes, *, cap_tokens=None):
            budgets.append(budget_bytes)
            return SimpleNamespace(
                max_total_num_tokens=budget_bytes // KV_BYTES_PER_TOKEN,
                max_running_requests=None,
            )

        schedule = SimpleNamespace(
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(kcc, "_is_hip", is_hip))
            stack.enter_context(
                patch.object(kcc, "get_schedule", return_value=schedule)
            )
            stack.enter_context(
                patch.object(
                    kcc,
                    "get_memory",
                    return_value=SimpleNamespace(enable_unified_memory=False),
                )
            )
            stack.enter_context(
                patch.object(
                    kcc,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=1),
                )
            )
            stack.enter_context(
                patch.object(
                    kcc,
                    "get_spec",
                    return_value=SimpleNamespace(
                        speculative_draft_attention_backend=draft_backend,
                        speculative_num_steps=num_steps,
                    ),
                )
            )
            stack.enter_context(
                patch.object(kcc, "attention_backends", return_value=backends)
            )
            stack.enter_context(
                patch.object(kcc, "get_device_memory_capacity", return_value=GPU_MEM_MB)
            )
            stack.enter_context(
                patch.object(
                    kcc,
                    "pre_capture_activation_reserve_mb",
                    return_value=ACTIVATION_RESERVE_MB,
                )
            )
            stack.enter_context(patch.object(kcc, "is_unlimited_ocr_v2", lambda c: 0))
            stack.enter_context(
                patch.object(
                    kcc.KVCacheConfigurator,
                    "_profile_available_bytes",
                    lambda self, pre: budget_gb * GB,
                )
            )
            stack.enter_context(
                patch.object(
                    kcc.KVCacheConfigurator, "config_from_budget", config_from_budget
                )
            )
            stack.enter_context(
                patch.object(
                    pool_configurator,
                    "create_memory_pool_configurator",
                    lambda kvc: SimpleNamespace(
                        finalize_with_max_running_requests=lambda config: config
                    ),
                )
            )
            config = kvc._resolve_memory_pool_config(PRE_MODEL_LOAD_GB)
        return config, budgets

    def _headroom(self, mem_fraction_static=0.816):
        slack = PRE_MODEL_LOAD_GB * (1 - mem_fraction_static) * GB
        return int(slack - ACTIVATION_RESERVE_MB * MB)

    def test_long_context_takes_the_overflow_from_the_kv_budget(self):
        # 140 GB of KV holds ~1.1M tokens, so the request pool resolves to 4096.
        kvc = self._configurator()
        config, budgets = self._run(kvc)
        workspace = 4096 * _per_row(_llama_8b())
        self.assertAlmostEqual(workspace / MB, 33280)
        self.assertEqual(budgets, [140 * GB, 140 * GB - (workspace - self._headroom())])
        self.assertEqual(config.max_running_requests, 4096)

    def test_uses_the_resolved_request_count(self):
        # 40 GB of KV resolves to 2048 requests, not 4096.
        kvc = self._configurator()
        config, budgets = self._run(kvc, budget_gb=40)
        self.assertEqual(config.max_running_requests, 2048)
        workspace = 2048 * _per_row(_llama_8b())
        self.assertEqual(budgets[1], 40 * GB - max(0, workspace - self._headroom()))

    def test_short_context_is_unchanged(self):
        kvc = self._configurator(_llama_8b(context_len=8192))
        config, budgets = self._run(kvc)
        self.assertEqual(budgets, [140 * GB, 140 * GB])

    def test_speculative_request_cap_and_draft_backends(self):
        # EAGLE3 sets max_running_requests=48; three draft steps build two
        # multi-step decode backends and one draft-extend backend.
        draft = _llama_8b()
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3, draft_model=draft
        )
        config, budgets = self._run(kvc, max_running_requests=48, num_steps=3)
        self.assertEqual(config.max_running_requests, 48)
        # 4 x 48 x 8.1 MB fits in the slack: the KV budget is untouched.
        self.assertEqual(budgets, [140 * GB, 140 * GB])
        with (
            patch.object(kcc, "_is_hip", True),
            patch.object(kcc, "attention_backends", return_value=("aiter", "aiter")),
            patch.object(
                kcc,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_attention_backend=None,
                    speculative_num_steps=3,
                ),
            ),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
        ):
            self.assertEqual(
                kvc._aiter_decode_workspace_bytes_per_row(), 4 * _per_row(draft)
            )

    def test_draft_on_another_backend_is_not_counted(self):
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, draft_model=_llama_8b()
        )
        with (
            patch.object(kcc, "_is_hip", True),
            patch.object(kcc, "attention_backends", return_value=("aiter", "aiter")),
            patch.object(
                kcc,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_attention_backend="triton",
                    speculative_num_steps=3,
                ),
            ),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
        ):
            self.assertEqual(
                kvc._aiter_decode_workspace_bytes_per_row(), _per_row(_llama_8b())
            )

    def test_gemma3_without_the_swa_pool_reserves_the_workspace(self):
        # On aiter the SWA pool is disabled for Gemma 2/3, so the backend takes
        # the paged decode path and allocates the workspace (~17 GB here).
        kvc = self._configurator(_gemma_3_12b(), is_hybrid_swa=False)
        config, budgets = self._run(kvc, budget_gb=40, mem_fraction_static=0.9)
        workspace = config.max_running_requests * _per_row(_gemma_3_12b())
        self.assertGreater(workspace, 16 * GB)
        reserved = workspace - self._headroom(0.9)
        self.assertGreater(reserved, 0)
        self.assertEqual(budgets, [40 * GB, 40 * GB - reserved])

    def test_no_workspace_with_the_swa_pool_mla_or_unified_attention(self):
        cases = {
            "swa pool": self._configurator(_gemma_3_12b(), is_hybrid_swa=True),
            "mla": self._configurator(use_mla_backend=True),
        }
        for name, kvc in cases.items():
            with self.subTest(name):
                _, budgets = self._run(kvc)
                self.assertEqual(budgets, [140 * GB])
        with envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(True):
            _, budgets = self._run(self._configurator())
        self.assertEqual(budgets, [140 * GB])

    def test_only_on_hip_with_aiter(self):
        _, budgets = self._run(self._configurator(), is_hip=False)
        self.assertEqual(budgets, [140 * GB])
        _, budgets = self._run(self._configurator(), backends=("triton", "triton"))
        self.assertEqual(budgets, [140 * GB])

    def test_explicit_mem_fraction_is_used_as_is(self):
        kvc = self._configurator(explicit_mem_fraction=0.8)
        config, budgets = self._run(kvc, mem_fraction_static=0.8)
        self.assertEqual(budgets, [140 * GB])
        self.assertEqual(config.max_running_requests, 4096)

    def test_huge_workspace_caps_requests_instead_of_the_kv_budget(self):
        # 10M context: ~620 MB of workspace per request.
        model = _llama_8b(context_len=10_000_000)
        kvc = self._configurator(model)
        config, budgets = self._run(kvc, budget_gb=60)
        self.assertGreaterEqual(budgets[1], 30 * GB)
        reserved = 60 * GB - budgets[1]
        workspace = config.max_running_requests * _per_row(model)
        self.assertLessEqual(workspace - self._headroom(), reserved)
        self.assertGreaterEqual(config.max_running_requests, 1)


if __name__ == "__main__":
    unittest.main()
