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


def _spec(
    draft_backend=None, num_steps=None, attention_mode="prefill", multi_layer=False
):
    return SimpleNamespace(
        speculative_draft_attention_backend=draft_backend,
        speculative_num_steps=num_steps,
        speculative_attention_mode=attention_mode,
        enable_multi_layer_eagle=multi_layer,
    )


def _exec(two_batch_overlap=False):
    return SimpleNamespace(
        overlap=SimpleNamespace(enable_two_batch_overlap=two_batch_overlap)
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
        two_batch_overlap=False,
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
                    return_value=_spec(draft_backend, num_steps),
                )
            )
            stack.enter_context(
                patch.object(kcc, "get_exec", return_value=_exec(two_batch_overlap))
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

    def _per_row_total(
        self, kvc, *, backends=("aiter", "aiter"), spec=None, two_batch_overlap=False
    ):
        """kvc._aiter_decode_workspace_bytes_per_row() under the given flags."""
        with (
            patch.object(kcc, "_is_hip", True),
            patch.object(kcc, "attention_backends", return_value=backends),
            patch.object(kcc, "get_spec", return_value=spec or _spec()),
            patch.object(kcc, "get_exec", return_value=_exec(two_batch_overlap)),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
        ):
            return kvc._aiter_decode_workspace_bytes_per_row()

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
        # EAGLE3 sets max_running_requests=48. The draft runner builds its own
        # backend, then two multi-step decode backends (3 steps) and one
        # draft-extend backend: 4 draft workspaces plus the target's.
        draft = _llama_8b()
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3, draft_model=draft
        )
        config, budgets = self._run(kvc, max_running_requests=48, num_steps=3)
        self.assertEqual(config.max_running_requests, 48)
        # 5 x 48 x 8.1 MB fits in the slack: the KV budget is untouched.
        self.assertEqual(budgets, [140 * GB, 140 * GB])
        self.assertEqual(
            self._per_row_total(kvc, spec=_spec(num_steps=3)), 5 * _per_row(draft)
        )

    def test_eagle_and_standalone_draft_backend_counts(self):
        draft = _llama_8b()
        per_row = _per_row(draft)
        for algo in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.STANDALONE):
            kvc = self._configurator(spec_algorithm=algo, draft_model=draft)
            with self.subTest(algo=algo.name):
                # One step: target + draft runner + draft-extend, no decode.
                self.assertEqual(
                    self._per_row_total(kvc, spec=_spec(num_steps=1)), 3 * per_row
                )
                # Five steps: target + draft runner + 4 decode + draft-extend.
                self.assertEqual(
                    self._per_row_total(kvc, spec=_spec(num_steps=5)), 7 * per_row
                )

    def test_multi_layer_eagle_and_frozen_kv_mtp_draft_counts(self):
        draft = _llama_8b()
        per_row = _per_row(draft)
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, draft_model=draft
        )
        # One runner backend and one draft-extend backend per step runner.
        self.assertEqual(
            self._per_row_total(kvc, spec=_spec(num_steps=3, multi_layer=True)),
            (1 + 2 * 3) * per_row,
        )
        # Frozen-KV MTP only builds its runner's backend.
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP, draft_model=draft
        )
        self.assertEqual(self._per_row_total(kvc, spec=_spec(num_steps=3)), 2 * per_row)

    def test_draft_on_another_backend_is_not_counted(self):
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, draft_model=_llama_8b()
        )
        self.assertEqual(
            self._per_row_total(kvc, spec=_spec("triton", num_steps=3)),
            _per_row(_llama_8b()),
        )

    def test_draft_extend_follows_speculative_attention_mode(self):
        # Prefill on triton, decode on aiter: the target and the draft runner
        # each hold one aiter backend, the 2 multi-step decode backends are
        # aiter, and the draft-extend backend is aiter only in decode mode.
        draft = _llama_8b()
        per_row = _per_row(draft)
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, draft_model=draft
        )
        backends = ("triton", "aiter")
        for mode, expected in (("prefill", 4), ("decode", 5)):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._per_row_total(
                        kvc,
                        backends=backends,
                        spec=_spec(num_steps=3, attention_mode=mode),
                    ),
                    expected * per_row,
                )

    def test_two_batch_overlap_counts_three_target_backends(self):
        model = _llama_8b(context_len=40960)
        kvc = self._configurator(model)
        self.assertEqual(
            self._per_row_total(kvc, two_batch_overlap=True), 3 * _per_row(model)
        )
        # 4096 requests: one workspace (~10.9 GB) fits in the slack, three do not.
        _, budgets = self._run(kvc)
        self.assertEqual(budgets, [140 * GB, 140 * GB])
        config, budgets = self._run(kvc, two_batch_overlap=True)
        self.assertEqual(config.max_running_requests, 4096)
        workspace = 4096 * 3 * _per_row(model)
        self.assertGreater(workspace, self._headroom())
        self.assertEqual(budgets[1], 140 * GB - (workspace - self._headroom()))

    def test_vectorized_5d_kv_cache_has_no_workspace(self):
        kvc = self._configurator()
        with (
            envs.SGLANG_USE_AITER.override(True),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override("vectorized_5d"),
        ):
            self.assertEqual(self._per_row_total(kvc), 0)
            _, budgets = self._run(kvc)
            self.assertEqual(budgets, [140 * GB])
            # SGLANG_USE_HND_KVCACHE takes precedence over the 5D layout.
            with envs.SGLANG_USE_HND_KVCACHE.override(True):
                self.assertEqual(self._per_row_total(kvc), _per_row(_llama_8b()))
        # The pool reads the layout env only with SGLANG_USE_AITER.
        with (
            envs.SGLANG_USE_AITER.override(False),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override("vectorized_5d"),
        ):
            self.assertEqual(self._per_row_total(kvc), _per_row(_llama_8b()))

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

    def test_single_request_cap_keeps_the_budget_non_negative(self):
        # 12M context: ~780 MB per request, more than half of a 1 GB budget,
        # and no slack above the activation reserve at 0.95. The cap formula
        # rounds to 0; one request still fits, so the cap is 1.
        model = _llama_8b(context_len=12_000_000)
        kvc = self._configurator(model)
        self.assertLessEqual(self._headroom(0.95), 0)
        config, budgets = self._run(kvc, budget_gb=1, mem_fraction_static=0.95)
        self.assertGreater(_per_row(model), GB // 2)
        self.assertEqual(config.max_running_requests, 1)
        self.assertEqual(budgets[1], GB - _per_row(model))
        self.assertGreaterEqual(budgets[1], 0)

    def test_single_request_over_the_budget_raises(self):
        # 100M context: ~6.5 GB of workspace for one request, over a 2 GB budget.
        kvc = self._configurator(_llama_8b(context_len=100_000_000))
        with self.assertRaisesRegex(RuntimeError, "single request"):
            self._run(kvc, budget_gb=2, mem_fraction_static=0.95)


if __name__ == "__main__":
    unittest.main()
