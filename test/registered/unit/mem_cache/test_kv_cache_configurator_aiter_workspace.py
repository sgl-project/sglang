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


def _per_row(model, context_len=None):
    return kcc.aiter_decode_workspace_bytes_per_row(
        num_heads=model.num_attention_heads,
        head_dim=model.head_dim,
        context_len=context_len or model.context_len,
    )


def _spec(
    draft_backend=None,
    num_steps=None,
    attention_mode="prefill",
    multi_layer=False,
    adaptive=False,
):
    return SimpleNamespace(
        speculative_draft_attention_backend=draft_backend,
        speculative_num_steps=num_steps,
        speculative_attention_mode=attention_mode,
        enable_multi_layer_eagle=multi_layer,
        speculative_adaptive=adaptive,
        speculative_adaptive_config=None,
    )


def _parallel(pp_size=1):
    return SimpleNamespace(
        attn_tp_size=1,
        pp_size=pp_size,
        world_group=SimpleNamespace(cpu_group=None),
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
        explicit_max_running_requests=None,
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
            _raw_input={
                "mem_fraction_static": explicit_mem_fraction,
                "max_running_requests": explicit_max_running_requests,
            }
        )
        return kvc

    def _run(
        self,
        kvc,
        *,
        budget_gb=140,
        budget_bytes=None,
        page_size=1,
        mem_fraction_static=0.816,
        max_running_requests=None,
        backends=("aiter", "aiter"),
        draft_backend=None,
        num_steps=None,
        is_hip=True,
        two_batch_overlap=False,
        other_pp_rank=None,
        budgets=None,
        sent_flags=None,
    ):
        """Run _resolve_memory_pool_config with a KV pool of KV_BYTES_PER_TOKEN
        per token; returns the config and the KV budgets it was sized from.

        other_pp_rank: (request_cap or None, does_not_fit, needs_reserve) of a
        second PP rank, merged into this rank's all-reduce. budgets /
        sent_flags: lists that collect the KV budgets and the flags this rank
        sent to the all-reduce, still filled when the run raises."""
        budgets = [] if budgets is None else budgets
        sent_flags = [] if sent_flags is None else sent_flags

        def all_reduce(tensor, op, group):
            self.assertEqual(op, kcc.torch.distributed.ReduceOp.MAX)
            sent_flags.append(tensor.tolist())
            cap, does_not_fit, needs_reserve = other_pp_rank
            other = kcc.torch.tensor(
                [-(cap or 1 << 62), int(does_not_fit), int(needs_reserve)]
            )
            tensor.copy_(kcc.torch.maximum(tensor, other))

        def config_from_budget(self, budget_bytes, *, cap_tokens=None):
            budgets.append(budget_bytes)
            return SimpleNamespace(
                max_total_num_tokens=budget_bytes // KV_BYTES_PER_TOKEN,
                max_running_requests=None,
            )

        schedule = SimpleNamespace(
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            page_size=page_size,
        )

        def calculate_pool_sizes(budget_bytes, page_size):
            tokens = max(budget_bytes, 0) // KV_BYTES_PER_TOKEN
            return pool_configurator.MemoryPoolConfig(
                max_total_num_tokens=tokens // page_size * page_size
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
                    return_value=_parallel(2 if other_pp_rank else 1),
                )
            )
            stack.enter_context(
                patch.object(kcc.torch.distributed, "all_reduce", all_reduce)
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
                    lambda self, pre: (
                        budget_gb * GB if budget_bytes is None else budget_bytes
                    ),
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
                        calculate_pool_sizes=calculate_pool_sizes,
                        finalize_with_max_running_requests=lambda config: config,
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
            patch.object(kcc, "get_parallel", return_value=_parallel()),
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
        # 40 GB of KV resolves to 2048 requests, not 4096. Their workspace
        # (~16 GB) fits in the slack; 4096 requests' (~33 GB) would not.
        kvc = self._configurator()
        config, budgets = self._run(kvc, budget_gb=40)
        self.assertEqual(config.max_running_requests, 2048)
        self.assertLess(2048 * _per_row(_llama_8b()), self._headroom())
        self.assertGreater(4096 * _per_row(_llama_8b()), self._headroom())
        # Nothing reserved: the pool is not re-derived.
        self.assertEqual(budgets, [40 * GB])

    def test_short_context_is_unchanged(self):
        kvc = self._configurator(_llama_8b(context_len=8192))
        config, budgets = self._run(kvc)
        self.assertEqual(budgets, [140 * GB])

    def test_speculative_request_cap_and_draft_backends(self):
        # EAGLE3 sets max_running_requests=48. The draft runner builds its own
        # backend, then two multi-step decode backends (3 steps) and one
        # draft-extend backend: 4 draft workspaces plus the target's. The
        # Llama-3.1-8B EAGLE3 draft checkpoint says 2048 tokens, but its runner
        # is built with the target's 131072, which sizes its workspaces.
        draft = _llama_8b(context_len=2048)
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3, draft_model=draft
        )
        config, budgets = self._run(kvc, max_running_requests=48, num_steps=3)
        self.assertEqual(config.max_running_requests, 48)
        per_row = self._per_row_total(kvc, spec=_spec(num_steps=3))
        self.assertEqual(per_row, 5 * _per_row(_llama_8b()))
        # 5 x 48 x 8.1 MB (~1.9 GB) fits in the slack: the KV budget is untouched.
        self.assertAlmostEqual(48 * per_row / GB, 1.904, places=3)
        self.assertEqual(budgets, [140 * GB])

    def test_draft_workspaces_use_the_target_context_length(self):
        # Heads come from the draft config, the context length from the target,
        # whether the draft checkpoint's own length is smaller or larger.
        target = _llama_8b()
        for draft_context_len in (2048, 1_048_576):
            draft = _llama_8b(num_attention_heads=16, context_len=draft_context_len)
            kvc = self._configurator(
                target, spec_algorithm=SpeculativeAlgorithm.EAGLE, draft_model=draft
            )
            with self.subTest(draft_context_len=draft_context_len):
                self.assertEqual(
                    self._per_row_total(kvc, spec=_spec(num_steps=3)),
                    _per_row(target) + 4 * _per_row(draft, target.context_len),
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
        self.assertEqual(budgets, [140 * GB])
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

    def test_explicit_mem_fraction_skips_the_reserve(self):
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

    def test_adaptive_counts_the_backends_of_every_candidate_step(self):
        # The default adaptive config's candidate steps are 0, 1, 3, 5 and 7.
        # Starting at 3, init_states builds the other four states, each with a
        # target backend, a draft-extend backend and steps - 1 decode backends.
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        self.assertEqual(resolve_candidate_steps_from_config(None), [0, 1, 3, 5, 7])
        draft = _llama_8b(context_len=2048)
        kvc = self._configurator(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3, draft_model=draft
        )
        target_backends = 1 + 4
        # Runner, then (draft-extend + decode) for steps 3, 0, 1, 5 and 7.
        draft_backends = 1 + (1 + 2) + (1 + 0) + (1 + 0) + (1 + 4) + (1 + 6)
        self.assertEqual(
            self._per_row_total(kvc, spec=_spec(num_steps=3, adaptive=True)),
            (target_backends + draft_backends) * _per_row(_llama_8b()),
        )
        # Multi-layer EAGLE never runs adaptive.
        self.assertEqual(
            self._per_row_total(
                kvc, spec=_spec(num_steps=3, multi_layer=True, adaptive=True)
            ),
            self._per_row_total(kvc, spec=_spec(num_steps=3, multi_layer=True)),
        )

    def test_explicit_max_running_requests_is_not_capped(self):
        # 10M context: ~620 MB per request, no slack above the activation
        # reserve at 0.95. 80 requests (~48 GB) are more than half of a 60 GB
        # budget: capped without the flag; kept uncapped with it (with no
        # headroom, the whole workspace is then taken from the KV budget).
        model = _llama_8b(context_len=10_000_000)
        self.assertLessEqual(self._headroom(0.95), 0)
        workspace = 80 * _per_row(model)
        self.assertGreater(workspace, 30 * GB)
        run = dict(budget_gb=60, mem_fraction_static=0.95, max_running_requests=80)
        config, _ = self._run(self._configurator(model), **run)
        self.assertLess(config.max_running_requests, 80)
        config, budgets = self._run(
            self._configurator(model, explicit_max_running_requests=80), **run
        )
        self.assertEqual(config.max_running_requests, 80)
        self.assertEqual(budgets, [60 * GB, 60 * GB - workspace])
        # 120 requests (~73 GB) do not fit in 60 GB.
        run["max_running_requests"] = 120
        with self.assertRaisesRegex(RuntimeError, "smaller --max-running-requests"):
            self._run(
                self._configurator(model, explicit_max_running_requests=120), **run
            )

    def test_pp_ranks_agree_on_the_cap_the_error_and_the_rederive(self):
        model = _llama_8b(context_len=40960)
        kvc = self._configurator(model)
        # Alone, this rank reserves nothing (one workspace fits in the slack).
        self.assertEqual(self._run(kvc)[1], [140 * GB])
        # Another rank reserves: this one re-derives too (config_from_budget
        # all-reduces), from its unchanged budget.
        _, budgets = self._run(kvc, other_pp_rank=(None, False, True))
        self.assertEqual(budgets, [140 * GB, 140 * GB])
        # Another rank caps the requests: this one takes the same cap.
        config, _ = self._run(kvc, other_pp_rank=(100, False, True))
        self.assertEqual(config.max_running_requests, 100)
        # Another rank cannot fit one request: every rank raises.
        with self.assertRaisesRegex(RuntimeError, "another pipeline-parallel rank"):
            self._run(kvc, other_pp_rank=(1, True, True))

    def test_reserve_must_leave_a_minimal_kv_pool(self):
        # 10M context, no slack above the activation reserve at 0.95, 16-token
        # pages. The reserve leaves the KV pool exactly one page, or one byte
        # less: the latter must raise the targeted error up front instead of
        # "Not enough memory" from the re-derived pool.
        model = _llama_8b(context_len=10_000_000)
        per_row = _per_row(model)
        page = 16 * KV_BYTES_PER_TOKEN
        cases = {
            "explicit max_running_requests": (
                dict(max_running_requests=80),
                dict(explicit_max_running_requests=80),
                80 * per_row,
                "smaller --max-running-requests",
            ),
            "cap of one request": ({}, {}, per_row, "single request"),
        }
        for name, (run_kw, kvc_kw, workspace, error) in cases.items():
            run = dict(mem_fraction_static=0.95, page_size=16, **run_kw)
            with self.subTest(name):
                config, budgets = self._run(
                    self._configurator(model, **kvc_kw),
                    budget_bytes=workspace + page,
                    **run,
                )
                self.assertEqual(budgets[1], page)
                with self.assertRaisesRegex(RuntimeError, error):
                    self._run(
                        self._configurator(model, **kvc_kw),
                        budget_bytes=workspace + page - 1,
                        **run,
                    )
            with self.subTest(name, pp=True):
                # With a second PP rank that fits, this rank still reports the
                # failure to the all-reduce, and raises after it without
                # re-deriving its pool (so it never reaches the token
                # all-reduce in config_from_budget alone).
                budgets, sent_flags = [], []
                with self.assertRaisesRegex(
                    RuntimeError, "another pipeline-parallel rank"
                ):
                    self._run(
                        self._configurator(model, **kvc_kw),
                        budget_bytes=workspace + page - 1,
                        other_pp_rank=(None, False, True),
                        budgets=budgets,
                        sent_flags=sent_flags,
                        **run,
                    )
                self.assertEqual(len(sent_flags), 1)
                self.assertEqual(sent_flags[0][1:], [1, 1])
                self.assertEqual(budgets, [workspace + page - 1])


if __name__ == "__main__":
    unittest.main()
