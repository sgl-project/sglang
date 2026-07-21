"""Unit tests for the model-override machinery: whitelist metadata, registry,
gate, publish wiring, and the per-arch golden diffs for migrated families."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.arg_groups.arg_utils import A, Arg, resolvable_fields
from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    register_model_override,
    validate_declarations,
)
from sglang.srt.runtime_context import (
    get_context,
    get_server_args,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase


@dataclasses.dataclass
class _FakeArgs:
    plain: A[int, "help text only"] = 0
    resolved_by_model: A[str, Arg(help="x", resolvable=True)] = "auto"
    also_resolved: A[Optional[int], Arg(help="y", resolvable=True)] = None
    metadata_but_not_overridable: A[bool, Arg(help="z")] = False


class TestModelOverridableWhitelist(CustomTestCase):
    def test_whitelist_derivation_from_annotated_metadata(self):
        self.assertEqual(
            resolvable_fields(_FakeArgs),
            frozenset({"resolved_by_model", "also_resolved"}),
        )

    def test_server_args_whitelist_is_exactly_the_migrated_fields(self):
        # Fields are whitelisted one family at a time by the migration
        # sweeps. This pin makes accidental tagging visible — extend it in
        # the same commit that tags a new field.
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            resolvable_fields(ServerArgs),
            frozenset(
                {
                    "dtype",
                    "enable_tf32_matmul",
                    "enable_multi_layer_eagle",
                    "swa_full_tokens_ratio",
                    "disable_hybrid_swa_memory",
                    "sampling_backend",
                    "attention_backend",
                    "page_size",
                    "moe_runner_backend",
                    "quantization",
                    "enable_dp_attention",
                    "enable_dp_lm_head",
                    "moe_a2a_backend",
                    "ep_size",
                    "moe_dense_tp_size",
                    "attn_cp_size",
                    "disable_overlap_schedule",
                    "uses_mamba_radix_cache",
                    "mamba_radix_cache_strategy",
                    "mamba_full_memory_ratio",
                    "speculative_moe_runner_backend",
                    "speculative_moe_a2a_backend",
                    "disable_shared_experts_fusion",
                    "kv_cache_dtype",
                    "dsa_prefill_backend",
                    "dsa_decode_backend",
                    "prefill_attention_backend",
                    "decode_attention_backend",
                    "flashinfer_allreduce_fusion_backend",
                    "fp8_gemm_runner_backend",
                    "disable_custom_all_reduce",
                    "enable_aiter_allreduce_fusion",
                }
            ),
        )


class _IsolatedRegistry(CustomTestCase):
    """Run each test against empty registries (they are process-global)."""

    def setUp(self):
        super().setUp()
        self._patches = [
            patch.dict(overrides_module.MODEL_OVERRIDES, clear=True),
            patch.dict(overrides_module._MODEL_OVERRIDE_FNS, clear=True),
            patch.object(overrides_module, "_PREDICATE_OVERRIDE_FNS", []),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()


class TestModelOverrideRegistry(_IsolatedRegistry):
    def test_const_then_callables_in_registration_order(self):
        overrides_module.MODEL_OVERRIDES["FakeForCausalLM"] = {"a": 1}

        @register_model_override("FakeForCausalLM")
        def _first(server_args, hf_config):
            return {"b": server_args.base + 1}

        @register_model_override("FakeForCausalLM")
        def _second(server_args, hf_config):
            return {"a": 3}

        declarations = collect_model_override_declarations(
            "FakeForCausalLM", SimpleNamespace(base=10), hf_config=None
        )
        self.assertEqual(
            declarations,
            [
                ("MODEL_OVERRIDES['FakeForCausalLM']", {"a": 1}),
                (_first.__qualname__, {"b": 11}),
                (_second.__qualname__, {"a": 3}),
            ],
        )

    def test_unknown_architecture_yields_nothing(self):
        self.assertEqual(
            collect_model_override_declarations("NoSuchArch", None, None), []
        )

    def test_empty_declarations_are_dropped(self):
        @register_model_override("FakeForCausalLM")
        def _nothing_applies(server_args, hf_config):
            return {}

        self.assertEqual(
            collect_model_override_declarations("FakeForCausalLM", None, None), []
        )

    def test_non_dict_return_is_rejected(self):
        @register_model_override("FakeForCausalLM")
        def _bad(server_args, hf_config):
            return None

        with self.assertRaises(TypeError):
            collect_model_override_declarations("FakeForCausalLM", None, None)

    def test_predicate_keyed_provider(self):
        from sglang.srt.arg_groups.overrides import register_model_override_predicate

        @register_model_override("FakeStep9ForCausalLM")
        def _exact(server_args, hf_config):
            return {"a": 1}

        @register_model_override_predicate(lambda arch: "Step9" in arch)
        def _by_predicate(server_args, hf_config):
            return {"b": 2}

        # matching arch: exact-keyed first, then predicate-keyed
        self.assertEqual(
            collect_model_override_declarations("FakeStep9ForCausalLM", None, None),
            [(_exact.__qualname__, {"a": 1}), (_by_predicate.__qualname__, {"b": 2})],
        )
        # non-matching arch: predicate does not fire
        self.assertEqual(
            collect_model_override_declarations("OtherForCausalLM", None, None), []
        )


class TestResolvedViewAndPasses(CustomTestCase):
    """Pipeline skeleton: read-only view semantics + transition invocation."""

    def test_view_forwards_reads_and_rejects_writes(self):
        from sglang.srt.arg_groups.overrides import ResolvedView

        live = SimpleNamespace(a=1, method=lambda: "m")
        view = ResolvedView(live)
        self.assertEqual(view.a, 1)
        self.assertEqual(view.method(), "m")  # method forwarding
        live.a = 2
        self.assertEqual(view.a, 2)  # live, not a snapshot
        with self.assertRaises(AttributeError):
            view.a = 3

    def test_view_overlay_wins(self):
        from sglang.srt.arg_groups.overrides import ResolvedView

        view = ResolvedView(SimpleNamespace(a=1, b=2), overlay={"a": 10})
        self.assertEqual(view.a, 10)
        self.assertEqual(view.b, 2)

    def test_run_pass_appends_stash_and_stays_pristine(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        live = SimpleNamespace(x=None, _resolved_overrides=[])

        def _fill_x(view):
            return {"x": "filled"} if view.x is None else {}

        run_post_process_pass(live, _fill_x)
        self.assertIsNone(live.x)  # never applied in place
        self.assertEqual(
            live._resolved_overrides, [(_fill_x.__qualname__, {"x": "filled"})]
        )
        # the next invocation sees the declared value through the overlay
        run_post_process_pass(live, _fill_x)
        self.assertEqual(len(live._resolved_overrides), 1)

    def test_run_pass_rejects_non_dict(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        with self.assertRaises(TypeError):
            run_post_process_pass(
                SimpleNamespace(_resolved_overrides=[]), lambda view: None
            )


class _IsolatedPublish(CustomTestCase):
    """Publishing writes the process-global context; save/restore around it."""

    def setUp(self):
        super().setUp()
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        reset_context()
        if self._saved_server_args is not None:
            get_context()._server_args = self._saved_server_args
        super().tearDown()


class TestPublishInstallsSlot(_IsolatedPublish):
    """Publish wiring: set_server_args installs the already-resolved object
    into the context-owned slot (no transformation at publish time)."""

    def test_dummy_fixture_has_empty_stash_and_publishes_cleanly(self):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        sa = ServerArgs(model_path="dummy")  # __post_init__ early-returns
        # The stash is created before the dummy short-circuit and stays empty.
        self.assertEqual(sa._resolved_overrides, [])
        set_global_server_args_for_scheduler(sa)
        self.assertIs(get_server_args(), sa)


class TestGoldenModelOverrides(_IsolatedPublish):
    """Per-arch golden diff for migrated families: the declarative path must
    reproduce the legacy imperative writes byte-identically on the
    materialized server_args fields; the publish round-trip returns the same
    object."""

    _MINI_CONFIG = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 512,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "torch_dtype": "bfloat16",
        # MLA shape fields (required by the MistralLarge3/Pixtral arch
        # family; inert extras for non-MLA control arches).
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
    }

    def _construct(self, arch, model_type, config_extra=None, **server_kwargs):
        from sglang.srt.server_args import ServerArgs

        # Golden resolution must be host-independent: accelerator-less CI
        # runners resolve only the base platform, where get_device() raises.
        server_kwargs.setdefault("device", "cuda")
        config = dict(self._MINI_CONFIG, architectures=[arch], model_type=model_type)
        config.update(config_extra or {})
        config_dir = tempfile.mkdtemp(prefix="golden_override_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(config, f)
        return ServerArgs(model_path=config_dir, **server_kwargs)

    def _publish(self, server_args):
        from sglang.srt.server_args import (
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(server_args)
        return get_server_args()

    def test_mistral_large3_forces_bfloat16(self):
        sa = self._construct("MistralLarge3ForCausalLM", "mistral")
        self.assertEqual(sa.dtype, "bfloat16")  # materialized at end of resolution
        self.assertIn(
            ("MODEL_OVERRIDES['MistralLarge3ForCausalLM']", {"dtype": "bfloat16"}),
            sa._resolved_overrides,
        )
        self.assertEqual(self._publish(sa).dtype, "bfloat16")

    def test_user_requested_dtype_is_still_overridden(self):
        # Legacy fidelity: the arch branch overwrote dtype unconditionally,
        # so the declaration must too. The pristine request survives on
        # provenance; the materialized field carries the override.
        sa = self._construct("MistralLarge3ForCausalLM", "mistral", dtype="float16")
        self.assertEqual(sa.dtype, "bfloat16")  # materialized
        self.assertEqual(self._publish(sa).dtype, "bfloat16")

    def test_control_arch_keeps_pristine_dtype(self):
        sa = self._construct("LlamaForCausalLM", "llama")
        self.assertEqual(sa.dtype, "auto")
        declared = {f for _s, d in sa._resolved_overrides for f in d}
        self.assertNotIn("dtype", declared)  # no arch declaration for Llama
        # publish still materializes the whitelisted leaf with the pristine
        # value: readers only ever read flags.
        self.assertEqual(self._publish(sa).dtype, "auto")

    def test_minimax_m2_enables_tf32_matmul(self):
        sa = self._construct("MiniMaxM2ForCausalLM", "llama")
        self.assertTrue(sa.enable_tf32_matmul)  # materialized
        self.assertIn(
            ("_minimax_m2_overrides", {"enable_tf32_matmul": True}),
            sa._resolved_overrides,
        )
        flags = self._publish(sa)
        self.assertTrue(flags.enable_tf32_matmul)
        self.assertFalse(flags.enable_multi_layer_eagle)  # pristine materialize

    def test_mimo_v2_declarations(self):
        # Callable-level golden: MiMoV2 archs are hybrid (config-shape heavy),
        # so the declaration is pinned directly for both provider inputs.
        from sglang.srt.arg_groups.overrides import _mimo_v2_overrides

        self.assertEqual(
            _mimo_v2_overrides(SimpleNamespace(speculative_algorithm="EAGLE"), None),
            {"enable_multi_layer_eagle": True},
        )
        self.assertEqual(
            _mimo_v2_overrides(SimpleNamespace(speculative_algorithm=None), None),
            {},
        )

    def test_mimo_v2_family_is_registered(self):
        self.assertEqual(
            collect_model_override_declarations(
                "MiMoV2FlashForCausalLM",
                SimpleNamespace(speculative_algorithm="EAGLE"),
                None,
            ),
            [("_mimo_v2_overrides", {"enable_multi_layer_eagle": True})],
        )

    def test_step3p_hierarchical_cache_golden(self):
        # SWA-hybrid arch: the mini config needs layer_types/sliding_window.
        config_extra = {
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 64,
        }
        sa = self._construct(
            "Step3p5ForCausalLM",
            "llama",
            config_extra=config_extra,
            enable_hierarchical_cache=True,
        )
        # materialized at the end of resolution
        self.assertEqual(sa.swa_full_tokens_ratio, 1.0)
        self.assertTrue(sa.disable_hybrid_swa_memory)
        flags = self._publish(sa)
        self.assertEqual(flags.swa_full_tokens_ratio, 1.0)
        self.assertTrue(flags.disable_hybrid_swa_memory)

    def test_gemma2_disables_hybrid_swa_memory(self):
        sa = self._construct("Gemma2ForCausalLM", "llama")
        self.assertTrue(sa.disable_hybrid_swa_memory)  # materialized
        self.assertIn(
            ("_gemma2_gemma3_overrides", {"disable_hybrid_swa_memory": True}),
            sa._resolved_overrides,
        )
        self.assertTrue(self._publish(sa).disable_hybrid_swa_memory)

    def test_olmo2_disables_hybrid_swa_memory(self):
        sa = self._construct("Olmo2ForCausalLM", "llama")
        self.assertTrue(sa.disable_hybrid_swa_memory)  # materialized
        self.assertTrue(self._publish(sa).disable_hybrid_swa_memory)

    def test_exaone_conditional_on_sliding_window_pattern(self):
        # With the pattern the branch also asserts an explicit backend.
        sa = self._construct(
            "Exaone4ForCausalLM",
            "llama",
            config_extra={"sliding_window_pattern": "LLLG"},
            attention_backend="fa3",
        )
        self.assertTrue(sa.disable_hybrid_swa_memory)  # materialized
        self.assertTrue(self._publish(sa).disable_hybrid_swa_memory)

    def test_exaone_without_pattern_declares_nothing(self):
        from sglang.srt.arg_groups.overrides import _exaone_overrides

        self.assertEqual(
            _exaone_overrides(None, SimpleNamespace(sliding_window_pattern=None)),
            {},
        )

    def test_gpt_oss_mxfp4_forces_bfloat16(self):
        from sglang.srt.layers.quantization import QUANTIZATION_METHODS

        if "mxfp4" not in QUANTIZATION_METHODS:
            # Registration is platform-gated (CUDA / CPU engine / MXFP-HIP);
            # plain CPU CI runners cannot construct an mxfp4 ModelConfig.
            self.skipTest("mxfp4 quantization is not registered on this platform")
        sa = self._construct(
            "GptOssForCausalLM",
            "llama",
            config_extra={"quantization_config": {"quant_method": "mxfp4"}},
        )
        self.assertEqual(sa.dtype, "bfloat16")  # materialized
        self.assertEqual(self._publish(sa).dtype, "bfloat16")

    def test_gpt_oss_without_mxfp4_keeps_pristine_dtype(self):
        sa = self._construct("GptOssForCausalLM", "llama")
        self.assertEqual(sa.dtype, "auto")
        self.assertEqual(self._publish(sa).dtype, "auto")

    def test_gpt_oss_xpu_dtype_validation_reads_pristine(self):
        from sglang.srt.arg_groups.overrides import _gpt_oss_overrides

        with patch.object(overrides_module, "is_xpu", return_value=True):
            with self.assertRaises(NotImplementedError):
                _gpt_oss_overrides(
                    SimpleNamespace(
                        dtype="float16",
                        is_attention_backend_not_set=lambda: False,
                    ),
                    SimpleNamespace(architectures=["GptOssForCausalLM"]),
                )

    def test_sampling_backend_default_pass(self):
        from sglang.srt.utils.common import is_flashinfer_available

        sa = self._construct("LlamaForCausalLM", "llama")
        expected = "flashinfer" if is_flashinfer_available() else "pytorch"
        self.assertEqual(sa.sampling_backend, expected)  # materialized
        self.assertIn(
            ("_sampling_backend_default", {"sampling_backend": expected}),
            sa._resolved_overrides,
        )
        self.assertEqual(self._publish(sa).sampling_backend, expected)

    def test_sampling_backend_user_choice_survives(self):
        sa = self._construct("LlamaForCausalLM", "llama", sampling_backend="pytorch")
        self.assertEqual(sa.sampling_backend, "pytorch")
        # the pass declared nothing; publish materializes the pristine choice
        self.assertEqual(self._publish(sa).sampling_backend, "pytorch")

    def test_deterministic_inference_forces_pytorch_sampling(self):
        sa = self._construct(
            "LlamaForCausalLM", "llama", enable_deterministic_inference=True
        )
        # two pass writers chain: default fill, then the deterministic force —
        # last writer wins; materialization lands the end state on the fields.
        self.assertEqual(sa.sampling_backend, "pytorch")
        flags = self._publish(sa)
        self.assertEqual(flags.sampling_backend, "pytorch")
        # the deterministic attention fill declared a compatible backend and
        # the compatibility default-fill then had nothing to do
        deterministic_fills = [
            decl["attention_backend"]
            for source, decl in sa._resolved_overrides
            if source == "_deterministic_attention_backend"
        ]
        self.assertEqual(len(deterministic_fills), 1)
        self.assertEqual(sa.attention_backend, deterministic_fills[0])
        self.assertEqual(flags.attention_backend, deterministic_fills[0])

    def test_deterministic_incompatible_backend_raises(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deterministic_attention_backend,
        )

        view = ResolvedView(
            SimpleNamespace(
                enable_deterministic_inference=True, attention_backend="flashmla"
            )
        )
        with self.assertRaises(ValueError):
            _deterministic_attention_backend(view)

    def test_deterministic_ascend_is_left_alone(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deterministic_sampling_backend,
        )

        view = ResolvedView(
            SimpleNamespace(
                enable_deterministic_inference=True, sampling_backend="ascend"
            )
        )
        self.assertEqual(_deterministic_sampling_backend(view), {})

    def test_dllm_forces_flashinfer_with_cuda_graph(self):
        # CUDA path: cuda graph enabled by default -> dllm forces flashinfer.
        # A real dllm arch: the page pass now runs regardless of the radix
        # switch and builds DllmConfig for it.
        sa = self._construct(
            "SDARForCausalLM",
            "llama",
            dllm_algorithm="LowConfidence",
            disable_radix_cache=True,
            attention_backend="triton",
        )
        self.assertEqual(sa.attention_backend, "flashinfer")  # materialized
        self.assertIn(
            ("_dllm_attention_backend", {"attention_backend": "flashinfer"}),
            sa._resolved_overrides,
        )
        # the deterministic fill lands on the attention_backend field
        self.assertEqual(self._publish(sa).attention_backend, "flashinfer")

    def test_attention_backend_leaf_materializes_end_state(self):
        # The default-fill pass declares the platform-selected backend; the
        # leaf must equal the last declared value while the server_args field
        # stays pristine (dual-apply retired).
        sa = self._construct("LlamaForCausalLM", "llama")
        declared_values = [
            d["attention_backend"]
            for _s, d in sa._resolved_overrides
            if "attention_backend" in d
        ]
        self.assertTrue(declared_values)  # default fill declared
        self.assertEqual(sa.attention_backend, declared_values[-1])  # materialized
        self.assertEqual(self._publish(sa).attention_backend, declared_values[-1])

    def test_post_materialize_pass_writes_through(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        # A pass invoked after materialization (a post-init slot, like the
        # legacy runner-side adjustments) declares AND writes through, so
        # field readers and the publish see the same end state.
        sa = self._construct("LlamaForCausalLM", "llama")
        resolved_before = sa.attention_backend

        def _force_triton(view):
            if view.attention_backend != "triton":
                return {"attention_backend": "triton"}
            return {}

        run_post_process_pass(sa, _force_triton)
        if resolved_before != "triton":
            self.assertEqual(sa.attention_backend, "triton")
        self.assertEqual(self._publish(sa).attention_backend, sa.attention_backend)

    def test_attention_backend_user_choice_declares_nothing_extra(self):
        sa = self._construct("LlamaForCausalLM", "llama", attention_backend="triton")
        self.assertEqual(sa.attention_backend, "triton")
        self.assertEqual(self._publish(sa).attention_backend, "triton")

    def test_compatibility_passes_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _attention_backend_default,
            _attention_backend_dual_chunk,
            _attention_backend_fa3_fp8_fallback,
            _attention_backend_platform_fallbacks,
        )

        # split-backend override wins over the default fill
        view = ResolvedView(
            SimpleNamespace(
                prefill_attention_backend="fa3",
                decode_attention_backend="fa3",
                attention_backend=None,
            )
        )
        self.assertEqual(_attention_backend_default(view), {"attention_backend": "fa3"})

        # fa3 + fp8_e5m2 falls back to triton
        view = ResolvedView(
            SimpleNamespace(attention_backend="fa3", kv_cache_dtype="fp8_e5m2")
        )
        self.assertEqual(
            _attention_backend_fa3_fp8_fallback(view),
            {"attention_backend": "triton"},
        )

        # amx fallback fires only without hardware support
        view = ResolvedView(
            SimpleNamespace(attention_backend="intel_amx", device="cpu")
        )
        with patch.object(overrides_module, "cpu_has_amx_support", return_value=False):
            self.assertEqual(
                _attention_backend_platform_fallbacks(view),
                {"attention_backend": "torch_native"},
            )
        with patch.object(overrides_module, "cpu_has_amx_support", return_value=True):
            self.assertEqual(_attention_backend_platform_fallbacks(view), {})

        # dual-chunk config: mismatched explicit backend raises verbatim
        def _mc(dual):
            return SimpleNamespace(
                get_model_config=lambda: SimpleNamespace(
                    hf_config=SimpleNamespace(dual_chunk_attention_config=dual)
                ),
                attention_backend="fa3",
            )

        with self.assertRaises(ValueError):
            _attention_backend_dual_chunk(ResolvedView(_mc({"a": 1})))
        self.assertEqual(_attention_backend_dual_chunk(ResolvedView(_mc(None))), {})

    def test_dllm_platform_paths_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dllm_attention_backend,
        )
        from sglang.srt.model_executor.cuda_graph_config import Backend

        def _view(**kw):
            defaults = dict(
                dllm_algorithm="LowConfidence",
                attention_backend=None,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend=Backend.DISABLED)
                ),
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        with patch.object(overrides_module, "is_hip", return_value=True):
            self.assertEqual(
                _dllm_attention_backend(_view()), {"attention_backend": "triton"}
            )
            self.assertEqual(
                _dllm_attention_backend(_view(attention_backend="aiter")), {}
            )
        with patch.object(overrides_module, "is_hip", return_value=False):
            with patch.object(overrides_module, "is_npu", return_value=True):
                self.assertEqual(
                    _dllm_attention_backend(_view()),
                    {"attention_backend": "ascend"},
                )
            with patch.object(overrides_module, "is_npu", return_value=False):
                # cuda graph disabled -> nothing to force
                self.assertEqual(_dllm_attention_backend(_view()), {})
                self.assertEqual(
                    _dllm_attention_backend(_view(dllm_algorithm=None)), {}
                )

    def test_page_size_default_pass(self):
        from sglang.srt.arg_groups.overrides import ResolvedView, _page_size_default

        # user-set page_size: nothing to declare
        self.assertEqual(
            _page_size_default(ResolvedView(SimpleNamespace(page_size=64))), {}
        )
        # default fill on non-HIP/non-MUSA platforms is 1
        with patch.object(overrides_module, "is_hip", return_value=False):
            with patch.object(overrides_module, "is_musa", return_value=False):
                self.assertEqual(
                    _page_size_default(ResolvedView(SimpleNamespace(page_size=None))),
                    {"page_size": 1},
                )
            with patch.object(overrides_module, "is_musa", return_value=True):
                self.assertEqual(
                    _page_size_default(ResolvedView(SimpleNamespace(page_size=None))),
                    {"page_size": 64},
                )

    def test_dllm_page_size_pass(self):
        from sglang.srt.arg_groups.overrides import ResolvedView, _dllm_page_size

        def _view(**kw):
            defaults = dict(
                dllm_algorithm="LowConfidence", disable_radix_cache=False, page_size=1
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        with patch(
            "sglang.srt.dllm.config.DllmConfig.from_server_args",
            return_value=SimpleNamespace(block_size=32),
        ):
            self.assertEqual(_view() and _dllm_page_size(_view()), {"page_size": 32})
            # aligned but larger than the block: the scheduler-init fallback
            # (folded into this pass) still caps the page at the block size
            self.assertEqual(_dllm_page_size(_view(page_size=64)), {"page_size": 32})
            self.assertEqual(_dllm_page_size(_view(page_size=32)), {})  # equal
            # radix disabled skips the alignment fill but keeps the cap
            self.assertEqual(_dllm_page_size(_view(disable_radix_cache=True)), {})
            self.assertEqual(
                _dllm_page_size(_view(disable_radix_cache=True, page_size=64)),
                {"page_size": 32},
            )
        self.assertEqual(_dllm_page_size(_view(dllm_algorithm=None)), {})

    def test_overlap_disable_passes(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dllm_overlap_disable,
            _pipeline_parallel_overlap_disable,
            _sparse_head_overlap_disable,
        )

        # pipeline parallelism: declares only when pp_size > 1
        self.assertEqual(
            _pipeline_parallel_overlap_disable(
                ResolvedView(SimpleNamespace(pp_size=1))
            ),
            {},
        )
        self.assertEqual(
            _pipeline_parallel_overlap_disable(
                ResolvedView(SimpleNamespace(pp_size=2))
            ),
            {"disable_overlap_schedule": True},
        )

        # dllm: guarded on the algorithm and the current value
        def _view(**kw):
            defaults = dict(
                dllm_algorithm="LowConfidence", disable_overlap_schedule=False
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        self.assertEqual(_dllm_overlap_disable(_view(dllm_algorithm=None)), {})
        self.assertEqual(
            _dllm_overlap_disable(_view(disable_overlap_schedule=True)), {}
        )
        self.assertEqual(
            _dllm_overlap_disable(_view()), {"disable_overlap_schedule": True}
        )

        # embeddings sparse head: keyed on the env var being set
        from sglang.srt.environ import envs

        view = ResolvedView(SimpleNamespace())
        with patch.object(
            envs.SGLANG_EMBEDDINGS_SPARSE_HEAD, "is_set", return_value=False
        ):
            self.assertEqual(_sparse_head_overlap_disable(view), {})
        with patch.object(
            envs.SGLANG_EMBEDDINGS_SPARSE_HEAD, "is_set", return_value=True
        ):
            self.assertEqual(
                _sparse_head_overlap_disable(view), {"disable_overlap_schedule": True}
            )

    def test_deepseek_v4_overrides_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import _deepseek_v4_overrides
        from sglang.srt.server_args import ServerArgs

        hf = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])

        def _args(**kw):
            defaults = dict(
                device="cuda",
                swa_full_tokens_ratio=ServerArgs.swa_full_tokens_ratio,
                moe_runner_backend="auto",
                get_model_config=lambda: SimpleNamespace(nvfp4_moe_meta=None),
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        self.assertEqual(
            _deepseek_v4_overrides(_args(), hf),
            {
                "attention_backend": "dsv4",
                "page_size": 256,
                "swa_full_tokens_ratio": 0.1,
            },
        )
        # NPU pool geometry
        self.assertEqual(
            _deepseek_v4_overrides(_args(device="npu"), hf)["page_size"], 128
        )
        # user-set window ratio survives
        self.assertNotIn(
            "swa_full_tokens_ratio",
            _deepseek_v4_overrides(_args(swa_full_tokens_ratio=0.5), hf),
        )
        # nvfp4 hybrid checkpoint routes the MoE runner
        self.assertEqual(
            _deepseek_v4_overrides(
                _args(
                    get_model_config=lambda: SimpleNamespace(nvfp4_moe_meta=object())
                ),
                hf,
            )["moe_runner_backend"],
            "flashinfer_trtllm_routed",
        )

    def test_deepseek_v4_sm120_moe_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deepseek_v4_sm120_moe,
        )

        def _view(arch="DeepseekV4ForCausalLM", **kw):
            hf = SimpleNamespace(architectures=[arch])
            defaults = dict(moe_runner_backend="auto")
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        with patch.object(overrides_module, "is_sm120_supported", return_value=True):
            self.assertEqual(
                _deepseek_v4_sm120_moe(_view()),
                {"moe_runner_backend": "flashinfer_mxfp4"},
            )
            self.assertEqual(
                _deepseek_v4_sm120_moe(_view(moe_runner_backend="triton")), {}
            )
            self.assertEqual(_deepseek_v4_sm120_moe(_view(arch="LlamaForCausalLM")), {})
        with patch.object(overrides_module, "is_sm120_supported", return_value=False):
            self.assertEqual(_deepseek_v4_sm120_moe(_view()), {})

    def test_nemotron_h_overrides_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        def _hf(quant_algo="NVFP4"):
            return SimpleNamespace(
                architectures=["NemotronHForCausalLM"],
                mlp_hidden_act="relu2",
                quantization_config={"quant_algo": quant_algo},
            )

        def _args(mc_quant, hf, **kw):
            mc = SimpleNamespace(quantization=mc_quant, hf_config=hf)
            defaults = dict(
                quantization=None,
                moe_runner_backend="auto",
                moe_a2a_backend="none",
                attention_backend=None,
                get_model_config=lambda: mc,
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        hf = _hf()
        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            # modelopt checkpoint: quant algo resolution + sm100 defaults
            self.assertEqual(
                _nemotron_h_overrides(_args("modelopt", hf), hf),
                {
                    "quantization": "modelopt_fp4",
                    "moe_runner_backend": "flashinfer_trtllm",
                    "attention_backend": "flashinfer",
                },
            )
            hf_mixed = _hf("MIXED_PRECISION")
            self.assertEqual(
                _nemotron_h_overrides(_args("modelopt", hf_mixed), hf_mixed)[
                    "quantization"
                ],
                "modelopt_mixed",
            )
        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
            patch.object(overrides_module, "is_cuda", return_value=True),
            patch.object(
                overrides_module, "get_device_capability", return_value=(9, 0)
            ),
        ):
            # SM80-SM90 fp4: marlin
            self.assertEqual(
                _nemotron_h_overrides(_args("modelopt_fp4", hf), hf),
                {"quantization": "modelopt_fp4", "moe_runner_backend": "marlin"},
            )
            # unquantized checkpoint: cutlass fallback, no quant declared
            self.assertEqual(
                _nemotron_h_overrides(_args(None, hf), hf),
                {"moe_runner_backend": "flashinfer_cutlass"},
            )
            # non-modelopt quantized checkpoint: nothing declared
            self.assertEqual(_nemotron_h_overrides(_args("fp8", hf), hf), {})
            # user-set moe backend survives
            self.assertEqual(
                _nemotron_h_overrides(_args(None, hf, moe_runner_backend="triton"), hf),
                {},
            )

    def test_speculative_moe_runner_default_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _speculative_moe_runner_default,
        )

        self.assertEqual(
            _speculative_moe_runner_default(
                ResolvedView(
                    SimpleNamespace(
                        speculative_moe_runner_backend=None, moe_runner_backend="triton"
                    )
                )
            ),
            {"speculative_moe_runner_backend": "triton"},
        )
        # user-set draft backend survives
        self.assertEqual(
            _speculative_moe_runner_default(
                ResolvedView(
                    SimpleNamespace(
                        speculative_moe_runner_backend="deep_gemm",
                        moe_runner_backend="auto",
                    )
                )
            ),
            {},
        )

    def test_dsa_split_backend_resolution_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dsa_split_backend_resolution,
        )

        def _view(arch="DeepseekV32ForCausalLM", **kw):
            hf = SimpleNamespace(architectures=[arch])
            defaults = dict(
                kv_cache_dtype="fp8_e4m3",
                dsa_prefill_backend=None,
                dsa_decode_backend=None,
                enable_hisparse=False,
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch.object(overrides_module, "is_npu", return_value=False),
            patch.object(overrides_module, "is_xpu", return_value=False),
            patch.object(overrides_module, "is_hip", return_value=False),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            # Hopper FP8 -> flashmla_kv both
            self.assertEqual(
                _dsa_split_backend_resolution(_view()),
                {
                    "dsa_prefill_backend": "flashmla_kv",
                    "dsa_decode_backend": "flashmla_kv",
                },
            )
            # Hopper bf16 -> flashmla_sparse / fa3
            self.assertEqual(
                _dsa_split_backend_resolution(_view(kv_cache_dtype="bfloat16")),
                {
                    "dsa_prefill_backend": "flashmla_sparse",
                    "dsa_decode_backend": "fa3",
                },
            )
            # user-set prefill survives; only decode defaulted
            self.assertEqual(
                _dsa_split_backend_resolution(_view(dsa_prefill_backend="trtllm")),
                {"dsa_decode_backend": "flashmla_kv"},
            )
            # hisparse arm takes precedence (CUDA fp8 -> flashmla_kv)
            self.assertEqual(
                _dsa_split_backend_resolution(_view(enable_hisparse=True)),
                {
                    "dsa_prefill_backend": "flashmla_kv",
                    "dsa_decode_backend": "flashmla_kv",
                },
            )
            # non-family arch declares nothing
            self.assertEqual(
                _dsa_split_backend_resolution(_view(arch="LlamaForCausalLM")), {}
            )
        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch.object(overrides_module, "is_npu", return_value=False),
            patch.object(overrides_module, "is_xpu", return_value=False),
            patch.object(overrides_module, "is_hip", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 4)),
        ):
            # ROCm with both unset -> tilelang
            self.assertEqual(
                _dsa_split_backend_resolution(_view(kv_cache_dtype="bfloat16")),
                {
                    "dsa_prefill_backend": "tilelang",
                    "dsa_decode_backend": "tilelang",
                },
            )

    def test_flashinfer_allreduce_fusion_passes(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deterministic_allreduce_fusion_disable,
            _enforce_disable_allreduce_fusion,
            _flashinfer_allreduce_fusion_auto_enable,
        )

        def _view(arch="Qwen3MoeForCausalLM", **kw):
            hf = SimpleNamespace(architectures=[arch])
            defaults = dict(
                flashinfer_allreduce_fusion_backend=None,
                tp_size=2,
                enable_dp_attention=False,
                nnodes=1,
                moe_a2a_backend="none",
                enforce_disable_flashinfer_allreduce_fusion=False,
                enable_deterministic_inference=False,
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        with (
            patch.object(overrides_module, "is_sm90_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
        ):
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(_view()),
                {"flashinfer_allreduce_fusion_backend": "auto"},
            )
            # guards: unsupported arch / tp==1 / dp attention / a2a backend
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(
                    _view(arch="LlamaForCausalLM")
                ),
                {},
            )
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(_view(tp_size=1)), {}
            )
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(
                    _view(enable_dp_attention=True)
                ),
                {},
            )
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(
                    _view(moe_a2a_backend="deepep")
                ),
                {},
            )
            # SM90 multi-node: blocked (nnodes>1 needs SM100)
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(_view(nnodes=2)), {}
            )
            # user-set backend survives
            self.assertEqual(
                _flashinfer_allreduce_fusion_auto_enable(
                    _view(flashinfer_allreduce_fusion_backend="trtllm")
                ),
                {},
            )

        # enforce-disable wins over everything
        self.assertEqual(
            _enforce_disable_allreduce_fusion(
                _view(
                    flashinfer_allreduce_fusion_backend="auto",
                    enforce_disable_flashinfer_allreduce_fusion=True,
                )
            ),
            {"flashinfer_allreduce_fusion_backend": None},
        )
        self.assertEqual(_enforce_disable_allreduce_fusion(_view()), {})

        # deterministic inference disables an enabled fusion
        self.assertEqual(
            _deterministic_allreduce_fusion_disable(
                _view(
                    flashinfer_allreduce_fusion_backend="auto",
                    enable_deterministic_inference=True,
                )
            ),
            {"flashinfer_allreduce_fusion_backend": None},
        )
        self.assertEqual(
            _deterministic_allreduce_fusion_disable(
                _view(enable_deterministic_inference=True)
            ),
            {},
        )

    def test_cutedsl_prefill_backend_fill_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _cutedsl_prefill_backend_fill,
        )

        def _view(**kw):
            defaults = dict(
                attention_backend=None,
                decode_attention_backend="cutedsl_mla",
                prefill_attention_backend=None,
                kv_cache_dtype="auto",
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            # decode-only cutedsl: prefill defaults to trtllm_mla
            self.assertEqual(
                _cutedsl_prefill_backend_fill(_view()),
                {"prefill_attention_backend": "trtllm_mla"},
            )
            # user-set prefill survives
            self.assertEqual(
                _cutedsl_prefill_backend_fill(_view(prefill_attention_backend="fa3")),
                {},
            )
            # cutedsl on the prefill side is rejected
            with self.assertRaises(AssertionError):
                _cutedsl_prefill_backend_fill(
                    _view(prefill_attention_backend="cutedsl_mla")
                )
            # unsupported kv dtype rejected
            with self.assertRaises(ValueError):
                _cutedsl_prefill_backend_fill(_view(kv_cache_dtype="fp8_e5m2"))
            # not a cutedsl config: nothing declared
            self.assertEqual(
                _cutedsl_prefill_backend_fill(_view(decode_attention_backend=None)),
                {},
            )
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            with self.assertRaises(ValueError):
                _cutedsl_prefill_backend_fill(_view())

    def test_moss_vl_overrides_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import _moss_vl_overrides

        def _args(**kw):
            defaults = dict(
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
            )
            defaults.update(kw)
            ns = SimpleNamespace(**defaults)
            ns.is_attention_backend_not_set = lambda: (
                ns.attention_backend is None
                and ns.prefill_attention_backend is None
                and ns.decode_attention_backend is None
            )
            ns.get_attention_backends = lambda: (
                ns.prefill_attention_backend or ns.attention_backend,
                ns.decode_attention_backend or ns.attention_backend,
            )
            return ns

        # nothing set: prefill defaults to flashinfer
        self.assertEqual(
            _moss_vl_overrides(_args(), None),
            {"prefill_attention_backend": "flashinfer"},
        )
        # compatible user choice passes with no declaration
        self.assertEqual(
            _moss_vl_overrides(_args(attention_backend="flashinfer"), None), {}
        )
        # incompatible user choice rejected
        with self.assertRaises(AssertionError):
            _moss_vl_overrides(_args(attention_backend="fa3"), None)

    def test_dsa_kv_cache_dtype_default_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dsa_kv_cache_dtype_default,
        )

        def _view(**kw):
            hf = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"])
            defaults = dict(
                kv_cache_dtype="auto",
                dsa_prefill_backend=None,
                dsa_decode_backend=None,
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch.object(overrides_module, "is_npu", return_value=False),
            patch.object(overrides_module, "is_xpu", return_value=False),
        ):
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                # Hopper: auto -> bfloat16
                self.assertEqual(
                    _dsa_kv_cache_dtype_default(_view()),
                    {"kv_cache_dtype": "bfloat16"},
                )
                # alias normalization
                self.assertEqual(
                    _dsa_kv_cache_dtype_default(_view(kv_cache_dtype="bf16")),
                    {"kv_cache_dtype": "bfloat16"},
                )
                # explicit value survives (no declaration)
                self.assertEqual(
                    _dsa_kv_cache_dtype_default(_view(kv_cache_dtype="fp8_e4m3")), {}
                )
                # unsupported dtype rejected
                with self.assertRaises(AssertionError):
                    _dsa_kv_cache_dtype_default(_view(kv_cache_dtype="fp8_e5m2"))
            with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
                # Blackwell: auto -> fp8
                self.assertEqual(
                    _dsa_kv_cache_dtype_default(_view()),
                    {"kv_cache_dtype": "fp8_e4m3"},
                )

    def test_deepseek_v4_kv_cache_dtype_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deepseek_v4_kv_cache_dtype,
        )

        def _view(arch="DeepseekV4ForCausalLM", **kw):
            hf = SimpleNamespace(architectures=[arch])
            defaults = dict(kv_cache_dtype="auto", device="cuda")
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        self.assertEqual(
            _deepseek_v4_kv_cache_dtype(_view()), {"kv_cache_dtype": "fp8_e4m3"}
        )
        # NPU pins bfloat16 regardless of the auto default
        self.assertEqual(
            _deepseek_v4_kv_cache_dtype(_view(device="npu")),
            {"kv_cache_dtype": "bfloat16"},
        )
        # explicit supported value survives
        self.assertEqual(
            _deepseek_v4_kv_cache_dtype(_view(kv_cache_dtype="bfloat16")), {}
        )
        with self.assertRaises(AssertionError):
            _deepseek_v4_kv_cache_dtype(_view(kv_cache_dtype="fp8_e5m2"))
        self.assertEqual(
            _deepseek_v4_kv_cache_dtype(_view(arch="LlamaForCausalLM")), {}
        )

    def test_deepseek_spec_moe_resolution_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deepseek_spec_moe_resolution,
        )
        from sglang.srt.environ import envs

        def _view(**kw):
            hf = SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
            defaults = dict(
                quantization="modelopt_fp4",
                speculative_algorithm="EAGLE",
                speculative_moe_runner_backend=None,
                speculative_moe_a2a_backend=None,
                ep_size=8,
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        with patch.object(overrides_module, "is_hip", return_value=True):
            with patch.object(
                envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE, "get", return_value=False
            ):
                self.assertEqual(
                    _deepseek_spec_moe_resolution(_view()),
                    {
                        "speculative_moe_runner_backend": "triton",
                        "speculative_moe_a2a_backend": "none",
                    },
                )
                # guards: quantization / algorithm / both fields user-set
                self.assertEqual(
                    _deepseek_spec_moe_resolution(_view(quantization="fp8")), {}
                )
                self.assertEqual(
                    _deepseek_spec_moe_resolution(_view(speculative_algorithm=None)),
                    {},
                )
                self.assertEqual(
                    _deepseek_spec_moe_resolution(
                        _view(
                            speculative_moe_runner_backend="triton",
                            speculative_moe_a2a_backend="none",
                        )
                    ),
                    {},
                )
            with patch.object(
                envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE, "get", return_value=True
            ):
                self.assertEqual(
                    _deepseek_spec_moe_resolution(_view()),
                    {
                        "speculative_moe_runner_backend": "deep_gemm",
                        "speculative_moe_a2a_backend": "deepep",
                    },
                )
                with self.assertRaises(ValueError):
                    _deepseek_spec_moe_resolution(_view(ep_size=1))
        # the arm is HIP-only
        with patch.object(overrides_module, "is_hip", return_value=False):
            self.assertEqual(_deepseek_spec_moe_resolution(_view()), {})

    def test_mamba_radix_cache_resolution_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _mamba_radix_cache_resolution,
            supports_mamba_cache_extra_buffer,
        )

        def _view(arch, layer_types=None, **kw):
            hf = SimpleNamespace(architectures=[arch])
            if layer_types is not None:
                hf.layer_types = layer_types
            defaults = dict(
                disable_radix_cache=False,
                mamba_radix_cache_strategy="auto",
                disable_overlap_schedule=False,
                page_size=None,
                linear_attn_backend="triton",
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        # arch guard: non-mamba arch declares nothing
        self.assertEqual(_mamba_radix_cache_resolution(_view("LlamaForCausalLM")), {})
        # radix cache disabled: nothing to resolve
        self.assertEqual(
            _mamba_radix_cache_resolution(
                _view("Qwen3NextForCausalLM", disable_radix_cache=True)
            ),
            {},
        )
        # auto + overlap wanted + extra-buffer support -> extra_buffer
        self.assertEqual(
            _mamba_radix_cache_resolution(_view("Qwen3NextForCausalLM")),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )
        # auto + no extra-buffer support (Lfm2) -> no_buffer + overlap disable
        self.assertEqual(
            _mamba_radix_cache_resolution(_view("Lfm2ForCausalLM")),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "no_buffer",
                "disable_overlap_schedule": True,
            },
        )
        # neither overlap nor paging wanted -> no_buffer even when supported
        declared = _mamba_radix_cache_resolution(
            _view("Qwen3NextForCausalLM", disable_overlap_schedule=True, page_size=1)
        )
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertIs(declared["disable_overlap_schedule"], True)
        # paging alone wants the extra buffer
        self.assertEqual(
            _mamba_radix_cache_resolution(
                _view(
                    "Qwen3NextForCausalLM", disable_overlap_schedule=True, page_size=64
                )
            )["mamba_radix_cache_strategy"],
            "extra_buffer",
        )
        # user-set strategy: only the routing marker is declared
        self.assertEqual(
            _mamba_radix_cache_resolution(
                _view(
                    "Qwen3NextForCausalLM",
                    mamba_radix_cache_strategy="extra_buffer_lazy",
                )
            ),
            {"uses_mamba_radix_cache": True},
        )
        # NemotronH routes through the pass (covered by the guard union,
        # not the branch chain — its hook invokes the handler)
        self.assertEqual(
            _mamba_radix_cache_resolution(_view("NemotronHForCausalLM")),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )
        # GraniteMoeHybrid is guarded on mamba layer types
        self.assertEqual(
            _mamba_radix_cache_resolution(
                _view("GraniteMoeHybridForCausalLM", layer_types=["attention"])
            ),
            {},
        )
        self.assertEqual(
            _mamba_radix_cache_resolution(
                _view("GraniteMoeHybridForCausalLM", layer_types=["mamba", "attention"])
            )["mamba_radix_cache_strategy"],
            "extra_buffer",
        )
        # extra-buffer support requires the triton linear-attn backend
        self.assertFalse(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(linear_attn_backend="fla"), "Qwen3NextForCausalLM"
            )
        )

    def test_qwen3_5_hybrid_coupled_declaration(self):
        from sglang.srt.arg_groups.overrides import _qwen3_5_hybrid_overrides

        def _args(default_backend, **kw):
            defaults = dict(
                attention_backend=None,
                _get_default_attn_backend=lambda **_: default_backend,
                use_mla_backend=lambda: False,
                get_model_config=lambda: None,
                mamba_radix_cache_strategy="auto",
                disable_radix_cache=False,
                speculative_algorithm=None,
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            # radix on + no extra buffer + no spec -> page_size=1 path
            self.assertEqual(
                _qwen3_5_hybrid_overrides(_args("trtllm_mha"), None),
                {"attention_backend": "triton", "page_size": 1},
            )
            # spec decoding present -> trtllm_mha + page 64 (coupled)
            self.assertEqual(
                _qwen3_5_hybrid_overrides(
                    _args("trtllm_mha", speculative_algorithm="EAGLE"), None
                ),
                {"attention_backend": "trtllm_mha", "page_size": 64},
            )
            # user-set backend: nothing declared
            self.assertEqual(
                _qwen3_5_hybrid_overrides(
                    _args("trtllm_mha", attention_backend="fa3"), None
                ),
                {},
            )
            # the mamba pass ran before this dispatch and stashed the
            # extra-buffer strategy: the callable must see it through the
            # view (SM100 hybrid keeps trtllm_mha + page 64)
            self.assertEqual(
                _qwen3_5_hybrid_overrides(
                    _args(
                        "trtllm_mha",
                        _resolved_overrides=[
                            (
                                "_mamba_radix_cache_declarations",
                                {"mamba_radix_cache_strategy": "extra_buffer"},
                            )
                        ],
                    ),
                    None,
                ),
                {"attention_backend": "trtllm_mha", "page_size": 64},
            )
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(_qwen3_5_hybrid_overrides(_args("fa3"), None), {})

    def test_qwen3vl_page_size(self):
        from sglang.srt.arg_groups.overrides import _qwen3vl_overrides

        with patch.object(overrides_module, "is_hip", return_value=True):
            with patch("sglang.srt.environ.envs.SGLANG_USE_AITER_UNIFIED_ATTN") as e:
                e.get.return_value = True
                self.assertEqual(
                    _qwen3vl_overrides(SimpleNamespace(page_size=None), None),
                    {"page_size": 16},
                )
                self.assertEqual(
                    _qwen3vl_overrides(SimpleNamespace(page_size=64), None), {}
                )

    def test_moe_runner_quant_constraint_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _moe_runner_backend_quant_constraints,
        )

        def _view(**kw):
            defaults = dict(quantization=None, moe_runner_backend="auto")
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            self.assertEqual(
                _moe_runner_backend_quant_constraints(
                    _view(quantization="nvfp4_online")
                ),
                {"moe_runner_backend": "flashinfer_trtllm"},
            )
            with self.assertRaises(ValueError):  # incompatible explicit backend
                _moe_runner_backend_quant_constraints(
                    _view(quantization="nvfp4_online", moe_runner_backend="triton")
                )
        self.assertEqual(
            _moe_runner_backend_quant_constraints(_view(quantization="mxfp8")),
            {"moe_runner_backend": "flashinfer_trtllm"},
        )
        with patch.object(overrides_module, "is_sm120_supported", return_value=True):
            self.assertEqual(
                _moe_runner_backend_quant_constraints(
                    _view(quantization="modelopt_fp4")
                ),
                {"moe_runner_backend": "flashinfer_cutlass"},
            )
        self.assertEqual(_moe_runner_backend_quant_constraints(_view()), {})

    def test_cutlass_moe_env_override_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _cutlass_moe_env_override,
        )

        with patch("sglang.srt.environ.envs.SGLANG_CUTLASS_MOE") as e:
            e.get.return_value = True
            self.assertEqual(
                _cutlass_moe_env_override(
                    ResolvedView(SimpleNamespace(quantization="fp8"))
                ),
                {"moe_runner_backend": "cutlass"},
            )
            with self.assertRaises(AssertionError):
                _cutlass_moe_env_override(
                    ResolvedView(SimpleNamespace(quantization=None))
                )
            e.get.return_value = False
            self.assertEqual(
                _cutlass_moe_env_override(ResolvedView(SimpleNamespace())), {}
            )

    def test_gguf_quantization_pass(self):
        from sglang.srt.arg_groups.overrides import ResolvedView, _gguf_quantization

        with patch(
            "sglang.srt.utils.hf_transformers_utils.check_gguf_file",
            return_value=True,
        ):
            self.assertEqual(
                _gguf_quantization(
                    ResolvedView(
                        SimpleNamespace(load_format="auto", model_path="x.gguf")
                    )
                ),
                {"quantization": "gguf"},
            )
            self.assertEqual(
                _gguf_quantization(
                    ResolvedView(
                        SimpleNamespace(load_format="safetensors", model_path="x")
                    )
                ),
                {},
            )

    def test_page_constraint_passes_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _fa4_page_constraint,
            _intel_xpu_page_constraint,
            _mla_backend_page_constraints,
        )

        def _view(**kw):
            defaults = dict(
                attention_backend=None,
                decode_attention_backend=None,
                prefill_attention_backend=None,
                page_size=1,
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        # flashmla snaps to 64 (unconditional within the backend match)
        self.assertEqual(
            _mla_backend_page_constraints(_view(attention_backend="flashmla")),
            {"page_size": 64},
        )
        # trtllm_mla with already-valid page: no declaration
        self.assertEqual(
            _mla_backend_page_constraints(
                _view(attention_backend="trtllm_mla", page_size=32)
            ),
            {},
        )
        # chained: flashmla via decode -> 64, then trtllm_mha accepts 64
        self.assertEqual(
            _mla_backend_page_constraints(
                _view(
                    decode_attention_backend="flashmla",
                    prefill_attention_backend="trtllm_mha",
                )
            ),
            {"page_size": 64},
        )
        # no matching backend: nothing declared
        self.assertEqual(_mla_backend_page_constraints(_view()), {})

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            self.assertEqual(
                _fa4_page_constraint(
                    _view(
                        attention_backend="fa4",
                        use_mla_backend=lambda: False,
                        speculative_eagle_topk=None,
                    )
                ),
                {"page_size": 128},
            )
            self.assertEqual(
                _fa4_page_constraint(
                    _view(
                        attention_backend="fa4",
                        use_mla_backend=lambda: False,
                        speculative_eagle_topk=2,  # EAGLE topk>1 keeps default
                    )
                ),
                {},
            )

        self.assertEqual(
            _intel_xpu_page_constraint(
                _view(
                    decode_attention_backend="intel_xpu",
                    use_mla_backend=lambda: False,
                )
            ),
            {"page_size": 128},
        )
        self.assertEqual(
            _intel_xpu_page_constraint(
                _view(
                    decode_attention_backend="intel_xpu",
                    use_mla_backend=lambda: True,
                    page_size=16,  # MLA decode accepts 16
                )
            ),
            {},
        )

    def test_monolith_attention_families_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import (
            _falcon_h1_jet_overrides,
            _gemma4_overrides,
            _glm4_moe_overrides,
            _granite_moe_hybrid_overrides,
            _lfm2_overrides,
            _llama4_overrides,
            _minicpm_v4_6_overrides,
        )

        def _args(**kw):
            defaults = dict(
                device="cuda",
                attention_backend=None,
                is_attention_backend_not_set=lambda: True,
                # keep the (now-absorbed) quant/moe blocks inert so these
                # assertions stay attention-only
                moe_runner_backend="triton",
                quantization=None,
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            self.assertEqual(
                _llama4_overrides(_args(), None), {"attention_backend": "trtllm_mha"}
            )
            self.assertEqual(_llama4_overrides(_args(device="cpu"), None), {})
            self.assertEqual(
                _llama4_overrides(_args(attention_backend="fa3"), None), {}
            )
            self.assertEqual(
                _gemma4_overrides(_args(), None), {"attention_backend": "trtllm_mha"}
            )
            self.assertEqual(
                _minicpm_v4_6_overrides(_args(), None),
                {"attention_backend": "triton"},
            )
            self.assertEqual(
                _falcon_h1_jet_overrides(_args(), None),
                {"attention_backend": "triton"},
            )
            self.assertEqual(
                _granite_moe_hybrid_overrides(
                    _args(), SimpleNamespace(layer_types=["mamba", "attention"])
                ),
                {"attention_backend": "flashinfer"},
            )
            self.assertEqual(
                _granite_moe_hybrid_overrides(
                    _args(), SimpleNamespace(layer_types=["attention"])
                ),
                {},
            )
            self.assertEqual(
                _lfm2_overrides(_args(), None), {"attention_backend": "flashinfer"}
            )
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(_minicpm_v4_6_overrides(_args(), None), {})
            with patch.object(overrides_module, "is_sm90_supported", return_value=True):
                self.assertEqual(
                    _llama4_overrides(_args(), None), {"attention_backend": "fa3"}
                )
            self.assertEqual(
                _gemma4_overrides(_args(), None), {"attention_backend": "triton"}
            )
        # Glm4Moe: unconditional tf32 declaration + (sm100) quant/moe absorption
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(
                _glm4_moe_overrides(None, None), {"enable_tf32_matmul": True}
            )
        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            self.assertEqual(
                _glm4_moe_overrides(
                    SimpleNamespace(
                        quantization=None,
                        _quantization_explicitly_unset=False,
                        moe_a2a_backend="none",
                        moe_runner_backend="auto",
                    ),
                    SimpleNamespace(
                        quantization_config={"quant_method": "modelopt_fp4"}
                    ),
                ),
                {
                    "quantization": "modelopt_fp4",
                    "moe_runner_backend": "flashinfer_trtllm",
                    "enable_tf32_matmul": True,
                },
            )

    def test_deepseek_moe_quant_slot_pass(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _deepseek_moe_quant_resolution,
        )

        def _view(arch="DeepseekV32ForCausalLM", quant_cfg=None, **kw):
            defaults = dict(
                quantization=None,
                _quantization_explicitly_unset=False,
                moe_a2a_backend="none",
                moe_runner_backend="auto",
                get_model_config=lambda: SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=[arch], quantization_config=quant_cfg
                    )
                ),
            )
            defaults.update(kw)
            return ResolvedView(SimpleNamespace(**defaults))

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            with patch.object(
                overrides_module, "get_quantization_config", return_value="fp8"
            ):
                # config-declared quant: detected + moe runner
                self.assertEqual(
                    _deepseek_moe_quant_resolution(_view()),
                    {
                        "quantization": "fp8",
                        "moe_runner_backend": "flashinfer_trtllm",
                    },
                )
            # non-deepseek arch guard (end-state list execution safety)
            self.assertEqual(
                _deepseek_moe_quant_resolution(_view(arch="LlamaForCausalLM")), {}
            )
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(_deepseek_moe_quant_resolution(_view()), {})

    def test_data_parallelism_and_a2a_passes(self):
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _a2a_backend_overrides,
            _a2a_ep_size,
            _data_parallelism_defaults,
        )

        self.assertEqual(
            _data_parallelism_defaults(
                ResolvedView(SimpleNamespace(dp_size=1, ep_join_mode=None))
            ),
            {"enable_dp_attention": False, "enable_dp_lm_head": False},
        )
        self.assertEqual(
            _data_parallelism_defaults(
                ResolvedView(SimpleNamespace(dp_size=2, ep_join_mode=None))
            ),
            {},
        )

        with patch("sglang.srt.environ.envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE") as e:
            e.get.return_value = False
            self.assertEqual(
                _a2a_backend_overrides(
                    ResolvedView(
                        SimpleNamespace(enable_waterfill=True, moe_a2a_backend="none")
                    )
                ),
                {"moe_a2a_backend": "deepep"},
            )
            e.get.return_value = True
            # megamoe env wins over the waterfill override (chained, last write)
            self.assertEqual(
                _a2a_backend_overrides(
                    ResolvedView(
                        SimpleNamespace(enable_waterfill=True, moe_a2a_backend="none")
                    )
                ),
                {"moe_a2a_backend": "megamoe"},
            )

        self.assertEqual(
            _a2a_ep_size(
                ResolvedView(SimpleNamespace(moe_a2a_backend="deepep", tp_size=8))
            ),
            {"ep_size": 8},
        )
        self.assertEqual(
            _a2a_ep_size(
                ResolvedView(SimpleNamespace(moe_a2a_backend="none", tp_size=8))
            ),
            {},
        )

    def test_deepseek_family_order_safe_declarations(self):
        from sglang.srt.arg_groups.overrides import _deepseek_family_overrides

        def _args(**kw):
            defaults = dict(
                is_attention_backend_not_set=lambda: True,
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                enable_prefill_cp=False,
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        # DSA path on CUDA: dsa fill + page 64
        with patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True
        ):
            with patch.object(overrides_module, "is_npu", return_value=False):
                with patch.object(overrides_module, "is_xpu", return_value=False):
                    with patch.object(overrides_module, "is_hip", return_value=False):
                        self.assertEqual(
                            _deepseek_family_overrides(_args(), None),
                            {"attention_backend": "dsa", "page_size": 64},
                        )
                    # HIP without the preshuffle path: page 1
                    with patch.object(overrides_module, "is_hip", return_value=True):
                        with patch(
                            "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
                            return_value=False,
                        ):
                            self.assertEqual(
                                _deepseek_family_overrides(_args(), None),
                                {"attention_backend": "dsa", "page_size": 1},
                            )
        # DSA CP (zigzag): the coupled parallel-field declaration
        with patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True
        ):
            with patch.object(overrides_module, "is_npu", return_value=False):
                with patch.object(overrides_module, "is_xpu", return_value=False):
                    with patch.object(overrides_module, "is_hip", return_value=False):
                        result = _deepseek_family_overrides(
                            _args(
                                enable_prefill_cp=True,
                                cp_strategy="zigzag",
                                tp_size=8,
                                dp_size=1,
                                ep_size=1,
                                moe_a2a_backend="none",
                                kv_cache_dtype="auto",
                            ),
                            None,
                        )
                        self.assertEqual(
                            result,
                            {
                                "attention_backend": "dsa",
                                "page_size": 64,
                                "enable_dp_attention": True,
                                "moe_dense_tp_size": 1,
                                "moe_a2a_backend": "deepep",
                                "ep_size": 8,
                                "attn_cp_size": 8,
                            },
                        )
                        # interleave CP with dp>1 must assert
                        with self.assertRaises(AssertionError):
                            _deepseek_family_overrides(
                                _args(
                                    enable_prefill_cp=True,
                                    cp_strategy="interleave",
                                    tp_size=8,
                                    dp_size=2,
                                ),
                                None,
                            )

        # MLA path on sm100: trtllm_mla fill (all three backends unset)
        with patch(
            "sglang.srt.configs.model_config.is_deepseek_dsa", return_value=False
        ):
            with patch.object(
                overrides_module, "is_sm100_supported", return_value=True
            ):
                self.assertEqual(
                    _deepseek_family_overrides(_args(), None),
                    {"attention_backend": "trtllm_mla"},
                )
                self.assertEqual(
                    _deepseek_family_overrides(
                        _args(decode_attention_backend="fa3"), None
                    ),
                    {},
                )
            with patch.object(
                overrides_module, "is_sm100_supported", return_value=False
            ):
                self.assertEqual(_deepseek_family_overrides(_args(), None), {})

    def test_qwen3_moe_family_quant_absorption(self):
        from sglang.srt.arg_groups.overrides import _qwen3_moe_family_overrides

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            with patch.object(
                overrides_module, "get_quantization_config", return_value="fp8"
            ):
                self.assertEqual(
                    _qwen3_moe_family_overrides(
                        SimpleNamespace(
                            quantization=None,
                            _quantization_explicitly_unset=False,
                            moe_a2a_backend="none",
                            moe_runner_backend="auto",
                        ),
                        SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
                    ),
                    {
                        "quantization": "fp8",
                        "moe_runner_backend": "flashinfer_trtllm",
                    },
                )
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(_qwen3_moe_family_overrides(None, None), {})

    def test_step3p_declarations_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import _step3p_overrides

        def _args(**kw):
            defaults = dict(
                speculative_algorithm=None,
                enable_hierarchical_cache=False,
                is_attention_backend_not_set=lambda: False,
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        self.assertEqual(
            _step3p_overrides(_args(speculative_algorithm="EAGLE"), None),
            {"enable_multi_layer_eagle": True},
        )
        self.assertEqual(
            _step3p_overrides(_args(enable_hierarchical_cache=True), None),
            {"swa_full_tokens_ratio": 1.0, "disable_hybrid_swa_memory": True},
        )
        self.assertEqual(_step3p_overrides(_args(), None), {})


class TestDeclarationValidation(CustomTestCase):
    def test_declarations_never_mutate_server_args(self):
        args = _FakeArgs()
        declarations = [("src", {"resolved_by_model": "dsv4", "also_resolved": 7})]
        validate_declarations(args, declarations)
        # validation is a pure whitelist check: the fields stay untouched
        self.assertEqual(args.resolved_by_model, _FakeArgs.resolved_by_model)
        self.assertEqual(args.also_resolved, _FakeArgs.also_resolved)

    def test_validation_rejects_unknown_fields(self):
        args = _FakeArgs()
        with self.assertRaises(ValueError):
            validate_declarations(args, [("src", {"nope": 1})])


if __name__ == "__main__":
    unittest.main()
