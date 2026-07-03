"""Unit tests for the model-override machinery: whitelist metadata, registry,
gate, publish wiring, and the per-arch golden diffs for migrated families."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

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
from sglang.srt.arg_groups.arg_utils import A, Arg, model_overridable_fields
from sglang.srt.arg_groups.overrides import (
    OverrideRecord,
    apply_declarations_to_server_args,
    apply_model_overrides,
    assert_flag_parity,
    collect_model_override_declarations,
    register_model_override,
)
from sglang.srt.runtime_context import (
    _StaticFlags,
    get_context,
    get_server_args,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase


@dataclasses.dataclass
class _FakeArgs:
    plain: A[int, "help text only"] = 0
    resolved_by_model: A[str, Arg(help="x", model_overridable=True)] = "auto"
    also_resolved: A[Optional[int], Arg(help="y", model_overridable=True)] = None
    metadata_but_not_overridable: A[bool, Arg(help="z")] = False


class TestModelOverridableWhitelist(CustomTestCase):
    def test_arg_defaults_to_not_overridable(self):
        self.assertFalse(Arg().model_overridable)

    def test_whitelist_derivation_from_annotated_metadata(self):
        self.assertEqual(
            model_overridable_fields(_FakeArgs),
            frozenset({"resolved_by_model", "also_resolved"}),
        )

    def test_server_args_whitelist_is_exactly_the_migrated_fields(self):
        # Fields are whitelisted one family at a time by the migration
        # sweeps. This pin makes accidental tagging visible — extend it in
        # the same commit that tags a new field.
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            model_overridable_fields(ServerArgs),
            frozenset(
                {
                    "dtype",
                    "enable_tf32_matmul",
                    "enable_multi_layer_eagle",
                    "swa_full_tokens_ratio",
                    "disable_hybrid_swa_memory",
                    "sampling_backend",
                    "attention_backend",
                }
            ),
        )

    def test_non_dataclass_yields_empty_whitelist(self):
        self.assertEqual(model_overridable_fields(SimpleNamespace), frozenset())


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

    def test_run_pass_appends_stash_and_dual_applies(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        live = SimpleNamespace(x=None, _resolved_overrides=[])

        def _fill_x(view):
            return {"x": "filled"} if view.x is None else {}

        run_post_process_pass(live, _fill_x)
        self.assertEqual(live.x, "filled")  # dual-applied in place
        self.assertEqual(
            live._resolved_overrides, [(_fill_x.__qualname__, {"x": "filled"})]
        )
        run_post_process_pass(live, _fill_x)  # now a no-op
        self.assertEqual(len(live._resolved_overrides), 1)

    def test_run_pass_rejects_non_dict(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        with self.assertRaises(TypeError):
            run_post_process_pass(
                SimpleNamespace(_resolved_overrides=[]), lambda view: None
            )


@dataclasses.dataclass
class _FakeAttnGroup(_StaticFlags):
    backend: str = "unset"


@dataclasses.dataclass
class _FakeFlags(_StaticFlags):
    attn: _FakeAttnGroup = dataclasses.field(default_factory=_FakeAttnGroup)
    resolved_by_model: str = "unset"
    also_resolved: Optional[int] = None


class TestApplyModelOverridesGate(CustomTestCase):
    def _fresh(self):
        return _FakeFlags(), _FakeArgs()

    def test_materializes_declared_and_pristine_leaves(self):
        flags, args = self._fresh()
        records = apply_model_overrides(
            flags, args, [("src", {"resolved_by_model": "dsv4"})]
        )
        self.assertEqual(flags.resolved_by_model, "dsv4")  # declared
        self.assertIsNone(flags.also_resolved)  # undeclared -> pristine value
        self.assertEqual(args.resolved_by_model, "auto")  # server_args untouched
        self.assertEqual(
            records, [OverrideRecord("src", "resolved_by_model", "auto", "dsv4")]
        )

    def test_last_writer_wins_then_terminal_wins_last(self):
        flags, args = self._fresh()
        records = apply_model_overrides(
            flags,
            args,
            [
                ("first", {"resolved_by_model": "a"}),
                ("second", {"resolved_by_model": "b"}),
            ],
            terminal=[("enforce_disable", {"resolved_by_model": "off"})],
        )
        self.assertEqual(flags.resolved_by_model, "off")
        self.assertEqual([r.resolved for r in records], ["a", "b", "off"])
        self.assertEqual(records[1].base, "a")  # provenance chains the writers

    def test_non_whitelisted_field_rejected_before_any_write(self):
        flags, args = self._fresh()
        with self.assertRaises(ValueError):
            apply_model_overrides(
                flags,
                args,
                [("ok", {"resolved_by_model": "x"}), ("bad", {"plain": 1})],
            )
        self.assertEqual(flags.resolved_by_model, "unset")  # transactional

    def test_missing_leaf_rejected_before_any_write(self):
        flags, args = self._fresh()
        with self.assertRaises(ValueError):
            apply_model_overrides(
                flags,
                args,
                [("src", {"resolved_by_model": "x"})],
                whitelist={"resolved_by_model", "field_without_leaf"},
            )
        self.assertEqual(flags.resolved_by_model, "unset")

    def test_frozen_flags_rejected(self):
        flags, args = self._fresh()
        flags.freeze()
        with self.assertRaises(RuntimeError):
            apply_model_overrides(flags, args, [("src", {"resolved_by_model": "x"})])

    def test_leaf_map_routes_to_group_leaf(self):
        flags, args = self._fresh()
        apply_model_overrides(
            flags,
            args,
            [("src", {"resolved_by_model": "fa3"})],
            whitelist={"resolved_by_model"},
            leaf_map={"resolved_by_model": "attn.backend"},
        )
        self.assertEqual(flags.attn.backend, "fa3")
        self.assertEqual(flags.resolved_by_model, "unset")  # flat leaf untouched


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


@dataclasses.dataclass
class _NoOverridableArgs:
    x: int = 1


class TestPublishResolvesFlags(_IsolatedPublish):
    """Publish wiring: stash-carrying publishes resolve into flags via the
    gate; publishes without the stash skip resolution."""

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

    def test_empty_stash_publish_runs_gate_as_noop(self):
        sa = _NoOverridableArgs()
        sa._resolved_overrides = []
        get_context().set_server_args(sa)
        self.assertIs(get_server_args(), sa)

    def test_non_whitelisted_declaration_fails_at_publish(self):
        from sglang.srt.runtime_context import get_flags

        flags_before = get_flags()
        sa = _NoOverridableArgs()
        sa._resolved_overrides = [("rogue", {"x": 2})]
        with self.assertRaises(ValueError):
            get_context().set_server_args(sa)
        # a failed publish must leave BOTH the slot and the flags untouched
        self.assertIs(get_flags(), flags_before)


class TestGoldenModelOverrides(_IsolatedPublish):
    """Per-arch golden diff for migrated families: the declarative path must
    reproduce the legacy imperative writes byte-identically on server_args
    (dual-apply) and materialize the same values on the flags tier at
    publish."""

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
        from sglang.srt.runtime_context import get_flags
        from sglang.srt.server_args import set_global_server_args_for_scheduler

        set_global_server_args_for_scheduler(server_args)
        return get_flags()

    def test_mistral_large3_forces_bfloat16(self):
        sa = self._construct("MistralLarge3ForCausalLM", "mistral")
        self.assertEqual(sa.dtype, "bfloat16")  # dual-apply == legacy write
        self.assertIn(
            ("MODEL_OVERRIDES['MistralLarge3ForCausalLM']", {"dtype": "bfloat16"}),
            sa._resolved_overrides,
        )
        self.assertEqual(self._publish(sa).dtype, "bfloat16")

    def test_pixtral_forces_bfloat16(self):
        sa = self._construct("PixtralForConditionalGeneration", "pixtral")
        self.assertEqual(sa.dtype, "bfloat16")
        self.assertEqual(self._publish(sa).dtype, "bfloat16")

    def test_user_requested_dtype_is_still_overridden(self):
        # Legacy fidelity: the arch branch overwrote dtype unconditionally,
        # so the declaration must too. The pristine request survives only on
        # provenance (and, post-V3, as the un-overridden server_args field).
        sa = self._construct("MistralLarge3ForCausalLM", "mistral", dtype="float16")
        self.assertEqual(sa.dtype, "bfloat16")
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
        self.assertTrue(sa.enable_tf32_matmul)  # dual-apply == legacy write
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
        # dual-apply == legacy writes
        self.assertEqual(sa.swa_full_tokens_ratio, 1.0)
        self.assertTrue(sa.disable_hybrid_swa_memory)
        flags = self._publish(sa)
        self.assertEqual(flags.swa_full_tokens_ratio, 1.0)
        self.assertTrue(flags.disable_hybrid_swa_memory)

    def test_gemma2_disables_hybrid_swa_memory(self):
        sa = self._construct("Gemma2ForCausalLM", "llama")
        self.assertTrue(sa.disable_hybrid_swa_memory)  # dual-apply == legacy
        self.assertIn(
            ("_gemma2_gemma3_overrides", {"disable_hybrid_swa_memory": True}),
            sa._resolved_overrides,
        )
        self.assertTrue(self._publish(sa).disable_hybrid_swa_memory)

    def test_olmo2_disables_hybrid_swa_memory(self):
        sa = self._construct("Olmo2ForCausalLM", "llama")
        self.assertTrue(sa.disable_hybrid_swa_memory)
        self.assertTrue(self._publish(sa).disable_hybrid_swa_memory)

    def test_exaone_conditional_on_sliding_window_pattern(self):
        # With the pattern the branch also asserts an explicit backend.
        sa = self._construct(
            "Exaone4ForCausalLM",
            "llama",
            config_extra={"sliding_window_pattern": "LLLG"},
            attention_backend="fa3",
        )
        self.assertTrue(sa.disable_hybrid_swa_memory)
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
        self.assertEqual(sa.dtype, "bfloat16")  # dual-apply == legacy
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
        self.assertEqual(sa.sampling_backend, expected)
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
        # last writer wins on the flags leaf and parity holds end-to-end.
        self.assertEqual(sa.sampling_backend, "pytorch")
        flags = self._publish(sa)
        self.assertEqual(flags.sampling_backend, "pytorch")
        # the deterministic attention fill declared a compatible backend and
        # the compatibility default-fill then had nothing to do
        self.assertIn(
            (
                "_deterministic_attention_backend",
                {"attention_backend": sa.attention_backend},
            ),
            sa._resolved_overrides,
        )
        self.assertEqual(flags.attn.backend, sa.attention_backend)

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
        sa = self._construct(
            "LlamaForCausalLM",
            "llama",
            dllm_algorithm="LowConfidence",
            disable_radix_cache=True,
        )
        self.assertEqual(sa.attention_backend, "flashinfer")
        self.assertIn(
            ("_dllm_attention_backend", {"attention_backend": "flashinfer"}),
            sa._resolved_overrides,
        )
        # first MAPPED leaf: attention_backend routes to flags.attn.backend
        self.assertEqual(self._publish(sa).attn.backend, "flashinfer")

    def test_attention_backend_leaf_materializes_end_state(self):
        # The default-fill pass declares the platform-selected backend; the
        # leaf must equal the final server_args value (publish parity).
        sa = self._construct("LlamaForCausalLM", "llama")
        declared = {f for _s, d in sa._resolved_overrides for f in d}
        self.assertIn("attention_backend", declared)  # default fill declared
        self.assertEqual(self._publish(sa).attn.backend, sa.attention_backend)

    def test_runner_side_adjustment_can_refresh_declaration(self):
        from sglang.srt.arg_groups.overrides import refresh_declared_fields

        sa = self._construct("LlamaForCausalLM", "llama")
        declared = {f for _s, d in sa._resolved_overrides for f in d}
        self.assertIn("attention_backend", declared)
        # Simulate a legacy runner-side overwrite between collection and publish
        # (model_specific_adjustment forces attention_backend for HRM-Text).
        sa.attention_backend = "fa3" if sa.attention_backend != "fa3" else "triton"
        with self.assertRaises(AssertionError):
            self._publish(sa)  # stale declaration breaks parity
        refresh_declared_fields(sa, ("attention_backend",))
        self.assertEqual(self._publish(sa).attn.backend, sa.attention_backend)

    def test_attention_backend_user_choice_declares_nothing_extra(self):
        sa = self._construct("LlamaForCausalLM", "llama", attention_backend="triton")
        self.assertEqual(sa.attention_backend, "triton")
        self.assertEqual(self._publish(sa).attn.backend, "triton")

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
        # Glm4Moe: unconditional tf32 declaration (quant/moe writes stay in
        # the branch until their field chains migrate)
        self.assertEqual(_glm4_moe_overrides(None, None), {"enable_tf32_matmul": True})

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


class TestDualApplyParity(CustomTestCase):
    def test_dual_apply_replays_and_parity_holds(self):
        flags, args = _FakeFlags(), _FakeArgs()
        declarations = [("src", {"resolved_by_model": "dsv4", "also_resolved": 7})]
        apply_model_overrides(flags, args, declarations)
        apply_declarations_to_server_args(args, declarations)
        self.assertEqual(args.resolved_by_model, "dsv4")
        self.assertEqual(args.also_resolved, 7)
        assert_flag_parity(flags, args, ["resolved_by_model", "also_resolved"])

    def test_parity_detects_drift(self):
        flags, args = _FakeFlags(), _FakeArgs()
        apply_model_overrides(flags, args, [("src", {"resolved_by_model": "x"})])
        # dual-apply skipped -> server_args still pristine -> drift is caught
        with self.assertRaises(AssertionError):
            assert_flag_parity(flags, args, ["resolved_by_model"])


if __name__ == "__main__":
    unittest.main()
