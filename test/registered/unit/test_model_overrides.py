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

from sglang.srt.arg_groups import attention_hook
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.arg_groups.arg_utils import A, Arg, resolvable_fields
from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    register_model_override,
    resolution_result,
    validate_declarations,
)
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
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
                    "enable_tp_lm_head_all_to_all",
                    "moe_a2a_backend",
                    "ep_size",
                    "moe_dense_tp_size",
                    "attn_cp_size",
                    "dcp_comm_backend",
                    "dcp_replicate_q_proj",
                    "disable_overlap_schedule",
                    "disable_radix_cache",
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
                    "fp4_gemm_runner_backend",
                    "disable_custom_all_reduce",
                    "enable_aiter_allreduce_fusion",
                    "enable_symm_mem",
                    "speculative_attention_mode",
                    "speculative_draft_attention_backend",
                }
            ),
        )


class TestDSparkCheckpointConfig(CustomTestCase):
    def test_sample_from_anchor_is_read_from_checkpoint_config(self):
        from sglang.srt.speculative.dspark_components.dspark_config import (
            get_dspark_sample_from_anchor,
        )

        config = SimpleNamespace(
            architectures=["UnrelatedDSparkModel"], sample_from_anchor=False
        )
        self.assertFalse(get_dspark_sample_from_anchor(config))
        self.assertTrue(get_dspark_sample_from_anchor(SimpleNamespace()))


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

    def test_dummy_fixture_publishes_the_object_it_resolved(self):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        sa = ServerArgs(model_path="dummy")  # construction resolves nothing
        self.assertFalse(hasattr(sa, "_resolved_overrides"))
        set_global_server_args_for_scheduler(sa)
        self.assertIs(get_server_args(), sa)
        # Publishing is what resolved it; the handlers ahead of the dummy
        # short-circuit still declare. What they decided is the projection --
        # the fields keep what the caller passed.
        from sglang.srt.arg_groups.overrides import resolution_result

        self.assertTrue(sa._resolved_overrides, "publishing declared nothing")
        for source, declared in sa._resolved_overrides:
            for field, value in declared.items():
                self.assertEqual(
                    resolution_result(sa, field), value, f"{source}: {field}"
                )


class TestGoldenModelOverrides(_IsolatedPublish):
    """Per-arch golden diff for migrated families: the declarative path must
    reproduce the legacy imperative writes byte-identically in the resolution
    result; the publish round-trip returns the same object.

    `_resolved` is how the assertions read it. A model-specific override only
    declares -- it does not write the field -- so the record keeps what the
    caller passed and the projection carries the override.
    """

    def _resolved(self, server_args, field):
        from sglang.srt.arg_groups.overrides import resolution_result

        return resolution_result(server_args, field)

    def _leaf(self, field):
        """The published value of `field`, whichever bag owns it.

        The publish round-trip is checked on the bags: the record the process
        publishes is the raw input, and the leaf is what every reader reads.
        """
        from sglang.srt.runtime_context import get_context

        return get_context().config_leaf(field)

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

    @staticmethod
    def _minicpm_overrides(
        architecture,
        *,
        sparse_attention=False,
        lightning_attention=False,
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        disaggregation_mode="null",
        enable_dp_attention=False,
        enable_hierarchical_cache=False,
    ):
        args = SimpleNamespace(
            attention_backend=attention_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            disaggregation_mode=disaggregation_mode,
            enable_dp_attention=enable_dp_attention,
            enable_hierarchical_cache=enable_hierarchical_cache,
        )
        mixer_types = []
        if sparse_attention:
            mixer_types.append("minicpm4")
        if lightning_attention:
            mixer_types.append("lightning-attn")
        if not mixer_types:
            mixer_types.append("minicpm4")
        declarations = collect_model_override_declarations(
            architecture,
            args,
            hf_config=MiniCPMHybridConfig(
                num_hidden_layers=len(mixer_types),
                num_attention_heads=1,
                num_key_value_heads=1,
                mixer_types=mixer_types,
                sparse_config={} if sparse_attention else None,
            ),
        )
        return {
            field: value
            for _, declaration in declarations
            for field, value in declaration.items()
        }

    def test_minicpm_disables_radix_cache_only_for_hybrid_layers(self):
        for architecture in ("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM"):
            with self.subTest(architecture=architecture):
                self.assertNotIn(
                    "disable_radix_cache",
                    self._minicpm_overrides(architecture),
                )
                self.assertTrue(
                    self._minicpm_overrides(architecture, sparse_attention=True)[
                        "disable_radix_cache"
                    ]
                )
                self.assertTrue(
                    self._minicpm_overrides(architecture, lightning_attention=True)[
                        "disable_radix_cache"
                    ]
                )

    def test_minicpm_rejects_dp_attention(self):
        for architecture in ("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(
                    ValueError,
                    "MiniCPM does not support DP attention",
                ):
                    self._minicpm_overrides(
                        architecture,
                        enable_dp_attention=True,
                    )

    def test_minicpm_rejects_hierarchical_cache_for_hybrid_models(self):
        for capability in ("sparse_attention", "lightning_attention"):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(
                    ValueError,
                    "MiniCPM SALA does not support hierarchical cache",
                ):
                    self._minicpm_overrides(
                        "MiniCPMSALAForCausalLM",
                        enable_hierarchical_cache=True,
                        **{capability: True},
                    )

    def test_sparse_minicpm_defaults_to_sparse_attention_backend(self):
        with patch.object(
            overrides_module,
            "is_blackwell_supported",
            return_value=False,
        ):
            for architecture in ("MiniCPMForCausalLM", "MiniCPMSALAForCausalLM"):
                with self.subTest(architecture=architecture):
                    self.assertEqual(
                        self._minicpm_overrides(
                            architecture,
                            sparse_attention=True,
                        )["attention_backend"],
                        "minicpm_flashattn",
                    )

    def test_minicpm_overrides_use_config_capabilities(self):
        args = SimpleNamespace(
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend=None,
            disaggregation_mode="null",
            enable_dp_attention=False,
            enable_hierarchical_cache=False,
        )
        config = SimpleNamespace(
            has_minicpm_sparse_attention=True,
            has_lightning_layers=False,
        )

        with patch.object(
            overrides_module, "is_blackwell_supported", return_value=False
        ):
            overrides = overrides_module._minicpm_sala_overrides(args, config)

        self.assertTrue(overrides["disable_radix_cache"])
        self.assertEqual(overrides["attention_backend"], "minicpm_flashattn")

    def test_sparse_minicpm_defaults_to_flashinfer_on_blackwell(self):
        with patch.object(
            overrides_module,
            "is_blackwell_supported",
            return_value=True,
        ):
            self.assertEqual(
                self._minicpm_overrides(
                    "MiniCPMSALAForCausalLM",
                    sparse_attention=True,
                )["attention_backend"],
                "minicpm_flashinfer",
            )

    def test_minicpm_preserves_explicit_attention_backend(self):
        overrides = self._minicpm_overrides(
            "MiniCPMSALAForCausalLM",
            sparse_attention=True,
            attention_backend="fa3",
        )
        self.assertNotIn("attention_backend", overrides)

    def test_sparse_minicpm_rejects_pd_disaggregation(self):
        for disaggregation_mode in ("prefill", "decode"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                with self.assertRaisesRegex(
                    ValueError,
                    "MiniCPM sparse attention does not support PD disaggregation",
                ):
                    self._minicpm_overrides(
                        "MiniCPMSALAForCausalLM",
                        sparse_attention=True,
                        disaggregation_mode=disaggregation_mode,
                    )
        for backend_field in (
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            with self.subTest(backend_field=backend_field):
                with self.assertRaisesRegex(
                    ValueError,
                    "MiniCPM sparse attention does not support PD disaggregation",
                ):
                    self._minicpm_overrides(
                        "MiniCPMSALAForCausalLM",
                        sparse_attention=True,
                        disaggregation_mode="decode",
                        **{backend_field: "minicpm_flashattn"},
                    )

    def test_minicpm_force_dense_uses_stock_attention_backend(self):
        with envs.SGLANG_MINICPM_FORCE_DENSE.override(True):
            self.assertNotIn(
                "attention_backend",
                self._minicpm_overrides(
                    "MiniCPMSALAForCausalLM",
                    sparse_attention=True,
                ),
            )
            self.assertEqual(
                self._minicpm_overrides(
                    "MiniCPMSALAForCausalLM",
                    sparse_attention=True,
                    attention_backend="minicpm_flashinfer",
                )["attention_backend"],
                "flashinfer",
            )
            with patch.object(
                overrides_module,
                "is_blackwell_supported",
                return_value=True,
            ):
                self.assertEqual(
                    self._minicpm_overrides(
                        "MiniCPMSALAForCausalLM",
                        sparse_attention=True,
                        attention_backend="minicpm_flashattn",
                    )["attention_backend"],
                    "fa4",
                )
            with patch.object(
                overrides_module,
                "is_blackwell_supported",
                return_value=False,
            ):
                split_overrides = self._minicpm_overrides(
                    "MiniCPMSALAForCausalLM",
                    sparse_attention=True,
                    prefill_attention_backend="minicpm_flashattn",
                    decode_attention_backend="minicpm_flashattn",
                )
                self.assertEqual(split_overrides["prefill_attention_backend"], "fa3")
                self.assertEqual(split_overrides["decode_attention_backend"], "fa3")

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
        server_args = ServerArgs(model_path=config_dir, **server_kwargs)
        server_args.resolve_once()
        return server_args

    def _publish(self, server_args):
        from sglang.srt.server_args import (
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(server_args)
        return get_server_args()

    def test_mistral_large3_forces_bfloat16(self):
        sa = self._construct("MistralLarge3ForCausalLM", "mistral")
        self.assertEqual(
            self._resolved(sa, "dtype"), "bfloat16"
        )  # materialized at end of resolution
        self.assertIn(
            ("MODEL_OVERRIDES['MistralLarge3ForCausalLM']", {"dtype": "bfloat16"}),
            sa._resolved_overrides,
        )
        self.assertEqual((self._publish(sa), self._leaf("dtype"))[1], "bfloat16")

    def test_user_requested_dtype_is_still_overridden(self):
        # Legacy fidelity: the arch branch overwrote dtype unconditionally,
        # so the declaration must too. The request survives on the record; the
        # projection carries the override.
        sa = self._construct("MistralLarge3ForCausalLM", "mistral", dtype="float16")
        self.assertEqual(self._resolved(sa, "dtype"), "bfloat16")
        self.assertEqual((self._publish(sa), self._leaf("dtype"))[1], "bfloat16")

    def test_control_arch_keeps_pristine_dtype(self):
        sa = self._construct("LlamaForCausalLM", "llama")
        self.assertEqual(self._resolved(sa, "dtype"), "auto")
        declared = {f for _s, d in sa._resolved_overrides for f in d}
        self.assertNotIn("dtype", declared)  # no arch declaration for Llama
        # publish still projects the whitelisted leaf with the pristine
        # value: readers only ever read flags.
        self.assertEqual((self._publish(sa), self._leaf("dtype"))[1], "auto")

    def test_minimax_m2_enables_tf32_matmul(self):
        sa = self._construct("MiniMaxM2ForCausalLM", "llama")
        self.assertTrue(self._resolved(sa, "enable_tf32_matmul"))
        self.assertIn(
            ("_minimax_m2_overrides", {"enable_tf32_matmul": True}),
            sa._resolved_overrides,
        )
        flags = self._publish(sa)
        self.assertTrue(self._leaf("enable_tf32_matmul"))
        self.assertFalse(self._leaf("enable_multi_layer_eagle"))  # the pristine value

    def test_minimax_m2_sm10x_nvfp4_uses_routed_trtllm(self):
        """MiniMax-M2 NVFP4 auto must avoid the unsupported plain TRT-LLM path."""
        # Every module that asks: the attention handler validates what the
        # override family picks, and each holds its own import.
        with patch.object(
            overrides_module, "is_sm100_supported", return_value=True
        ), patch.object(attention_hook, "is_sm100_supported", return_value=True):
            explicit = self._construct(
                "MiniMaxM2ForCausalLM",
                "llama",
                quantization="modelopt_fp4",
                moe_runner_backend="flashinfer_cutlass",
            )
            non_nvfp4 = self._construct(
                "MiniMaxM2ForCausalLM", "llama", quantization="fp8"
            )
            nvfp4 = self._construct(
                "MiniMaxM2ForCausalLM", "llama", quantization="modelopt_fp4"
            )

        self.assertEqual(
            self._resolved(explicit, "moe_runner_backend"), "flashinfer_cutlass"
        )
        self.assertEqual(self._resolved(non_nvfp4, "moe_runner_backend"), "auto")
        self.assertEqual(
            self._resolved(nvfp4, "moe_runner_backend"), "flashinfer_trtllm_routed"
        )
        self.assertTrue(self._resolved(nvfp4, "disable_shared_experts_fusion"))
        self.assertIn(
            (
                "_minimax_m2_overrides",
                {
                    "enable_tf32_matmul": True,
                    "moe_runner_backend": "flashinfer_trtllm_routed",
                },
            ),
            nvfp4._resolved_overrides,
        )
        self.assertIn(
            ("_moe_runner_fusion_disable", {"disable_shared_experts_fusion": True}),
            nvfp4._resolved_overrides,
        )

        # Thor (SM110) and other architectures keep the existing auto behavior.
        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
            patch.object(overrides_module, "is_sm120_supported", return_value=False),
        ):
            non_sm10x = self._construct(
                "MiniMaxM2ForCausalLM", "llama", quantization="modelopt_fp4"
            )
        self.assertEqual(self._resolved(non_sm10x, "moe_runner_backend"), "auto")

        self._publish(nvfp4)
        self.assertEqual(get_exec().moe.moe_runner_backend, "flashinfer_trtllm_routed")

    def test_mimo_v2_declarations(self):
        # Callable-level golden: MiMoV2 archs are hybrid (config-shape heavy),
        # so the declaration is pinned directly for both provider inputs.
        from sglang.srt.arg_groups.overrides import _mimo_v2_overrides

        def _args(**kw):
            defaults = dict(speculative_algorithm=None, moe_runner_backend="auto")
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        # Non-SM100: the MoE pin must not fire, so hf_config is never inspected.
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(
                _mimo_v2_overrides(_args(speculative_algorithm="EAGLE"), None),
                {"enable_multi_layer_eagle": True},
            )
            self.assertEqual(_mimo_v2_overrides(_args(), None), {})

    def test_mimo_v2_sm100_fp8_pins_flashinfer_trtllm_moe(self):
        """Blackwell FP8 must not be left on the triton fused-MoE runner."""
        from sglang.srt.arg_groups.overrides import _mimo_v2_overrides

        def _args(**kw):
            defaults = dict(speculative_algorithm=None, moe_runner_backend="auto")
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            with patch.object(
                overrides_module, "get_quantization_config", return_value="fp8"
            ):
                self.assertEqual(
                    _mimo_v2_overrides(_args(), None),
                    {"moe_runner_backend": "flashinfer_trtllm"},
                )
                # An explicit user choice is never overwritten.
                self.assertEqual(
                    _mimo_v2_overrides(_args(moe_runner_backend="triton"), None), {}
                )
            # FP4 checkpoints run through flashinfer_mxfp4, so they must not be
            # pinned to flashinfer_trtllm.
            with patch.object(
                overrides_module, "get_quantization_config", return_value="mxfp4"
            ):
                self.assertEqual(_mimo_v2_overrides(_args(), None), {})

    def test_mimo_v2_family_is_registered(self):
        with patch.object(overrides_module, "is_sm100_supported", return_value=False):
            self.assertEqual(
                collect_model_override_declarations(
                    "MiMoV2FlashForCausalLM",
                    SimpleNamespace(
                        speculative_algorithm="EAGLE", moe_runner_backend="auto"
                    ),
                    None,
                ),
                [("_mimo_v2_overrides", {"enable_multi_layer_eagle": True})],
            )

    def _nemotron_h_args(self, *, quantized_layers):
        hf_config = SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
            mlp_hidden_act="relu2",
            quantization_config={
                "quant_algo": "MIXED_PRECISION",
                "quant_method": "modelopt_mixed",
                "quantized_layers": quantized_layers,
            },
        )
        model_config = SimpleNamespace(
            quantization="modelopt_mixed", hf_config=hf_config
        )
        return (
            SimpleNamespace(
                quantization="modelopt_fp4",
                moe_runner_backend="auto",
                moe_a2a_backend="none",
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                speculative_algorithm=None,
                speculative_eagle_topk=None,
                speculative_draft_attention_backend=None,
                page_size=None,
                mamba_radix_cache_strategy="auto",
                _model_config=model_config,
            ),
            hf_config,
        )

    def test_nemotron_h_w4a16_moe_uses_marlin_on_sm100(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(
            quantized_layers={
                "backbone.layers.1.mixer.experts.0.up_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
                "backbone.layers.1.mixer.experts.0.down_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
                "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
            }
        )

        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
        ):
            self.assertEqual(
                _nemotron_h_overrides(server_args, hf_config),
                {
                    "quantization": "modelopt_mixed",
                    "moe_runner_backend": "marlin",
                    "attention_backend": "trtllm_mha",
                },
            )

    def test_nemotron_h_nvfp4_moe_keeps_flashinfer_trtllm_on_sm100(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(
            quantized_layers={
                "backbone.layers.1.mixer.experts.0.up_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "backbone.layers.1.mixer.experts.0.down_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
            }
        )

        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
        ):
            self.assertEqual(
                _nemotron_h_overrides(server_args, hf_config),
                {
                    "quantization": "modelopt_mixed",
                    "moe_runner_backend": "flashinfer_trtllm",
                    "attention_backend": "trtllm_mha",
                },
            )

    def test_nemotron_h_speculation_uses_arch_specific_attention_on_blackwell(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        cases = {
            True: {
                "attention_backend": "trtllm_mha",
                "page_size": 64,
                "mamba_radix_cache_strategy": "extra_buffer",
                "speculative_draft_attention_backend": "trtllm_mha",
            },
            False: {
                "attention_backend": "triton",
                "speculative_draft_attention_backend": "flashinfer",
            },
        }
        for is_sm100, expected in cases.items():
            with self.subTest(is_sm100=is_sm100):
                server_args, hf_config = self._nemotron_h_args(quantized_layers={})
                server_args.speculative_algorithm = "EAGLE"

                with (
                    patch.object(
                        overrides_module,
                        "is_blackwell_supported",
                        return_value=True,
                    ),
                    patch.object(
                        overrides_module,
                        "is_sm100_supported",
                        return_value=is_sm100,
                    ),
                ):
                    overrides = _nemotron_h_overrides(server_args, hf_config)
                    for key, value in expected.items():
                        self.assertEqual(overrides[key], value)

    def test_nemotron_h_sm100_speculative_draft_backend_matrix(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        for algorithm in ("EAGLE", "NEXTN", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                server_args, hf_config = self._nemotron_h_args(quantized_layers={})
                server_args.speculative_algorithm = algorithm
                with (
                    patch.object(
                        overrides_module, "is_blackwell_supported", return_value=True
                    ),
                    patch.object(
                        overrides_module, "is_sm100_supported", return_value=True
                    ),
                ):
                    overrides = _nemotron_h_overrides(server_args, hf_config)
                self.assertEqual(overrides["attention_backend"], "trtllm_mha")
                self.assertEqual(
                    overrides["speculative_draft_attention_backend"],
                    "trtllm_mha",
                )

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})
        server_args.speculative_algorithm = "DFLASH"
        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            overrides = _nemotron_h_overrides(server_args, hf_config)
        self.assertEqual(overrides["attention_backend"], "trtllm_mha")
        self.assertNotIn("speculative_draft_attention_backend", overrides)

    def test_nemotron_h_sm100_speculation_preserves_explicit_cache_and_draft(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})
        server_args.speculative_algorithm = "DSPARK"
        server_args.page_size = 128
        server_args.mamba_radix_cache_strategy = "extra_buffer_lazy"
        server_args.speculative_draft_attention_backend = "flashinfer"

        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            overrides = _nemotron_h_overrides(server_args, hf_config)

        self.assertEqual(overrides["attention_backend"], "trtllm_mha")
        self.assertNotIn("page_size", overrides)
        self.assertNotIn("mamba_radix_cache_strategy", overrides)
        self.assertNotIn("speculative_draft_attention_backend", overrides)

    def test_nemotron_h_sm100_topk_tree_falls_back_to_triton(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})
        server_args.speculative_algorithm = "EAGLE"
        server_args.speculative_eagle_topk = 4

        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            overrides = _nemotron_h_overrides(server_args, hf_config)

        self.assertEqual(overrides["attention_backend"], "triton")
        self.assertEqual(overrides["speculative_draft_attention_backend"], "flashinfer")
        self.assertNotIn("page_size", overrides)
        self.assertNotIn("mamba_radix_cache_strategy", overrides)

    def test_nemotron_h_target_only_sm120_defers_to_generic_attention_default(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})

        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
        ):
            self.assertNotIn(
                "attention_backend", _nemotron_h_overrides(server_args, hf_config)
            )

    def test_nemotron_h_target_only_sm100_uses_trtllm_mha(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})

        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            self.assertEqual(
                _nemotron_h_overrides(server_args, hf_config)["attention_backend"],
                "trtllm_mha",
            )

    def test_nemotron_h_explicit_split_attention_backend_wins(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(quantized_layers={})
        server_args.speculative_algorithm = "DFLASH"
        server_args.prefill_attention_backend = "triton"
        server_args.speculative_draft_attention_backend = "fa3"

        with (
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            overrides = _nemotron_h_overrides(server_args, hf_config)
        self.assertNotIn("attention_backend", overrides)
        self.assertNotIn("speculative_draft_attention_backend", overrides)

    def test_nemotron_h_w4a16_moe_rejects_a2a_backend(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(
            quantized_layers={
                "backbone.layers.1.mixer.experts.0.up_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
            }
        )
        server_args.moe_a2a_backend = "deepep"

        with self.assertRaisesRegex(ValueError, "moe-a2a-backend=none"):
            _nemotron_h_overrides(server_args, hf_config)

    def test_nemotron_h_w4a16_moe_rejects_non_marlin_runner(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        server_args, hf_config = self._nemotron_h_args(
            quantized_layers={
                "backbone.layers.1.mixer.experts.0.up_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
            }
        )
        server_args.moe_runner_backend = "flashinfer_trtllm"

        with self.assertRaisesRegex(ValueError, "moe-runner-backend=marlin"):
            _nemotron_h_overrides(server_args, hf_config)

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
        self.assertEqual(self._resolved(sa, "swa_full_tokens_ratio"), 1.0)
        self.assertTrue(self._resolved(sa, "disable_hybrid_swa_memory"))
        flags = self._publish(sa)
        self.assertEqual(self._leaf("swa_full_tokens_ratio"), 1.0)
        self.assertTrue(self._leaf("disable_hybrid_swa_memory"))

    def test_gemma2_disables_hybrid_swa_memory(self):
        sa = self._construct("Gemma2ForCausalLM", "llama")
        self.assertTrue(self._resolved(sa, "disable_hybrid_swa_memory"))  # materialized
        self.assertIn(
            ("_gemma2_gemma3_overrides", {"disable_hybrid_swa_memory": True}),
            sa._resolved_overrides,
        )
        self.assertTrue((self._publish(sa), self._leaf("disable_hybrid_swa_memory"))[1])

    def test_olmo2_disables_hybrid_swa_memory(self):
        sa = self._construct("Olmo2ForCausalLM", "llama")
        self.assertTrue(self._resolved(sa, "disable_hybrid_swa_memory"))  # materialized
        self.assertTrue((self._publish(sa), self._leaf("disable_hybrid_swa_memory"))[1])

    def test_exaone_conditional_on_sliding_window_pattern(self):
        # With the pattern the branch also asserts an explicit backend.
        sa = self._construct(
            "Exaone4ForCausalLM",
            "llama",
            config_extra={"sliding_window_pattern": "LLLG"},
            attention_backend="fa3",
        )
        self.assertTrue(self._resolved(sa, "disable_hybrid_swa_memory"))  # materialized
        self.assertTrue((self._publish(sa), self._leaf("disable_hybrid_swa_memory"))[1])

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
        self.assertEqual(self._resolved(sa, "dtype"), "bfloat16")
        self.assertEqual((self._publish(sa), self._leaf("dtype"))[1], "bfloat16")

    def test_gpt_oss_without_mxfp4_keeps_pristine_dtype(self):
        sa = self._construct("GptOssForCausalLM", "llama")
        self.assertEqual(self._resolved(sa, "dtype"), "auto")
        self.assertEqual((self._publish(sa), self._leaf("dtype"))[1], "auto")

    def test_gpt_oss_xpu_dtype_validation_reads_pristine(self):
        from sglang.srt.arg_groups.overrides import _gpt_oss_overrides

        with patch.object(overrides_module, "is_xpu", return_value=True):
            with self.assertRaises(NotImplementedError):
                _gpt_oss_overrides(
                    SimpleNamespace(
                        dtype="float16",
                        attention_backend="triton",
                        prefill_attention_backend=None,
                        decode_attention_backend=None,
                    ),
                    SimpleNamespace(architectures=["GptOssForCausalLM"]),
                )

    def test_sampling_backend_default_pass(self):
        from sglang.srt.utils.common import is_flashinfer_available

        sa = self._construct("LlamaForCausalLM", "llama")
        expected = "flashinfer" if is_flashinfer_available() else "pytorch"
        self.assertEqual(
            self._resolved(sa, "sampling_backend"), expected
        )  # materialized
        self.assertIn(
            ("_sampling_backend_default", {"sampling_backend": expected}),
            sa._resolved_overrides,
        )
        self.assertEqual(
            (self._publish(sa), self._leaf("sampling_backend"))[1], expected
        )

    def test_sampling_backend_user_choice_survives(self):
        sa = self._construct("LlamaForCausalLM", "llama", sampling_backend="pytorch")
        self.assertEqual(self._resolved(sa, "sampling_backend"), "pytorch")
        # the pass declared nothing; publish materializes the pristine choice
        self.assertEqual(
            (self._publish(sa), self._leaf("sampling_backend"))[1], "pytorch"
        )

    def test_deterministic_inference_forces_pytorch_sampling(self):
        sa = self._construct(
            "LlamaForCausalLM", "llama", enable_deterministic_inference=True
        )
        # two pass writers chain: default fill, then the deterministic force --
        # last writer wins. The end state lives in the stash, which is what the
        # projection reads and the bags are built from; the field still holds
        # what the caller passed.
        self.assertEqual(resolution_result(sa, "sampling_backend"), "pytorch")
        flags = self._publish(sa)
        self.assertEqual(self._leaf("sampling_backend"), "pytorch")
        # the deterministic attention fill declared a compatible backend and
        # the compatibility default-fill then had nothing to do
        deterministic_fills = [
            decl["attention_backend"]
            for source, decl in sa._resolved_overrides
            if source == "_deterministic_attention_backend"
        ]
        self.assertEqual(len(deterministic_fills), 1)
        self.assertEqual(
            resolution_result(sa, "attention_backend"), deterministic_fills[0]
        )
        self.assertEqual(self._leaf("attention_backend"), deterministic_fills[0])

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
        self.assertEqual(
            self._resolved(sa, "attention_backend"), "flashinfer"
        )  # materialized
        self.assertIn(
            ("_dllm_attention_backend", {"attention_backend": "flashinfer"}),
            sa._resolved_overrides,
        )
        # the deterministic fill lands on the attention_backend field
        self.assertEqual(
            (self._publish(sa), self._leaf("attention_backend"))[1], "flashinfer"
        )

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
        self.assertEqual(
            self._resolved(sa, "attention_backend"), declared_values[-1]
        )  # materialized
        self.assertEqual(
            (self._publish(sa), self._leaf("attention_backend"))[1], declared_values[-1]
        )

    def test_a_pass_after_resolution_declares_without_writing(self):
        from sglang.srt.arg_groups.overrides import run_post_process_pass

        sa = self._construct("LlamaForCausalLM", "llama")
        raw_before = sa.attention_backend

        def _force_triton(view):
            return {"attention_backend": "triton"}

        run_post_process_pass(sa, _force_triton)

        self.assertEqual("triton", self._resolved(sa, "attention_backend"))
        self.assertEqual(
            (self._publish(sa), self._leaf("attention_backend"))[1], "triton"
        )
        self.assertEqual(
            raw_before,
            sa.attention_backend,
            "the pass wrote the field, so the record stopped answering with the "
            "operator's input",
        )

    def test_a_pass_that_declares_nothing_runs_on_the_published_record(self):
        """A validation slot has to survive a rebuild on the same record.

        `Engine.shutdown()` leaves the launch published, and `Engine(server_args=sa)`
        with the same instance calls `check_server_args()` again before
        republishing. `_hisparse_validation` reaches the pass runner from there
        and returns nothing, so refusing on identity alone would fail the
        second launch.
        """
        from sglang.srt.arg_groups.overrides import run_post_process_pass
        from sglang.srt.runtime_context import publish, reset_context

        sa = self._construct("LlamaForCausalLM", "llama")
        self.addCleanup(reset_context)
        publish(sa, role="scheduler")

        def _declares_nothing(view):
            return {}

        run_post_process_pass(sa, _declares_nothing)  # must not raise

        def _declares_something(view):
            return {"attention_backend": "triton"}

        with self.assertRaisesRegex(ValueError, r"on the published config"):
            run_post_process_pass(sa, _declares_something)

    def test_attention_backend_user_choice_declares_nothing_extra(self):
        sa = self._construct("LlamaForCausalLM", "llama", attention_backend="triton")
        self.assertEqual(self._resolved(sa, "attention_backend"), "triton")
        self.assertEqual(
            (self._publish(sa), self._leaf("attention_backend"))[1], "triton"
        )

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
                _model_config=SimpleNamespace(
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
                moe_a2a_backend="none",
                moe_runner_backend="auto",
                _model_config=SimpleNamespace(is_fp4_experts=True, nvfp4_moe_meta=None),
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        with (
            envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            self.assertEqual(
                _deepseek_v4_overrides(_args(), hf),
                {
                    "attention_backend": "dsv4",
                    "moe_runner_backend": "flashinfer_mxfp4",
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
        # An explicit user choice takes precedence over the model default.
        self.assertNotIn(
            "moe_runner_backend",
            _deepseek_v4_overrides(_args(moe_runner_backend="triton"), hf),
        )
        # FlashInfer MXFP4 only supports the standard (non-A2A) dispatcher.
        with (
            envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            self.assertNotIn(
                "moe_runner_backend",
                _deepseek_v4_overrides(_args(moe_a2a_backend="deepep"), hf),
            )
        # Runtime FP4-to-FP8 dequantization must retain the generic FP8 runner.
        with (
            envs.SGLANG_DSV4_FP4_DEQUANT.override(True),
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
        ):
            self.assertNotIn(
                "moe_runner_backend",
                _deepseek_v4_overrides(_args(), hf),
            )
        # FP8 checkpoints and non-CUDA platforms keep their platform-specific
        # auto-resolution paths.
        fp8_model_config = SimpleNamespace(is_fp4_experts=False, nvfp4_moe_meta=None)
        self.assertNotIn(
            "moe_runner_backend",
            _deepseek_v4_overrides(_args(_model_config=fp8_model_config), hf),
        )
        self.assertNotIn(
            "moe_runner_backend",
            _deepseek_v4_overrides(_args(device="npu"), hf),
        )
        with patch.object(overrides_module, "is_hip", return_value=True):
            self.assertNotIn(
                "moe_runner_backend",
                _deepseek_v4_overrides(_args(), hf),
            )
        # Unsupported NVIDIA architectures keep the generic auto-resolution
        # path instead of selecting a FlashInfer kernel that cannot launch.
        with (
            patch.object(overrides_module, "is_sm90_supported", return_value=False),
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
            patch.object(overrides_module, "is_sm120_supported", return_value=False),
        ):
            self.assertNotIn(
                "moe_runner_backend",
                _deepseek_v4_overrides(_args(), hf),
            )
        # SM120 uses the same model hook; no later pass is needed.
        with (
            envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            patch.object(overrides_module, "is_sm90_supported", return_value=False),
            patch.object(overrides_module, "is_sm100_supported", return_value=False),
            patch.object(overrides_module, "is_sm120_supported", return_value=True),
        ):
            self.assertEqual(
                _deepseek_v4_overrides(_args(), hf)["moe_runner_backend"],
                "flashinfer_mxfp4",
            )
        # nvfp4 hybrid checkpoint routes the MoE runner
        self.assertEqual(
            _deepseek_v4_overrides(
                _args(
                    _model_config=SimpleNamespace(
                        is_fp4_experts=False, nvfp4_moe_meta=object()
                    )
                ),
                hf,
            )["moe_runner_backend"],
            "flashinfer_trtllm_routed",
        )

    def test_nemotron_h_overrides_at_callable_level(self):
        from sglang.srt.arg_groups.overrides import _nemotron_h_overrides

        def _hf(quant_algo="NVFP4", *, include_quantization_config=True):
            hf = SimpleNamespace(
                architectures=["NemotronHForCausalLM"],
                mlp_hidden_act="relu2",
            )
            if include_quantization_config:
                hf.quantization_config = {"quant_algo": quant_algo}
            return hf

        def _args(mc_quant, hf, **kw):
            mc = SimpleNamespace(quantization=mc_quant, hf_config=hf)
            defaults = dict(
                quantization=None,
                moe_runner_backend="auto",
                moe_a2a_backend="none",
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                speculative_algorithm=None,
                speculative_eagle_topk=None,
                speculative_draft_attention_backend=None,
                page_size=None,
                mamba_radix_cache_strategy="auto",
                _model_config=mc,
            )
            defaults.update(kw)
            args = SimpleNamespace(**defaults)
            return args

        hf = _hf()
        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
        ):
            # modelopt checkpoint: quant algo resolution + sm100 defaults
            self.assertEqual(
                _nemotron_h_overrides(_args("modelopt", hf), hf),
                {
                    "quantization": "modelopt_fp4",
                    "moe_runner_backend": "flashinfer_trtllm",
                    "attention_backend": "trtllm_mha",
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

        hf_without_quant_cfg = _hf(include_quantization_config=False)
        with (
            patch.object(overrides_module, "is_sm100_supported", return_value=True),
            patch.object(overrides_module, "is_blackwell_supported", return_value=True),
        ):
            for modelopt_quantization in ("modelopt_fp8", "modelopt_fp4"):
                with self.subTest(modelopt_quantization=modelopt_quantization):
                    self.assertEqual(
                        _nemotron_h_overrides(
                            _args(modelopt_quantization, hf_without_quant_cfg),
                            hf_without_quant_cfg,
                        ),
                        {
                            "quantization": modelopt_quantization,
                            "moe_runner_backend": "flashinfer_trtllm",
                            "attention_backend": "trtllm_mha",
                        },
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
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
            patch.object(overrides_module, "is_hip", return_value=False),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            self.assertEqual(
                _dsa_split_backend_resolution(_view(arch="GlmMoeDsaForCausalLM")),
                {
                    "dsa_prefill_backend": "flashinfer_sparse_mla",
                    "dsa_decode_backend": "flashinfer_sparse_mla",
                },
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
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
                linear_attn_prefill_backend=None,
            )
            defaults.update(kw)
            return ResolvedView(
                SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
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
        self.assertEqual(
            _mamba_radix_cache_resolution(_view("BailingMoeV3ForCausalLM")),
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
        self.assertTrue(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(
                    linear_attn_backend="triton",
                    linear_attn_prefill_backend="flashinfer",
                ),
                "Qwen3_5MoeForConditionalGeneration",
            )
        )

    def test_qwen3_5_hybrid_coupled_declaration(self):
        from sglang.srt.arg_groups.overrides import _qwen3_5_hybrid_overrides

        def _args(default_backend, **kw):
            defaults = dict(
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                mamba_radix_cache_strategy="auto",
                disable_radix_cache=False,
                speculative_algorithm=None,
            )
            defaults.update(kw)
            args = SimpleNamespace(**defaults)
            args.default_backend_for_test = default_backend
            args._model_config = SimpleNamespace(attention_arch=AttentionArch.MHA)
            return args

        with patch.object(
            overrides_module, "is_sm100_supported", return_value=True
        ), patch.object(
            overrides_module,
            "get_default_attn_backend",
            lambda server_args, **_: server_args.default_backend_for_test,
        ):
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

    def test_m3_fp8_attn_gemm_resolution(self):
        from sglang.srt.arg_groups.overrides import _minimax_m3_overrides
        from sglang.srt.server_args import m3_fp8_attn_gemm_enabled

        def _args(**kw):
            defaults = dict(
                attention_backend="trtllm_mha",
                kv_cache_dtype="fp8_e4m3",
            )
            defaults.update(kw)
            return SimpleNamespace(**defaults)

        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=True):
            # e4m3 + trtllm_mha + SM100: mode active
            self.assertTrue(m3_fp8_attn_gemm_enabled(_args()))
            # fa4 dense backend: mode inactive (no fp8-q GEMM path)
            self.assertFalse(m3_fp8_attn_gemm_enabled(_args(attention_backend="fa4")))
            # bf16 KV: mode inactive
            self.assertFalse(m3_fp8_attn_gemm_enabled(_args(kv_cache_dtype="auto")))
            # e5m2: mode inactive (fmha_sm100's variant lookup would silently
            # dispatch the e4m3 kernel)
            self.assertFalse(m3_fp8_attn_gemm_enabled(_args(kv_cache_dtype="fp8_e5m2")))
            # SGLANG_DISABLE_M3_FP8_ATTN_GEMM kill switch wins over an
            # otherwise-active config
            with envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.override(True):
                self.assertFalse(m3_fp8_attn_gemm_enabled(_args()))
        with patch("sglang.srt.utils.common.is_sm100_supported", return_value=False):
            # non-SM100: mode inactive
            self.assertFalse(m3_fp8_attn_gemm_enabled(_args()))

        def _m3_args(**kw):
            defaults = dict(
                quantization=None,
                _quantization_explicitly_unset=True,
                attention_backend=None,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                page_size=None,
                moe_runner_backend="auto",
                kv_cache_dtype="auto",
            )
            defaults.update(kw)
            ns = SimpleNamespace(**defaults)
            return ns

        hf = SimpleNamespace()
        with patch.object(overrides_module, "is_hip", return_value=False), patch.object(
            overrides_module, "is_sm100_supported", return_value=True
        ), patch.object(overrides_module, "get_quantization_config", return_value=None):
            # fp8_e4m3 KV: SM100 backend default flips to trtllm_mha (the only
            # dense backend with the fp8-q GEMM path); page snaps to 128
            ov = _minimax_m3_overrides(_m3_args(kv_cache_dtype="fp8_e4m3"), hf)
            self.assertEqual(ov["attention_backend"], "trtllm_mha")
            self.assertEqual(ov["page_size"], 128)
            # auto KV: fa4 stays the SM100 default
            ov = _minimax_m3_overrides(_m3_args(), hf)
            self.assertEqual(ov["attention_backend"], "fa4")
            self.assertEqual(ov["page_size"], 128)
            # e5m2 KV: stays on fa4 + the widening Triton path, and warns
            with self.assertLogs(
                "sglang.srt.arg_groups.overrides", level="WARNING"
            ) as logs:
                ov = _minimax_m3_overrides(_m3_args(kv_cache_dtype="fp8_e5m2"), hf)
            self.assertEqual(ov["attention_backend"], "fa4")
            self.assertIn("fp8_e5m2", "\n".join(logs.output))
            # explicit backend choice is never overridden
            ov = _minimax_m3_overrides(
                _m3_args(kv_cache_dtype="fp8_e4m3", attention_backend="fa4"), hf
            )
            self.assertNotIn("attention_backend", ov)
            # kill switch also reverts the SM100 backend default to fa4
            with envs.SGLANG_DISABLE_M3_FP8_ATTN_GEMM.override(True):
                ov = _minimax_m3_overrides(_m3_args(kv_cache_dtype="fp8_e4m3"), hf)
            self.assertEqual(ov["attention_backend"], "fa4")
            self.assertEqual(ov["page_size"], 128)

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
                speculative_draft_attention_backend=None,
                page_size=1,
                # `use_mla_backend` reads the model configuration; a non-MLA
                # one keeps these assertions about the page constraints.
                _model_config=SimpleNamespace(attention_arch=None),
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
        # trtllm_mha accepts 128 (trtllm-gen dynamic tokens-per-page kernels)
        self.assertEqual(
            _mla_backend_page_constraints(
                _view(attention_backend="trtllm_mha", page_size=128)
            ),
            {},
        )
        # trtllm_mha with an unsupported page still snaps to 64
        self.assertEqual(
            _mla_backend_page_constraints(
                _view(attention_backend="trtllm_mha", page_size=256)
            ),
            {"page_size": 64},
        )
        # chained: cutlass_mla decode -> 128, then trtllm_mha prefill keeps 128
        self.assertEqual(
            _mla_backend_page_constraints(
                _view(
                    decode_attention_backend="cutlass_mla",
                    prefill_attention_backend="trtllm_mha",
                )
            ),
            {"page_size": 128},
        )
        # no matching backend: nothing declared
        self.assertEqual(_mla_backend_page_constraints(_view()), {})

        with patch.object(overrides_module, "is_sm100_supported", return_value=True):
            self.assertEqual(
                _fa4_page_constraint(
                    _view(
                        attention_backend="fa4",
                        speculative_eagle_topk=None,
                    )
                ),
                {"page_size": 128},
            )
            self.assertEqual(
                _fa4_page_constraint(
                    _view(
                        attention_backend="fa4",
                        speculative_eagle_topk=2,  # EAGLE topk>1 keeps default
                    )
                ),
                {},
            )

        self.assertEqual(
            _intel_xpu_page_constraint(
                _view(
                    decode_attention_backend="intel_xpu",
                )
            ),
            {"page_size": 128},
        )
        self.assertEqual(
            _intel_xpu_page_constraint(
                _view(
                    decode_attention_backend="intel_xpu",
                    _model_config=SimpleNamespace(attention_arch=AttentionArch.MLA),
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
                prefill_attention_backend=None,
                decode_attention_backend=None,
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
                _model_config=SimpleNamespace(
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

        self.assertEqual(
            _a2a_ep_size(
                ResolvedView(
                    SimpleNamespace(moe_a2a_backend="deepep", ep_size=1, tp_size=8)
                )
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
                attention_backend="triton",
                prefill_attention_backend=None,
                decode_attention_backend=None,
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
