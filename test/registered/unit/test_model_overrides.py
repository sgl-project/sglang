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
            frozenset({"dtype", "enable_tf32_matmul", "enable_multi_layer_eagle"}),
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
    """R0 publish wiring: stash-carrying publishes resolve into flags via the
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

    def _construct(self, arch, model_type, **server_kwargs):
        from sglang.srt.server_args import ServerArgs

        # Golden resolution must be host-independent: accelerator-less CI
        # runners resolve only the base platform, where get_device() raises.
        server_kwargs.setdefault("device", "cuda")
        config = dict(self._MINI_CONFIG, architectures=[arch], model_type=model_type)
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
        self.assertEqual(
            sa._resolved_overrides,
            [("MODEL_OVERRIDES['MistralLarge3ForCausalLM']", {"dtype": "bfloat16"})],
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
        self.assertEqual(sa._resolved_overrides, [])
        # publish still materializes the whitelisted leaf with the pristine
        # value: readers only ever read flags.
        self.assertEqual(self._publish(sa).dtype, "auto")

    def test_minimax_m2_enables_tf32_matmul(self):
        sa = self._construct("MiniMaxM2ForCausalLM", "llama")
        self.assertTrue(sa.enable_tf32_matmul)  # dual-apply == legacy write
        self.assertEqual(
            sa._resolved_overrides,
            [("_minimax_m2_overrides", {"enable_tf32_matmul": True})],
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
