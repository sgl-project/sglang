"""Unit tests for the model-override machinery: whitelist metadata, registry
(V3a — declarations only, nothing calls this in production yet)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
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
from sglang.srt.runtime_context import _StaticFlags
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

    def test_server_args_whitelist_empty_at_skeleton(self):
        # No ServerArgs field is tagged yet: the V3 sweeps whitelist fields
        # one family at a time. This pin makes accidental tagging visible.
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(model_overridable_fields(ServerArgs), frozenset())


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
