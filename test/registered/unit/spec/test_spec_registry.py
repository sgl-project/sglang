"""Unit tests for the speculative algorithm plugin registry."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    _REGISTRY,
    CustomSpecAlgo,
    _assert_custom_spec_algo_conforms,
    _reserved_names,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RegistryIsolated(CustomTestCase):
    """Snapshot and restore the global registry so tests don't leak."""

    def setUp(self):
        self._snapshot = _REGISTRY.copy()
        _REGISTRY.clear()

    def tearDown(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._snapshot)


class TestFromString(_RegistryIsolated):
    def test_none_input_returns_none_member(self):
        self.assertIs(SpeculativeAlgorithm.from_string(None), SpeculativeAlgorithm.NONE)

    def test_builtin_name_returns_enum(self):
        self.assertIs(
            SpeculativeAlgorithm.from_string("EAGLE"), SpeculativeAlgorithm.EAGLE
        )
        self.assertIs(
            SpeculativeAlgorithm.from_string("NGRAM"), SpeculativeAlgorithm.NGRAM
        )

    def test_builtin_name_is_case_insensitive(self):
        self.assertIs(
            SpeculativeAlgorithm.from_string("eagle"), SpeculativeAlgorithm.EAGLE
        )

    def test_unknown_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown speculative algorithm"):
            SpeculativeAlgorithm.from_string("NOT_REGISTERED")

    def test_registered_plugin_returns_custom_spec(self):
        @SpeculativeAlgorithm.register("MY_FOO")
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertEqual(algo.name, "MY_FOO")

    def test_registered_plugin_lookup_is_case_insensitive(self):
        @SpeculativeAlgorithm.register("MY_FOO")
        def _factory(server_args):
            return MagicMock

        self.assertIs(
            SpeculativeAlgorithm.from_string("my_foo"),
            SpeculativeAlgorithm.from_string("MY_FOO"),
        )


class TestRegister(_RegistryIsolated):
    def test_register_returns_factory_unchanged(self):
        def _factory(server_args):
            return MagicMock

        decorated = SpeculativeAlgorithm.register("MY_FOO")(_factory)
        self.assertIs(decorated, _factory)

    def test_two_distinct_registrations_are_independent(self):
        @SpeculativeAlgorithm.register("FOO")
        def _foo_factory(server_args):
            return MagicMock

        @SpeculativeAlgorithm.register("BAR")
        def _bar_factory(server_args):
            return MagicMock

        foo = SpeculativeAlgorithm.from_string("FOO")
        bar = SpeculativeAlgorithm.from_string("BAR")
        self.assertIsNot(foo, bar)
        self.assertNotEqual(foo, bar)
        self.assertEqual(foo.name, "FOO")
        self.assertEqual(bar.name, "BAR")

    def test_duplicate_name_raises(self):
        @SpeculativeAlgorithm.register("MY_FOO")
        def _factory(server_args):
            return MagicMock

        with self.assertRaisesRegex(ValueError, "already registered"):

            @SpeculativeAlgorithm.register("MY_FOO")
            def _factory2(server_args):
                return MagicMock

    def test_reserved_name_raises(self):
        for reserved in _reserved_names():
            with self.assertRaisesRegex(ValueError, "reserved"):
                SpeculativeAlgorithm.register(reserved)

    def test_reserved_names_cover_all_enum_members(self):
        # Reserved names are derived from the enum, so every builtin (including
        # FROZEN_KV_MTP, which a hand-maintained list had omitted) is reserved.
        for member in SpeculativeAlgorithm:
            self.assertIn(member.name, _reserved_names())
        self.assertIn("NEXTN", _reserved_names())  # CLI alias

    def test_reserved_name_is_case_insensitive(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            SpeculativeAlgorithm.register("frozen_kv_mtp")

    def test_register_is_case_insensitive_on_collision(self):
        @SpeculativeAlgorithm.register("MY_FOO")
        def _factory(server_args):
            return MagicMock

        with self.assertRaisesRegex(ValueError, "already registered"):

            @SpeculativeAlgorithm.register("my_foo")
            def _factory2(server_args):
                return MagicMock


class TestCustomSpecAlgoInterface(_RegistryIsolated):
    """CustomSpecAlgo must duck-type SpeculativeAlgorithm enum values."""

    def setUp(self):
        super().setUp()

        @SpeculativeAlgorithm.register("MY_FOO", supports_overlap=False)
        def _factory(server_args):
            return MagicMock

        self.algo = SpeculativeAlgorithm.from_string("MY_FOO")

    def test_is_predicates_all_false_except_speculative(self):
        self.assertFalse(self.algo.is_none())
        self.assertFalse(self.algo.is_eagle())
        self.assertFalse(self.algo.is_eagle3())
        self.assertFalse(self.algo.is_frozen_kv_mtp())
        self.assertFalse(self.algo.is_dflash())
        self.assertFalse(self.algo.is_standalone())
        self.assertFalse(self.algo.is_ngram())
        self.assertTrue(self.algo.is_speculative())
        # A registered plugin is never NONE -> is_some() mirrors the enum.
        self.assertTrue(self.algo.is_some())

    def test_is_some_matches_enum_semantics(self):
        # is_some() is called on spec algos in overlap_utils.py; a CustomSpecAlgo
        # must answer it the same way the enum does (True iff not NONE).
        self.assertEqual(self.algo.is_some(), not self.algo.is_none())
        self.assertEqual(SpeculativeAlgorithm.EAGLE.is_some(), self.algo.is_some())

    def test_supports_overlap_false_warns_deprecation(self):
        # supports_overlap=False plugins run the V2 schema synchronously; the
        # removed V1 path is surfaced as a deprecation warning at create time.
        server_args = MagicMock()
        server_args.disable_overlap_schedule = True
        with self.assertLogs("sglang.srt.speculative.spec_registry", "WARNING") as logs:
            self.algo.create_worker(server_args)
        self.assertTrue(any("deprecated" in line for line in logs.output))

        @SpeculativeAlgorithm.register("MY_V2", supports_overlap=True)
        def _factory(server_args):
            return MagicMock

        v2 = SpeculativeAlgorithm.from_string("MY_V2")
        server_args.disable_overlap_schedule = False
        self.assertIs(v2.create_worker(server_args), MagicMock)

    def test_create_worker_calls_factory(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = True
        worker_cls = self.algo.create_worker(server_args)
        self.assertIs(worker_cls, MagicMock)

    def test_create_worker_raises_on_overlap_mismatch(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = False
        with self.assertRaisesRegex(ValueError, "does not support overlap"):
            self.algo.create_worker(server_args)


class TestValidatorHook(_RegistryIsolated):
    def test_validator_invocation_is_caller_driven(self):
        validator = MagicMock()

        @SpeculativeAlgorithm.register("MY_FOO", validate_server_args=validator)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertIs(algo.validate_server_args, validator)
        # Callers (e.g. ServerArgs.__post_init__) must invoke the hook themselves;
        # CustomSpecAlgo does not call it from create_worker.
        validator.assert_not_called()


class TestSubclassOverride(_RegistryIsolated):
    """Plugins can subclass CustomSpecAlgo to override is_*() / create_worker."""

    def test_subclass_overrides_is_eagle(self):
        class EagleLike(CustomSpecAlgo):
            def is_eagle(self) -> bool:
                return True

        @SpeculativeAlgorithm.register(
            "MY_LIKE_EAGLE", supports_overlap=True, spec_class=EagleLike
        )
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_LIKE_EAGLE")
        self.assertIsInstance(algo, EagleLike)
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertTrue(algo.is_eagle())
        # Other predicates default to False
        self.assertFalse(algo.is_ngram())
        self.assertFalse(algo.is_dflash())

    def test_subclass_overrides_create_worker(self):
        class CustomDispatch(CustomSpecAlgo):
            def create_worker(self, server_args):
                return "custom-dispatched"

        @SpeculativeAlgorithm.register("MY_CUSTOM", spec_class=CustomDispatch)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_CUSTOM")
        # Custom dispatch bypasses default overlap check
        self.assertEqual(algo.create_worker(MagicMock()), "custom-dispatched")


class TestConformanceGuard(_RegistryIsolated):
    """register_algorithm rejects spec classes that drift from the enum's
    is_*() / supports_*() duck-typing interface."""

    def test_base_custom_spec_algo_conforms(self):
        # The shipped base class must satisfy its own contract.
        _assert_custom_spec_algo_conforms(CustomSpecAlgo)

    def test_conforming_subclass_passes(self):
        class Good(CustomSpecAlgo):
            def is_eagle(self) -> bool:
                return True

        _assert_custom_spec_algo_conforms(Good)  # does not raise

    @staticmethod
    def _spec_class_missing(method: str) -> type:
        # Build a class exposing the full enum interface except `method`. A real
        # subclass can't be "missing" an inherited method, so the failure mode
        # the guard catches is the base class itself dropping one — simulated
        # here with a standalone class.
        interface = {
            name
            for name in vars(SpeculativeAlgorithm)
            if name.startswith(("is_", "supports_"))
        }
        body = {m: (lambda self: False) for m in interface if m != method}
        return type("Broken", (), body)

    def test_missing_predicate_raises(self):
        Broken = self._spec_class_missing("is_some")
        with self.assertRaisesRegex(TypeError, "is_some"):
            _assert_custom_spec_algo_conforms(Broken)

    def test_register_rejects_nonconforming_spec_class(self):
        Broken = self._spec_class_missing("is_some")
        with self.assertRaisesRegex(TypeError, "missing duck-typed methods"):

            @SpeculativeAlgorithm.register("MY_BROKEN", spec_class=Broken)
            def _factory(server_args):
                return MagicMock

    def test_enum_interface_subset_of_custom_spec_algo(self):
        # Every is_*/supports_* method on the enum exists on CustomSpecAlgo.
        # vars() (not dir()) because EnumMeta.__dir__ hides instance methods.
        interface = {
            name
            for name in vars(SpeculativeAlgorithm)
            if name.startswith(("is_", "supports_"))
        }
        self.assertTrue(interface)  # guard against an empty (no-op) interface
        self.assertEqual(interface - set(dir(CustomSpecAlgo)), set())


class TestCrossTypeIdentity(_RegistryIsolated):
    """A plugin algo and a builtin enum value must never compare equal."""

    def test_plugin_not_equal_to_builtin(self):
        @SpeculativeAlgorithm.register("MY_FOO")
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string("MY_FOO")
        self.assertNotEqual(algo, SpeculativeAlgorithm.EAGLE)
        self.assertNotEqual(algo, SpeculativeAlgorithm.NONE)
        self.assertIsNot(algo, SpeculativeAlgorithm.EAGLE)


if __name__ == "__main__":
    unittest.main(verbosity=3)
