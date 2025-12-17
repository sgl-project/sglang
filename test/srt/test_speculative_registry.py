import unittest

from sglang.srt.speculative import spec_info as spec_info_module
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    register_speculative_algorithm,
)


class DummyWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class SpeculativeRegistryTests(unittest.TestCase):
    def test_nextn_alias_maps_to_eagle(self):
        eagle = SpeculativeAlgorithm.from_string("EAGLE")
        alias = SpeculativeAlgorithm.from_string("NEXTN")
        self.assertIs(alias, eagle)

    def test_register_speculative_algorithm_registers_worker_and_flags(self):
        original_next_value = SpeculativeAlgorithm._next_value
        algo = register_speculative_algorithm(
            "TEST_SPEC_ALGO",
            DummyWorker,
            aliases=("TEST_SPEC_ALIAS",),
            flags=("EAGLE",),
            override_worker=True,
        )
        self.addCleanup(self._cleanup_registered_algorithm, algo, ("TEST_SPEC_ALIAS",))
        self.addCleanup(
            setattr, SpeculativeAlgorithm, "_next_value", original_next_value
        )

        self.assertIs(SpeculativeAlgorithm.from_string("TEST_SPEC_ALGO"), algo)
        self.assertIs(SpeculativeAlgorithm.from_string("TEST_SPEC_ALIAS"), algo)
        self.assertTrue(algo.is_eagle())
        self.assertIs(SpeculativeAlgorithm.from_value(int(algo)), algo)
        self.assertIn(algo, list(spec_info_module._REGISTERED_WORKERS))

        worker = algo.create_draft_worker(example_arg=42)
        self.assertIsInstance(worker, DummyWorker)
        self.assertEqual(worker.kwargs["example_arg"], 42)

    def test_builtin_algorithms_flags_and_factories(self):
        cases = {
            "NONE": {
                "is_none": True,
                "is_eagle": False,
                "is_eagle3": False,
                "is_standalone": False,
                "is_ngram": False,
                "has_factory": False,
            },
            "EAGLE": {
                "is_none": False,
                "is_eagle": True,
                "is_eagle3": False,
                "is_standalone": False,
                "is_ngram": False,
                "has_factory": True,
            },
            "EAGLE3": {
                "is_none": False,
                "is_eagle": True,
                "is_eagle3": True,
                "is_standalone": False,
                "is_ngram": False,
                "has_factory": True,
            },
            "STANDALONE": {
                "is_none": False,
                "is_eagle": False,
                "is_eagle3": False,
                "is_standalone": True,
                "is_ngram": False,
                "has_factory": True,
            },
            "NGRAM": {
                "is_none": False,
                "is_eagle": False,
                "is_eagle3": False,
                "is_standalone": False,
                "is_ngram": True,
                "has_factory": True,
            },
        }

        for name, expectations in cases.items():
            with self.subTest(name=name):
                algo = SpeculativeAlgorithm.from_string(name)
                self.assertEqual(algo.name, name)
                self.assertEqual(algo.is_none(), expectations["is_none"])
                self.assertEqual(algo.is_eagle(), expectations["is_eagle"])
                self.assertEqual(algo.is_eagle3(), expectations["is_eagle3"])
                self.assertEqual(algo.is_standalone(), expectations["is_standalone"])
                self.assertEqual(algo.is_ngram(), expectations["is_ngram"])

                has_factory = algo._draft_worker_factory is not None
                self.assertEqual(has_factory, expectations["has_factory"])
                self.assertIs(SpeculativeAlgorithm.from_value(int(algo)), algo)

        self.assertIs(SpeculativeAlgorithm.from_string(None), SpeculativeAlgorithm.NONE)

    def test_iteration_returns_registration_order(self):
        names = [algo.name for algo in SpeculativeAlgorithm._registration_order]
        for required in ["NONE", "EAGLE", "EAGLE3", "STANDALONE", "NGRAM"]:
            self.assertIn(required, names)

    def test_create_draft_worker_returns_none_for_none_algorithm(self):
        self.assertIsNone(SpeculativeAlgorithm.NONE.create_draft_worker())

    def test_register_draft_worker_override(self):
        algo = SpeculativeAlgorithm.from_string("EAGLE")
        original_factory = algo._draft_worker_factory

        def dummy_factory(_: SpeculativeAlgorithm, **kwargs):
            return "dummy"

        SpeculativeAlgorithm.register_draft_worker(algo, dummy_factory)
        self.addCleanup(
            SpeculativeAlgorithm.register_draft_worker, algo, original_factory
        )

        self.assertEqual(algo.create_draft_worker(), "dummy")

    def _cleanup_registered_algorithm(self, algorithm: SpeculativeAlgorithm, aliases):
        name = algorithm.name
        SpeculativeAlgorithm._registry_by_value.pop(algorithm.value, None)
        SpeculativeAlgorithm._registry_by_name.pop(name, None)
        if hasattr(SpeculativeAlgorithm, name):
            delattr(SpeculativeAlgorithm, name)

        for alias in aliases:
            SpeculativeAlgorithm._registry_by_name.pop(alias, None)

        try:
            SpeculativeAlgorithm._registration_order.remove(algorithm)
        except ValueError:
            pass

        for flag_values in SpeculativeAlgorithm._flags.values():
            flag_values.discard(algorithm.value)

        spec_info_module._REGISTERED_WORKERS.pop(algorithm, None)


if __name__ == "__main__":
    unittest.main()
