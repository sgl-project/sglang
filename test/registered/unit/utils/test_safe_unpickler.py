import pickle
import unittest
from collections import OrderedDict, defaultdict, deque
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.utils.common import MultiprocessingSerializer, safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestSafeUnpickler(CustomTestCase):
    def test_rejects_dangerous_builtin_globals(self):
        for name in ("__import__", "getattr", "eval", "exec", "compile", "open"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \(builtins\.{name}\)"
                ),
            ):
                # GLOBAL resolves the callable but does not invoke it. This exercises
                # the deserialization boundary without constructing an exploit chain.
                safe_pickle_loads(f"cbuiltins\n{name}\n.".encode())

    def test_rejects_unlisted_standard_library_globals(self):
        for module, name in (
            ("copyreg", "_reconstructor"),
            ("operator", "attrgetter"),
            ("types", "FunctionType"),
        ):
            with (
                self.subTest(module=module, name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \({module}\.{name}\)"
                ),
            ):
                safe_pickle_loads(f"c{module}\n{name}\n.".encode())

    def test_round_trips_safe_standard_library_types(self):
        value = SimpleNamespace(
            values=OrderedDict([("items", deque([1, 2]))]),
            factory=defaultdict(list, {"items": [3]}),
            index=slice(1, 4),
            parser=partial(int, base=10),
        )

        restored = safe_pickle_loads(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )

        self.assertEqual(restored.values, value.values)
        self.assertEqual(restored.factory, value.factory)
        self.assertEqual(restored.index, value.index)
        self.assertEqual(restored.parser("11"), 11)

    def test_round_trips_tensor_payload(self):
        value = [("weight", torch.arange(6).reshape(2, 3))]

        restored = MultiprocessingSerializer.deserialize(
            MultiprocessingSerializer.serialize(value)
        )

        self.assertEqual(restored[0][0], "weight")
        self.assertTrue(torch.equal(restored[0][1], value[0][1]))


if __name__ == "__main__":
    unittest.main()
