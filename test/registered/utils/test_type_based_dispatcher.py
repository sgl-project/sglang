import unittest

from sglang.srt.utils.common import TypeBasedDispatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Base:
    pass


class Child(Base):
    pass


class GrandChild(Child):
    pass


class Unknown:
    pass


class TestTypeBasedDispatcher(CustomTestCase):
    def test_exact_type_then_first_registered_base(self):
        dispatcher = TypeBasedDispatcher(
            [(Base, lambda _: "base"), (Child, lambda _: "child")]
        )
        self.assertEqual(dispatcher(Child()), "child")
        # No exact match: registration order decides, not MRO distance; the
        # second call is served from the MRO cache.
        self.assertEqual(dispatcher(GrandChild()), "base")
        self.assertEqual(dispatcher(GrandChild()), "base")

    def test_unknown_type_raises_until_fallback_is_set(self):
        dispatcher = TypeBasedDispatcher([(Base, lambda _: "base")])
        for _ in range(2):
            with self.assertRaises(ValueError):
                dispatcher(Unknown())
        dispatcher.add_fallback_fn(lambda _: "fallback")
        self.assertEqual(dispatcher(Unknown()), "fallback")

    def test_iadd_keeps_existing_handlers(self):
        dispatcher = TypeBasedDispatcher([(Base, lambda _: "first")])
        dispatcher += TypeBasedDispatcher(
            [(Base, lambda _: "second"), (Unknown, lambda _: "unknown")]
        )
        self.assertEqual(dispatcher(Base()), "first")
        self.assertEqual(dispatcher(Unknown()), "unknown")


if __name__ == "__main__":
    unittest.main()
