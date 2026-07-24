"""Unit tests for pass_context, InductorPass, and CallableInductorPass in srt/compilation/inductor_pass.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.compilation.inductor_pass import (
    CallableInductorPass,
    InductorPass,
    PassContext,
    get_pass_context,
    pass_context,
)
from sglang.test.test_utils import CustomTestCase


class TestPassContextManager(CustomTestCase):
    def test_get_pass_context_raises_outside_context(self):
        with self.assertRaises(AssertionError):
            get_pass_context()

    def test_get_pass_context_returns_correct_shape_inside(self):
        with pass_context(42):
            ctx = get_pass_context()
            self.assertEqual(ctx.runtime_shape, 42)

    def test_get_pass_context_returns_none_shape_when_none_passed(self):
        with pass_context(None):
            ctx = get_pass_context()
            self.assertIsNone(ctx.runtime_shape)

    def test_context_restored_to_none_on_normal_exit(self):
        with pass_context(10):
            pass
        with self.assertRaises(AssertionError):
            get_pass_context()

    def test_context_restored_after_exception_inside(self):
        try:
            with pass_context(99):
                raise RuntimeError("intentional")
        except RuntimeError:
            pass
        with self.assertRaises(AssertionError):
            get_pass_context()

    def test_nested_inner_context_overrides_shape(self):
        with pass_context(10):
            self.assertEqual(get_pass_context().runtime_shape, 10)
            with pass_context(20):
                self.assertEqual(get_pass_context().runtime_shape, 20)
            # outer restored correctly
            self.assertEqual(get_pass_context().runtime_shape, 10)

    def test_pass_context_is_a_pass_context_instance(self):
        with pass_context(7):
            ctx = get_pass_context()
            self.assertIsInstance(ctx, PassContext)


class TestInductorPassHashSource(CustomTestCase):
    def test_hash_source_string_is_deterministic(self):
        h1 = InductorPass.hash_source("hello world")
        h2 = InductorPass.hash_source("hello world")
        self.assertEqual(h1, h2)

    def test_hash_source_different_strings_differ(self):
        h1 = InductorPass.hash_source("alpha")
        h2 = InductorPass.hash_source("beta")
        self.assertNotEqual(h1, h2)

    def test_hash_source_function_is_deterministic(self):
        def my_func():
            return 1

        h1 = InductorPass.hash_source(my_func)
        h2 = InductorPass.hash_source(my_func)
        self.assertEqual(h1, h2)

    def test_hash_source_different_functions_differ(self):
        def func_a():
            return 1

        def func_b():
            return 2

        h1 = InductorPass.hash_source(func_a)
        h2 = InductorPass.hash_source(func_b)
        self.assertNotEqual(h1, h2)

    def test_hash_source_class_is_deterministic(self):
        class MyPass(InductorPass):
            def __call__(self, graph):
                pass

        p = MyPass()
        h1 = InductorPass.hash_source(p)
        h2 = InductorPass.hash_source(p)
        self.assertEqual(h1, h2)

    def test_hash_source_returns_hex_string(self):
        h = InductorPass.hash_source("test")
        self.assertIsInstance(h, str)
        # sha256 hexdigest is 64 hex chars
        self.assertEqual(len(h), 64)
        int(h, 16)  # must be valid hex


class TestInductorPassHashDict(CustomTestCase):
    def test_hash_dict_is_deterministic(self):
        d = {"a": 1, "b": 2}
        h1 = InductorPass.hash_dict(d)
        h2 = InductorPass.hash_dict(d)
        self.assertEqual(h1, h2)

    def test_hash_dict_key_order_independent(self):
        h1 = InductorPass.hash_dict({"a": 1, "b": 2})
        h2 = InductorPass.hash_dict({"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_hash_dict_different_contents_differ(self):
        h1 = InductorPass.hash_dict({"x": 1})
        h2 = InductorPass.hash_dict({"x": 2})
        self.assertNotEqual(h1, h2)

    def test_hash_dict_returns_hex_string(self):
        h = InductorPass.hash_dict({"k": "v"})
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)


class TestCallableInductorPass(CustomTestCase):
    def test_call_invokes_the_wrapped_callable(self):
        calls = []

        def tracker(graph):
            calls.append(graph)

        p = CallableInductorPass(tracker)
        sentinel = object()
        p(sentinel)
        self.assertEqual(calls, [sentinel])

    def test_uuid_matches_hash_source_of_callable(self):
        def my_fn(graph):
            pass

        p = CallableInductorPass(my_fn)
        expected = InductorPass.hash_source(my_fn)
        self.assertEqual(p.uuid(), expected)

    def test_explicit_uuid_overrides_auto_hash(self):
        def my_fn(graph):
            pass

        p = CallableInductorPass(my_fn, uuid="custom-uuid")
        self.assertEqual(p.uuid(), "custom-uuid")

    def test_uuid_is_stable_across_calls(self):
        def fn(g):
            pass

        p = CallableInductorPass(fn)
        self.assertEqual(p.uuid(), p.uuid())

    def test_two_callable_passes_with_different_fns_have_different_uuids(self):
        def fn_a(g):
            return 1

        def fn_b(g):
            return 2

        pa = CallableInductorPass(fn_a)
        pb = CallableInductorPass(fn_b)
        self.assertNotEqual(pa.uuid(), pb.uuid())


if __name__ == "__main__":
    unittest.main()
