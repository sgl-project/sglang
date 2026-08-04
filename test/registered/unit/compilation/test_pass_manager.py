"""Unit tests for PostGradPassManager in srt/compilation/pass_manager.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch.fx as fx

from sglang.srt.compilation.fix_functionalization import FixFunctionalizationPass
from sglang.srt.compilation.inductor_pass import (
    SGLangInductorPass,
    pass_context,
)
from sglang.srt.compilation.pass_manager import PostGradPassManager
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Stub passes used across tests
# ---------------------------------------------------------------------------


class _AlwaysApplicablePass(SGLangInductorPass):
    """A pass that always runs and records calls."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def __call__(self, graph: fx.Graph):
        self.call_count += 1

    def is_applicable_for_shape(self, shape):
        return True


class _ShapeFilteredPass(SGLangInductorPass):
    """A pass that is only applicable for a specific shape."""

    def __init__(self, accept_shape):
        super().__init__()
        self.accept_shape = accept_shape
        self.call_count = 0

    def __call__(self, graph: fx.Graph):
        self.call_count += 1

    def is_applicable_for_shape(self, shape):
        return shape == self.accept_shape


def _empty_graph() -> fx.Graph:
    """Return a minimal valid FX graph (placeholder + output) for pass calls."""
    g = fx.Graph()
    g.output(None)
    return g


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestPostGradPassManagerAdd(CustomTestCase):
    def test_add_with_valid_inductor_pass_succeeds(self):
        pm = PostGradPassManager()
        pm.add(_AlwaysApplicablePass())
        self.assertEqual(len(pm.passes), 1)

    def test_add_with_non_inductor_pass_raises_assertion_error(self):
        pm = PostGradPassManager()
        with self.assertRaises(AssertionError):
            pm.add(object())  # not an InductorPass

    def test_add_multiple_passes_all_stored(self):
        pm = PostGradPassManager()
        pm.add(_AlwaysApplicablePass())
        pm.add(_AlwaysApplicablePass())
        self.assertEqual(len(pm.passes), 2)

    def test_add_with_plain_function_raises(self):
        pm = PostGradPassManager()
        with self.assertRaises(AssertionError):
            pm.add(lambda g: None)


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


class TestPostGradPassManagerConfigure(CustomTestCase):
    def test_configure_creates_fix_functionalization_attribute(self):
        pm = PostGradPassManager()
        pm.configure()
        self.assertTrue(hasattr(pm, "fix_functionalization"))

    def test_configure_creates_fix_functionalization_pass_instance(self):
        pm = PostGradPassManager()
        pm.configure()
        self.assertIsInstance(pm.fix_functionalization, FixFunctionalizationPass)

    def test_configure_creates_pass_config_dict(self):
        pm = PostGradPassManager()
        pm.configure()
        self.assertIsInstance(pm.pass_config, dict)


# ---------------------------------------------------------------------------
# __call__() -- shape-based dispatch
# ---------------------------------------------------------------------------


class TestPostGradPassManagerCall(CustomTestCase):
    def _make_configured_pm(self):
        pm = PostGradPassManager()
        pm.configure()
        return pm

    def test_applicable_pass_is_invoked(self):
        pm = self._make_configured_pm()
        p = _AlwaysApplicablePass()
        pm.add(p)
        g = _empty_graph()
        with pass_context(8):
            pm(g)
        self.assertEqual(p.call_count, 1)

    def test_inapplicable_pass_is_skipped(self):
        pm = self._make_configured_pm()
        # Only applicable for shape 99, but we run with shape 8.
        p = _ShapeFilteredPass(accept_shape=99)
        pm.add(p)
        g = _empty_graph()
        with pass_context(8):
            pm(g)
        self.assertEqual(p.call_count, 0)

    def test_shape_filtered_pass_runs_on_matching_shape(self):
        pm = self._make_configured_pm()
        p = _ShapeFilteredPass(accept_shape=42)
        pm.add(p)
        g = _empty_graph()
        with pass_context(42):
            pm(g)
        self.assertEqual(p.call_count, 1)

    def test_fix_functionalization_always_runs(self):
        pm = self._make_configured_pm()
        # Add a pass that is never applicable; fix_functionalization should still run.
        p = _ShapeFilteredPass(accept_shape=-1)
        pm.add(p)
        g = _empty_graph()
        # Should not raise; fix_functionalization runs even when user passes are skipped.
        with pass_context(7):
            pm(g)
        self.assertEqual(p.call_count, 0)

    def test_passes_invoked_in_registration_order(self):
        pm = self._make_configured_pm()
        order = []

        class _OrderRecorder(SGLangInductorPass):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag

            def __call__(self, graph):
                order.append(self.tag)

        pm.add(_OrderRecorder("first"))
        pm.add(_OrderRecorder("second"))
        g = _empty_graph()
        with pass_context(1):
            pm(g)
        self.assertEqual(order, ["first", "second"])

    def test_call_without_pass_context_raises(self):
        pm = self._make_configured_pm()
        g = _empty_graph()
        with self.assertRaises(AssertionError):
            pm(g)  # no pass_context active


# ---------------------------------------------------------------------------
# uuid()
# ---------------------------------------------------------------------------


class TestPostGradPassManagerUuid(CustomTestCase):
    def test_uuid_returns_a_string(self):
        pm = PostGradPassManager()
        pm.configure()
        self.assertIsInstance(pm.uuid(), str)

    def test_uuid_is_stable(self):
        pm = PostGradPassManager()
        pm.configure()
        self.assertEqual(pm.uuid(), pm.uuid())

    def test_uuid_changes_when_pass_added(self):
        pm = PostGradPassManager()
        pm.configure()
        uuid_before = pm.uuid()
        pm.add(_AlwaysApplicablePass())
        uuid_after = pm.uuid()
        self.assertNotEqual(uuid_before, uuid_after)

    def test_uuid_returns_hex_string(self):
        pm = PostGradPassManager()
        pm.configure()
        h = pm.uuid()
        self.assertEqual(len(h), 64)
        int(h, 16)  # valid hex


if __name__ == "__main__":
    unittest.main()
