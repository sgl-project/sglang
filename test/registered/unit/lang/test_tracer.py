"""Unit tests for the frontend-language tracer."""

import unittest

import sglang as sgl
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.ir import SglGen
from sglang.lang.tracer import extract_prefix_by_tracing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1.0, suite="base-a-test-cpu")


@sgl.function
def _role_program(s, question):
    with s.user():
        s += "You are a helpful assistant. " * 8
        s += question
    s += sgl.gen("answer")


@sgl.function
def _var_scope_program(s, question):
    s += "Prefix "
    with s.var_scope("scoped"):
        s += question
    s += sgl.gen("answer")


class TestTracerContextManagers(CustomTestCase):
    def test_trace_role_context_manager(self):
        """Program.trace() must support role context managers such as `with s.user():`."""
        tracer = _role_program.trace()
        self.assertIsInstance(tracer.flatten_nodes()[-1], SglGen)

    def test_trace_var_scope_context_manager(self):
        """Program.trace() must record variables created by `with s.var_scope(...):`."""
        tracer = _var_scope_program.trace()
        self.assertIn("scoped", tracer.variables)
        self.assertIsInstance(tracer.flatten_nodes()[-1], SglGen)

    def test_extract_prefix_through_role_context_manager(self):
        """extract_prefix_by_tracing must extract constant text inside role context managers."""
        prefix = extract_prefix_by_tracing(_role_program, BaseBackend())
        self.assertIn("You are a helpful assistant. " * 8, prefix)


if __name__ == "__main__":
    unittest.main()
