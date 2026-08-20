"""The plugin hook takes the call the dispatch makes.

`CustomSpecAlgo` is the out-of-tree extension point: a registered algorithm's
method is called through the same dispatch as the built-in ones, and nothing in
the tree implements it, so a drift between the two sides only ever surfaces in
somebody's plugin. It has drifted twice, both times on the disaggregation
draft-input builder: the built-in dropped a parameter the hook kept, so every
plugin call would have hit a TypeError.

What is pinned here:

  * every method the dispatch may call on either type takes the same arguments
    on both -- the set is intersected out of the two types, never listed;
  * the call the dispatch actually writes binds on both types.
"""

import ast
import inspect
import unittest
from pathlib import Path

from sglang.srt.disaggregation import decode_schedule_batch_mixin
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dispatched_methods():
    """Methods carried by both types. The dispatch calls them on whichever it
    holds without knowing which, so their argument lists must agree."""
    enum_methods = {
        name
        for name, value in vars(SpeculativeAlgorithm).items()
        if inspect.isfunction(value)
    }
    hook_methods = {
        name
        for name, value in vars(CustomSpecAlgo).items()
        if inspect.isfunction(value)
    }
    return sorted(enum_methods & hook_methods)


def _dispatch_calls():
    """Every call the decode dispatch makes on the algorithm object, read from
    its source: `(method, positional count, keyword names)`."""
    source = Path(decode_schedule_batch_mixin.__file__).read_text(encoding="utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "spec_algorithm"
        ):
            continue
        calls.append((func.attr, len(node.args), tuple(kw.arg for kw in node.keywords)))
    return calls


def _parameters(function):
    """Parameter names, without any catch-alls."""
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


class TestDispatchedSignatures(CustomTestCase):
    def test_the_hook_and_the_built_in_agree(self):
        methods = _dispatched_methods()
        self.assertNotEqual(methods, [], "the dispatched set derived to nothing")
        mismatches = []
        for name in methods:
            hook = _parameters(getattr(CustomSpecAlgo, name))
            builtin = _parameters(getattr(SpeculativeAlgorithm, name))
            if hook != builtin:
                mismatches.append(
                    f"{name}: CustomSpecAlgo{tuple(hook)} vs "
                    f"SpeculativeAlgorithm{tuple(builtin)}"
                )
        self.assertEqual(
            mismatches,
            [],
            "a plugin implementing the hook would be called with the "
            "dispatch's arguments:\n  " + "\n  ".join(mismatches),
        )

    def test_the_dispatch_call_binds_on_both_types(self):
        calls = _dispatch_calls()
        self.assertNotEqual(calls, [], "no dispatch call found to bind against")
        for method, positional, keywords in calls:
            self.assertIn(method, _dispatched_methods())
            for owner in (CustomSpecAlgo, SpeculativeAlgorithm):
                arguments = [None] * (1 + positional)
                inspect.signature(getattr(owner, method)).bind(
                    *arguments, **{name: None for name in keywords}
                )


if __name__ == "__main__":
    unittest.main()
