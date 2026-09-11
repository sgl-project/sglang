"""Unit tests for the out-of-tree resolution-hook registry: whitelist
enforcement, wrap-vs-replace semantics, multi-registrant composition, and the
end-to-end proof that an override reaches a real `resolve_once()`."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ast
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups import pipeline as pipeline_module
from sglang.srt.arg_groups import resolution_hooks as hooks_module
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.resolution_hooks import (
    _OVERRIDABLE_HOOKS,
    register_resolution_hook,
    run_hook,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

# `model_path="dummy"` short-circuits the pipeline before this step ever
# runs (the fixture the rest of this file uses on purpose, since it does not
# need the step to have run). The end-to-end proof below needs the real
# pipeline, so it needs a real, tiny HF config on disk instead.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class _IsolatedRegistry(CustomTestCase):
    """Run each test against an empty registry (it is process-global)."""

    def setUp(self):
        super().setUp()
        self._patch = patch.dict(hooks_module._HOOKS, clear=True)
        self._patch.start()
        self.addCleanup(self._patch.stop)


class TestWhitelist(_IsolatedRegistry):
    def test_an_unlisted_name_is_refused_at_registration(self):
        with self.assertRaisesRegex(ValueError, "not an overridable resolution hook"):

            @register_resolution_hook("handle_something_nobody_whitelisted")
            def _fn(server_args, previous):
                pass

    def test_the_whitelisted_name_registers(self):
        @register_resolution_hook("handle_cuda_graph_config")
        def _fn(server_args, previous):
            pass

        self.assertIn(_fn, hooks_module._HOOKS["handle_cuda_graph_config"])


class TestWhitelistMatchesThePipeline(CustomTestCase):
    """The whitelist and `pipeline.py` are two hand-written lists that have to
    agree; nothing keeps them in sync on its own. This derives both sides
    fresh and asserts they are the same set, so a name added to one without
    the other fails here instead of surfacing as "my override never runs" or
    "this step nobody can ever replace" months later."""

    def _run_hook_call_targets(self) -> set:
        """The first argument of every `run_hook(...)` call in pipeline.py,
        statically -- the set of steps the pipeline actually dispatches
        through the registry."""
        tree = ast.parse(open(pipeline_module.__file__).read())
        names = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_hook"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                names.add(node.args[0].id)
        return names

    def test_every_whitelisted_hook_has_a_call_site(self):
        called = self._run_hook_call_targets()
        self.assertEqual(
            (sorted(_OVERRIDABLE_HOOKS - called), sorted(called - _OVERRIDABLE_HOOKS)),
            ([], []),
            "(whitelisted but never called, called but not whitelisted) -- "
            "both should be empty",
        )


class TestRunHook(_IsolatedRegistry):
    """``run_hook`` reads the name off the function it is handed
    (``builtin.__name__``), so every ``builtin`` stand-in below that needs to
    line up with a registration is a real named function called
    ``handle_cuda_graph_config`` -- a lambda's ``__name__`` is ``'<lambda>'``
    and would silently miss the registry entirely."""

    def test_nothing_registered_runs_the_builtin_directly(self):
        calls = []
        run_hook(calls.append, "sa")
        self.assertEqual(calls, ["sa"])

    def test_an_override_that_calls_previous_wraps_the_builtin(self):
        order = []

        @register_resolution_hook("handle_cuda_graph_config")
        def _wraps(server_args, previous):
            order.append(("before", server_args))
            previous(server_args)
            order.append(("after", server_args))

        def handle_cuda_graph_config(server_args):
            order.append(("builtin", server_args))

        run_hook(handle_cuda_graph_config, "sa")
        self.assertEqual(
            order,
            [("before", "sa"), ("builtin", "sa"), ("after", "sa")],
        )

    def test_an_override_that_never_calls_previous_replaces_the_builtin(self):
        builtin_ran = []

        @register_resolution_hook("handle_cuda_graph_config")
        def _replaces(server_args, previous):
            pass  # deliberately does not call `previous`

        def handle_cuda_graph_config(server_args):
            builtin_ran.append(server_args)

        run_hook(handle_cuda_graph_config, "sa")
        self.assertEqual(builtin_ran, [], "the replaced builtin must not have run")

    def test_two_registrants_compose_last_registered_outermost(self):
        order = []

        @register_resolution_hook("handle_cuda_graph_config")
        def _first(server_args, previous):
            order.append("first-before")
            previous(server_args)
            order.append("first-after")

        @register_resolution_hook("handle_cuda_graph_config")
        def _second(server_args, previous):
            order.append("second-before")
            previous(server_args)
            order.append("second-after")

        def handle_cuda_graph_config(server_args):
            order.append("builtin")

        run_hook(handle_cuda_graph_config, "sa")
        self.assertEqual(
            order,
            [
                "second-before",  # last registered runs first (outermost)
                "first-before",
                "builtin",
                "first-after",
                "second-after",
            ],
        )


class TestEndToEnd(_IsolatedRegistry):
    """The proof that matters: a real `resolve_once()` picks up the override,
    at the same position the built-in occupied, without disturbing the
    neighboring steps documented at that call site."""

    def setUp(self):
        super().setUp()
        self._config_dir = tempfile.mkdtemp(prefix="resolution_hook_registry_")
        self.addCleanup(shutil.rmtree, self._config_dir, ignore_errors=True)
        with open(os.path.join(self._config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)

    def test_an_override_reaches_resolution_result(self):
        @register_resolution_hook("handle_cuda_graph_config")
        def _mark_it(server_args, previous):
            previous(server_args)
            from sglang.srt.arg_groups.overrides import declare_resolution

            declare_resolution(server_args, "test_plugin", random_seed=999)
            server_args._test_plugin_ran = True

        sa = ServerArgs(model_path=self._config_dir, device="cuda")
        sa.resolve_once()
        self.assertTrue(getattr(sa, "_test_plugin_ran", False))
        self.assertEqual(resolution_result(sa, "random_seed"), 999)
        # And the neighboring step (must run right after, per the comment at
        # the call site) still ran and still saw a real config to chunk.
        self.assertIsNotNone(
            resolution_result(sa, "chunked_prefill_size"),
            "apply_glm5_chunked_prefill_default's neighbor did not run",
        )

    def test_with_nothing_registered_resolution_is_unchanged(self):
        baseline = ServerArgs(model_path=self._config_dir, device="cuda")
        baseline.resolve_once()
        sa = ServerArgs(model_path=self._config_dir, device="cuda")
        sa.resolve_once()
        self.assertEqual(
            resolution_result(sa, "chunked_prefill_size"),
            resolution_result(baseline, "chunked_prefill_size"),
        )
        self.assertEqual(
            resolution_result(sa, "cuda_graph_config"),
            resolution_result(baseline, "cuda_graph_config"),
        )


if __name__ == "__main__":
    unittest.main()
