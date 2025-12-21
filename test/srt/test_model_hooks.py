import argparse
import json
import unittest

import hooks_test_data as hooks_impl
import torch
import torch.nn as nn

from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

HOOK_MODULE = "hooks_test_data"
HOOK_FACTORY = f"{HOOK_MODULE}:dummy_hook_factory"


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(4, 2),
            nn.ReLU(),
        )
        self.outer = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            self.inner,
        )

    def forward(self, x):
        return self.outer(x)


class TestAttachHooks(CustomTestCase):
    """Tests for register_forward_hooks / resolve_callable integration."""

    def setUp(self):
        # Per-test list and token to reset the contextvar in the impl module
        self.hook_calls = []
        self._token = hooks_impl.set_recorder(self.hook_calls.append)

    def tearDown(self):
        hooks_impl.reset_recorder(self._token)

    def test_hook_is_attached(self):
        """Hook from a factory string is registered and fired."""
        hook_specs = [
            {
                "target_modules": ["outer.0", "outer.1"],
                "hook_factory": HOOK_FACTORY,
                "config": {"tag": "forward-ok"},
            },
            {
                "target_modules": ["inner.*"],
                "hook_factory": HOOK_FACTORY,
                "config": {"tag": "forward-ok"},
            },
        ]

        model = TinyModel()
        register_forward_hooks(model, hook_specs)

        x = torch.randn(3, 4)
        _ = model(x)

        self.assertEqual(
            len(self.hook_calls),
            4,
            "Forward hook was not called correct number of times",
        )
        tags = {call["tag"] for call in self.hook_calls}
        self.assertIn("forward-ok", tags)

    def test_no_matching_modules_does_not_crash(self):
        """Hook spec with no matching modules should not crash."""
        model = TinyModel()
        hook_specs = [
            {
                "name": "no_match",
                "target_modules": ["does_not_exist.*"],
                "hook_factory": HOOK_FACTORY,
                "config": {"tag": "unused"},
            }
        ]

        register_forward_hooks(model, hook_specs)

        x = torch.randn(3, 4)
        _ = model(x)

        self.assertEqual(len(self.hook_calls), 0)

    def test_cli_hooks_reach_model(self):
        """
        Ensure that when hooks are provided via CLI, they are parsed into
        ServerArgs, passed to register_forward_hooks, and actually
        run during a forward pass.
        """
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        hooks_spec = [
            {
                "name": "outer_and_inner_from_cli",
                "target_modules": ["outer.0", "outer.1", "inner.*"],
                "hook_factory": HOOK_FACTORY,
                "config": {"tag": "cli-hook"},
            }
        ]

        cli_args = [
            "--model-path",
            "Qwen/Qwen2-7B-Instruct",  # Dummy value; not used in this test
            "--forward-hooks",
            json.dumps(hooks_spec),
        ]

        args = parser.parse_args(cli_args)
        server_args = ServerArgs.from_cli_args(args)

        self.assertEqual(server_args.forward_hooks, hooks_spec)

        model = TinyModel()
        register_forward_hooks(model, server_args.forward_hooks)

        x = torch.randn(3, 4)
        _ = model(x)

        # We expect hooks on outer.0, outer.1, inner.0, inner.1  => 4 calls
        self.assertEqual(
            len(self.hook_calls),
            4,
            "CLI-configured hooks did not fire expected number of times",
        )

        tags = {call["tag"] for call in self.hook_calls}
        self.assertEqual(tags, {"cli-hook"})


if __name__ == "__main__":
    unittest.main()
