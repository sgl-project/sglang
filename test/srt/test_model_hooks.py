import argparse
import json

import torch
import torch.nn as nn

from sglang.srt.model_executor.hook_manager import register_hooks
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

HOOK_CALLS = []


def dummy_hook_factory(config):
    """Factory that returns a forward hook capturing a tag from config."""
    tag = config.get("tag", "default")

    def hook(module, inputs, output):
        HOOK_CALLS.append(
            {
                "module_type": type(module).__name__,
                "tag": tag,
                "shape": tuple(output.shape),
            }
        )
        return output

    return hook


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
    """Tests for ModelRunner.register_hooks / resolve_callable integration."""

    def setUp(self):
        HOOK_CALLS.clear()

    def test_hook_is_attached(self):
        """Hook from a factory string is registered and fired."""
        hook_specs = [
            {
                "target_modules": ["outer.0", "outer.1"],
                "hook_factory": "test_model_hooks:dummy_hook_factory",
                "config": {"tag": "forward-ok"},
            },
            {
                "target_modules": ["inner.*"],
                "hook_factory": "test_model_hooks:dummy_hook_factory",
                "config": {"tag": "forward-ok"},
            },
        ]

        model = TinyModel()
        register_hooks(model, hook_specs)

        x = torch.randn(3, 4)
        _ = model(x)

        self.assertEqual(
            len(HOOK_CALLS),
            4,
            "Forward hook was not called correct number of times",
        )
        tags = {call["tag"] for call in HOOK_CALLS}
        self.assertIn("forward-ok", tags)

    def test_no_matching_modules_does_not_crash(self):
        """Hook spec with no matching modules should not crash."""
        model = TinyModel()
        hook_specs = [
            {
                "name": "no_match",
                "target_modules": ["does_not_exist.*"],
                "hook_factory": "test_model_hooks:dummy_hook_factory",
                "config": {"tag": "unused"},
            }
        ]

        register_hooks(model, hook_specs)

        x = torch.randn(3, 4)
        _ = model(x)

        # No hooks should have fired
        self.assertEqual(len(HOOK_CALLS), 0)

    def test_cli_hooks_reach_model(self):
        """
        Ensure that when hooks are provided via CLI, they are parsed into
        ServerArgs, passed to ModelRunner.register_hooks, and actually
        run during a forward pass.
        """
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        hooks_spec = [
            {
                "name": "outer_and_inner_from_cli",
                "target_modules": ["outer.0", "outer.1", "inner.*"],
                "hook_factory": "test_model_hooks:dummy_hook_factory",
                "config": {"tag": "cli-hook"},
            }
        ]

        cli_args = [
            "--model-path",
            "Qwen/Qwen2-7B-Instruct",  # Dummy value; not used in this test
            "--hooks",
            json.dumps(hooks_spec),
        ]

        args = parser.parse_args(cli_args)
        server_args = ServerArgs.from_cli_args(args)

        self.assertEqual(server_args.hooks, hooks_spec)

        model = TinyModel()
        register_hooks(model, server_args.hooks)

        x = torch.randn(3, 4)
        _ = model(x)

        # We expect hooks on outer.0, outer.1, inner.0, inner.1  => 4 calls
        self.assertEqual(
            len(HOOK_CALLS),
            4,
            "CLI-configured hooks did not fire expected number of times",
        )

        tags = {call["tag"] for call in HOOK_CALLS}
        self.assertEqual(tags, {"cli-hook"})


if __name__ == "__main__":
    pass
    # unittest.main()
