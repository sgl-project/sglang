import unittest

import torch
import torch.nn as nn

from sglang.srt.model_executor import model_runner as mr
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
        return self.mlp(x)


class DummyModelRunner:
    """
    Minimal stand-in for ModelRunner that only exposes `model` and
    reuses the real ModelRunner.register_hooks implementation.
    """

    def __init__(self):
        self.model = TinyModel()

    # Reuse the ModelRunner.register_hooks method as an unbound function
    register_hooks = mr.ModelRunner.register_hooks


class TestModelRunnerHooks(CustomTestCase):
    """Tests for ModelRunner.register_hooks / resolve_callable integration."""

    def setUp(self):
        HOOK_CALLS.clear()

    def test_hook_is_called(self):
        """Hook from a factory string is registered and fired."""
        runner = DummyModelRunner()
        hook_specs = [
            {
                # TinyModel.named_modules() includes: "", "mlp", "mlp.0", "mlp.1", "mlp.2"
                "target_modules": ["outer.0", "outer.1"],
                # IMPORTANT: this path assumes the file lives at
                # sglang/test/test_model_runner_hooks.py
                "hook_factory": ("sglang.test.srt.test_model_hooks:dummy_hook_factory"),
                "config": {"tag": "forward-ok"},
            },
            {
                "target_modules": ["outer.inner.*"],
                "hook_factory": ("sglang.test.set.test_model_hooks:dummy_hook_factory"),
                "config": {"tag": "forward-ok"},
            },
        ]

        runner.register_hooks(hook_specs)

        x = torch.randn(3, 4)
        _ = runner.model(x)

        self.assertEqual(
            len(HOOK_CALLS),
            4,
            "Forward hook was not called correct number of times",
        )

        tags = {call["tag"] for call in HOOK_CALLS}
        self.assertIn("forward-ok", tags)

    def test_no_matching_modules_does_not_crash(self):
        """Hook spec with no matching modules should not crash."""
        runner = DummyModelRunner()

        hook_specs = [
            {
                "name": "no_match",
                "target_modules": ["does_not_exist.*"],
                "hook_factory": ("sglang.test.test_model_hooks:dummy_hook_factory"),
                "config": {"tag": "unused"},
            }
        ]

        runner.register_hooks(hook_specs)

        x = torch.randn(3, 4)
        _ = runner.model(x)

        # No hooks should have fired
        self.assertEqual(len(HOOK_CALLS), 0)


if __name__ == "__main__":
    unittest.main()
