# SPDX-License-Identifier: Apache-2.0
"""The torch_npu compatibility shim must apply the caller's replacement list."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def apply_requested_patches(patches):
    # Stand in for torch_npu's attribute replacement, without importing the NPU runtime.
    for destination, replacement in patches:
        parent = torch
        *parts, name = destination.split(".")
        for part in parts:
            parent = getattr(parent, part)
        setattr(parent, name, replacement)


class TestTorchNPUPatchUtils(CustomTestCase):
    def setUp(self):
        super().setUp()
        # Cover both Diffusion's functions and SRT's nested activity replacements.
        self.patches = [
            ["profiler.profile", object()],
            ["profiler.schedule", object()],
            ["profiler.ProfilerActivity.CUDA", object()],
            ["profiler.ProfilerActivity.CPU", object()],
        ]
        for destination, _ in self.patches:
            parent = torch
            *parts, name = destination.split(".")
            for part in parts:
                parent = getattr(parent, part)
            self.enterContext(patch.object(parent, name, getattr(parent, name)))

    def assert_replacements_applied(self):
        for destination, replacement in self.patches:
            with self.subTest(destination=destination):
                value = torch
                for part in destination.split("."):
                    value = getattr(value, part)
                self.assertIs(value, replacement)

    def new_api(self):
        apply = Mock(side_effect=apply_requested_patches)
        module = SimpleNamespace(
            _apply_all_patches=Mock(),
            _init=SimpleNamespace(
                patches=SimpleNamespace(
                    monkey_patches=SimpleNamespace(_apply_patches=apply)
                )
            ),
        )
        return module, apply

    def test_legacy_api_applies_requested_replacements(self):
        apply = Mock(side_effect=apply_requested_patches)
        apply_torch_npu_patches(SimpleNamespace(_apply_patches=apply), self.patches)
        self.assertIs(apply.call_args.args[0], self.patches)
        self.assert_replacements_applied()

    def test_relocated_api_applies_requested_replacements_not_builtin_patches(self):
        module, apply = self.new_api()
        # Multiple import sites may request the same replacements.
        for _ in range(2):
            apply_torch_npu_patches(module, self.patches)
            self.assert_replacements_applied()
        self.assertEqual(apply.call_count, 2)
        self.assertIs(apply.call_args.args[0], self.patches)
        module._apply_all_patches.assert_not_called()

    def test_relocated_api_does_not_require_apply_all_patches(self):
        module, apply = self.new_api()
        del module._apply_all_patches
        apply_torch_npu_patches(module, self.patches)
        apply.assert_called_once_with(self.patches)
        self.assert_replacements_applied()

    def test_apply_all_patches_alone_is_not_a_replacement_api(self):
        module = SimpleNamespace(_apply_all_patches=Mock())
        with self.assertRaisesRegex(AttributeError, "monkey_patches"):
            apply_torch_npu_patches(module, self.patches)
        module._apply_all_patches.assert_not_called()

    def test_legacy_entrypoint_takes_precedence_when_both_exist(self):
        module, relocated_apply = self.new_api()
        module._apply_patches = Mock(side_effect=apply_requested_patches)
        apply_torch_npu_patches(module, self.patches)
        self.assert_replacements_applied()
        relocated_apply.assert_not_called()
        module._apply_all_patches.assert_not_called()

    def test_replacement_failure_is_not_reported_as_missing_api(self):
        module, apply = self.new_api()
        apply.side_effect = AttributeError("invalid replacement destination")
        with self.assertRaisesRegex(AttributeError, "invalid replacement destination"):
            apply_torch_npu_patches(module, self.patches)
        module._apply_all_patches.assert_not_called()

    def test_unsupported_api_is_not_silently_accepted(self):
        original = torch.profiler.profile
        with self.assertRaisesRegex(AttributeError, "torch_npu must provide"):
            apply_torch_npu_patches(SimpleNamespace(), self.patches)
        self.assertIs(torch.profiler.profile, original)


if __name__ == "__main__":
    unittest.main()
