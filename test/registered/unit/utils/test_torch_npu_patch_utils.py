# SPDX-License-Identifier: Apache-2.0
"""The compatibility shim forwards the requested profiler replacements."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTorchNPUPatchUtils(unittest.TestCase):
    def test_legacy_and_relocated_apis_receive_requested_patches(self):
        patches = [["profiler.profile", object()], ["profiler.schedule", object()]]
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                relocated = Mock()
                module = SimpleNamespace(
                    _apply_all_patches=Mock(),
                    _init=SimpleNamespace(
                        patches=SimpleNamespace(
                            monkey_patches=SimpleNamespace(_apply_patches=relocated)
                        )
                    ),
                )
                if legacy:
                    module._apply_patches = Mock()
                apply_torch_npu_patches(module, patches)
                selected = module._apply_patches if legacy else relocated
                selected.assert_called_once_with(patches)
                if legacy:
                    relocated.assert_not_called()
                module._apply_all_patches.assert_not_called()

    def test_unsupported_api_is_not_silently_accepted(self):
        for module in (SimpleNamespace(), SimpleNamespace(_apply_all_patches=Mock())):
            with self.subTest(module=module):
                with self.assertRaisesRegex(AttributeError, "torch_npu must provide"):
                    apply_torch_npu_patches(module, [])

    def test_replacement_failure_is_not_reported_as_missing_api(self):
        module = SimpleNamespace(
            _apply_patches=Mock(side_effect=AttributeError("invalid destination"))
        )
        with self.assertRaisesRegex(AttributeError, "invalid destination"):
            apply_torch_npu_patches(module, [])


if __name__ == "__main__":
    unittest.main()
