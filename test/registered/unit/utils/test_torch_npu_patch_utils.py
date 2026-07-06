import types
import unittest

from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestTorchNpuPatchUtils(unittest.TestCase):
    def test_apply_torch_npu_patches_uses_targeted_api_when_available(self):
        calls = []
        torch_npu = types.SimpleNamespace(
            _apply_patches=lambda patches: calls.append(("_apply_patches", patches)),
            _apply_all_patches=lambda: calls.append(("_apply_all_patches", None)),
        )
        patches = [["profiler.profile", object()]]

        apply_torch_npu_patches(torch_npu, patches)

        self.assertEqual(calls, [("_apply_patches", patches)])

    def test_apply_torch_npu_patches_uses_all_patches_api_when_targeted_api_missing(
        self,
    ):
        calls = []
        torch_npu = types.SimpleNamespace(
            _apply_all_patches=lambda: calls.append("_apply_all_patches")
        )

        apply_torch_npu_patches(torch_npu, [["profiler.profile", object()]])

        self.assertEqual(calls, ["_apply_all_patches"])

    def test_apply_torch_npu_patches_requires_supported_api(self):
        with self.assertRaises(AttributeError):
            apply_torch_npu_patches(types.SimpleNamespace(), [])


if __name__ == "__main__":
    unittest.main()
