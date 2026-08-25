import os
import sys
import types
import unittest
from pathlib import Path


def install_namespace_packages() -> None:
    workspace = (
        Path(os.environ["TEST_SRCDIR"])
        / os.environ["TEST_WORKSPACE"]
        / "python/sglang"
    )
    sglang = types.ModuleType("sglang")
    sglang.__path__ = [str(workspace)]
    kernels = types.ModuleType("sglang.kernels")
    kernels.__path__ = [str(workspace / "kernels")]
    sys.modules["sglang"] = sglang
    sys.modules["sglang.kernels"] = kernels


class KernelMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_namespace_packages()

    def test_cpu_metadata_selects_without_torch(self) -> None:
        sys.modules.pop("torch", None)
        from sglang.kernels.registry import register_kernel
        from sglang.kernels.selector import select_kernel
        from sglang.kernels.spec import (
            CapabilityRequirement,
            DeviceType,
            KernelBackend,
            KernelSpec,
            PlatformInfo,
        )

        spec = KernelSpec(
            op="bazel.identity",
            backend=KernelBackend.TORCH,
            target="builtins:id",
            capabilities=frozenset(
                [CapabilityRequirement(device=DeviceType.CPU)]
            ),
        )
        register_kernel(spec)

        self.assertEqual(select_kernel("bazel.identity"), spec)
        self.assertTrue(spec.is_available(PlatformInfo()))
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
