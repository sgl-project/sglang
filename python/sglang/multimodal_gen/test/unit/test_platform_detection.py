# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import platforms


class NVMLUnavailableError(Exception):
    pass


class TestCudaPlatformDetection(unittest.TestCase):
    def test_torch_fallback_excludes_hip(self):
        cases = (
            ("6.0", None),
            (
                None,
                "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
            ),
        )

        for hip_version, expected in cases:
            with (
                self.subTest(hip_version=hip_version),
                patch(
                    "sglang.multimodal_gen.utils.import_pynvml",
                    side_effect=NVMLUnavailableError,
                ),
                patch.object(platforms.os.path, "isfile", return_value=False),
                patch.object(platforms.os.path, "exists", return_value=False),
                patch.object(torch.version, "hip", hip_version, create=True),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
            ):
                self.assertEqual(platforms.cuda_platform_plugin(), expected)


if __name__ == "__main__":
    unittest.main()
