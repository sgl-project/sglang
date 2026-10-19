import unittest

import pybase64
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOutputSerialization(unittest.TestCase):
    def test_base64_preserves_bytes_and_missing_items(self):
        tensor = torch.arange(120, dtype=torch.int32).reshape(5, 4, 6)
        items = [tensor, tensor.transpose(0, 1), torch.empty((0, 2)), None]
        expected = [
            pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
            if item is not None
            else None
            for item in items
        ]
        encode = DetokenizerManager._b64_encode_per_request
        self.assertEqual(encode(items), expected)
        self.assertIsNone(encode(None))
        self.assertEqual(encode([]), [])


if __name__ == "__main__":
    unittest.main()
