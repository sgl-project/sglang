"""The mm-item hash contract of the Python Qwen path.

Python items bind the model-facing tensor, its metadata and the processor
configuration before transport.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.cache.identity import (  # noqa: E402
    resolve_multimodal_item_identity,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _fixtures import make_processor  # noqa: E402
from _mm_rust_utils import PROCESSOR_CONFIGS, image_bytes  # noqa: E402

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


class TestQwenRustMmHashes(CustomTestCase):
    def setUp(self):
        self.processor = make_processor(self, PROCESSOR_CONFIGS["qwen2_5_vl"])

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def test_python_hashes_feature_and_processor_metadata(self):
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=[image_bytes(96, 80)],
                input_text=[1, 2, 3, 4],
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="hash"
                ),
            )
        )
        for item in output.mm_items:
            expected = resolve_multimodal_item_identity(
                feature=item.feature,
                modality=item.modality.name,
                input_format=item.format.name,
                namespace=item.processor_fingerprint,
                metadata=item.model_specific_data,
            )
            item.set_pad_value()
            self.assertEqual(item.cache_key, expected)
            self.assertEqual(len(item.cache_key), 32)


if __name__ == "__main__":
    unittest.main()
