"""Full native Qwen driver parity at the scheduler-input boundary."""

import asyncio
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.mm_utils import hash_feature  # noqa: E402
from sglang.srt.managers.rust_server import MmProcessorHost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _fixtures import make_processor, snapshot  # noqa: E402
from _utils import (
    PROCESSOR_CONFIGS,
    image_bytes,
    request_payload,
    spec_json,
)  # noqa: E402

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal import _core

    DRIVER = _core.qwen_vl.process_native_mm_payload
except (AttributeError, ImportError):
    DRIVER = None


@unittest.skipUnless(DRIVER, "sglang-mm native Qwen driver not built")
class TestQwenSchedulerInputParity(CustomTestCase):
    def setUp(self):
        self.config = PROCESSOR_CONFIGS["qwen2_5_vl"]
        self.processor = make_processor(self.config)

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def compare(self, sources):
        input_ids = []
        for _ in sources:
            input_ids.extend((1, 2, 3, 4))
        raw = DRIVER(
            request_payload(input_ids, sources),
            spec_json(self.config, image_token_id=2),
        )
        ids, features, grids, hashes, offsets, mrope, delta = raw

        host = MmProcessorHost.__new__(MmProcessorHost)
        host._native = {
            **self.config,
            "feature_dim": 3
            * self.config["temporal_patch_size"]
            * self.config["patch_size"] ** 2,
            "image_token_id": 2,
            "vision_start_token_id": 1,
            "vision_end_token_id": 3,
            "video_token_id": 5,
        }
        rust_output = host.build_native_mm(
            (features, grids, hashes, offsets, mrope, delta)
        )
        request = SimpleNamespace(video_data=None, audio_data=None, rid="parity")
        python_output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=sources,
                input_text=input_ids,
                request_obj=request,
            )
        )

        rust = snapshot(ids, rust_output)
        python = snapshot(python_output.input_ids, python_output)
        for key in ("input_ids", "grids", "offsets", "mrope", "delta", "tokens"):
            with self.subTest(field=key):
                if isinstance(rust[key], np.ndarray):
                    np.testing.assert_array_equal(rust[key], python[key])
                else:
                    self.assertEqual(rust[key], python[key])
        diff = np.abs(rust["features"] - python["features"])
        self.assertLess(diff.max(), 0.06)
        self.assertLess(diff.mean(), 1e-3)

        row = 0
        for python_item, grid, expected_hash in zip(
            python_output.mm_items, grids, hashes
        ):
            rows = int(np.prod(grid))
            native_bytes = np.ascontiguousarray(
                rust["features"][row : row + rows]
            ).tobytes()
            self.assertEqual(expected_hash, _core.common.data_hash(native_bytes))
            row += rows
            expected_python_hash = hash_feature(python_item.feature)
            python_item.set_pad_value()
            self.assertEqual(python_item.hash, expected_python_hash)

    def test_bytes_data_url_file_and_multiple_images(self):
        first, second = image_bytes(96, 80), image_bytes(112, 88, 1)
        data_url = "data:image/png;base64," + base64.b64encode(first).decode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(first)
            cases = ([first], [data_url], [path.as_uri()], [first, second])
            for sources in cases:
                with self.subTest(
                    source_count=len(sources), type=type(sources[0]).__name__
                ):
                    self.compare(sources)


if __name__ == "__main__":
    unittest.main()
