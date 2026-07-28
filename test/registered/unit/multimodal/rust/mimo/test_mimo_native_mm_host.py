"""``NativeMmHost`` (managers/rust_server.py) parity for the MiMo-V2 family:
the spec built by ``resolve_native_spec()`` drives the real native Rust driver
and ``build_native_mm()`` wraps its buffers; the result must be
indistinguishable from the Python ``MiMoV2Processor`` output at the
scheduler-input boundary — including the 1-D-rope position contract
(``[3, len]`` arange, delta 0)."""

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
from sglang.srt.managers.rust_server import NativeMmHost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mimo_fixtures import make_processor, snapshot  # noqa: E402
from _mm_rust_utils import image_bytes, load_core  # noqa: E402

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

CORE = load_core()
DRIVER = getattr(getattr(CORE, "mimo_v2", None), "process_native_mm", None)

# `<|vision_start|><|image_pad|><|vision_end|>hello` in the fixture vocab —
# the chat template's per-image block followed by text.
IMAGE_BLOCK_IDS = (1, 2, 3, 4)


@unittest.skipUnless(DRIVER, "sglang-mm native MiMo driver not built")
class TestMimoNativeMmHost(CustomTestCase):
    def setUp(self):
        self.processor = make_processor()

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def make_host(self):
        """A drain-adapter host whose spec comes from ``resolve_native_spec()`` — the
        production extraction path — rather than a hand-built dict."""
        host = NativeMmHost.__new__(NativeMmHost)
        host.model_config = SimpleNamespace(hf_config=self.processor.hf_config)
        host.mm_processor = self.processor
        host._processor = self.processor._processor
        host.server_args = self.processor.server_args
        host._native = None
        return host, host.resolve_native_spec()

    def compare(self, sources):
        host, spec = self.make_host()
        self.assertIsNotNone(
            spec, "resolve_native_spec() rejected the fixture processor"
        )
        self.assertIn('"family": "mimo_v2"', spec)

        input_ids = []
        for _ in sources:
            input_ids.extend(IMAGE_BLOCK_IDS)
        raw = DRIVER(input_ids, sources, spec)
        ids, features, grids, hashes, offsets, mrope, delta = raw

        rust_output = host.build_native_mm(
            # Inline entry shape (single-rank; `shm_names=None`). The shm
            # shape's wrapping contract is pinned by test_build_native_mm.
            (features, None, grids, hashes, offsets, mrope, delta)
        )
        request = SimpleNamespace(video_data=None, audio_data=None, rid="parity")
        python_output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=sources,
                audio_data=None,
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
        # Both paths resize and normalize in f32 (no u8 re-quantization), so
        # the envelope only covers accumulation-order noise in the bilinear
        # kernels — orders of magnitude below the ±2.2 feature scale.
        diff = np.abs(rust["features"] - python["features"])
        self.assertLess(diff.max(), 5e-3)
        self.assertLess(diff.mean(), 1e-4)

        # The native hashes are the drain-time identity (set_pad_value skips
        # hash_feature); each must cover exactly its item's feature bytes.
        row = 0
        for grid, expected_hash in zip(grids, hashes):
            rows = int(np.prod(grid))
            native_bytes = np.ascontiguousarray(
                rust["features"][row : row + rows]
            ).tobytes()
            self.assertEqual(expected_hash, CORE.common.content_hash(native_bytes))
            row += rows
        for python_item in python_output.mm_items:
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
