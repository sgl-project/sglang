"""End-to-end parity at the scheduler-input boundary.

`test_preprocess.py` pins the `preprocess` binding; this covers the whole path a
request takes — `process_native_mm` then `build_native_mm` — and compares every
field the scheduler reads. Both HF backends are asserted bitwise: the Rust resize
clones PIL's fixed-point bicubic and ATen's uint8 antialias kernel, so whichever
one a server is configured with is reproduced exactly.
"""

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

from sglang.srt.managers.rust_server import NativeMmHost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _fixtures import make_processor, snapshot  # noqa: E402
from _mm_rust_utils import PROCESSOR_CONFIGS, image_bytes, load_core  # noqa: E402

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

CORE = load_core()
DRIVER = getattr(getattr(CORE, "qwen_vl", None), "process_native_mm", None)

EQUAL_FIELDS = ("input_ids", "grids", "offsets", "delta", "tokens")


@unittest.skipUnless(DRIVER, "sglang-mm native Qwen driver not built")
class TestQwenE2eParity(CustomTestCase):
    """The PIL backend (`--disable-fast-image-processor`)."""

    image_processor = "Qwen2VLImageProcessorPil"

    def setUp(self):
        import transformers.models.qwen2_vl as qwen2_vl

        self.processor = make_processor(
            PROCESSOR_CONFIGS["qwen2_5_vl"], getattr(qwen2_vl, self.image_processor)
        )

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def native_spec(self):
        """Via the production gate, so a gate that stops recognizing this
        processor fails here too."""
        from sglang.srt.managers.multimodal_processor import import_processors

        import_processors("sglang.srt.multimodal.processors")
        host = NativeMmHost.__new__(NativeMmHost)
        host.model_config = SimpleNamespace(hf_config=self.processor.hf_config)
        host._processor = self.processor._processor
        host.server_args = self.processor.server_args
        spec = host.resolve_native_spec()
        self.assertIsNotNone(spec, f"gate rejected {self.image_processor}")
        return spec

    def both_paths(self, spec, sources):
        input_ids = [t for _ in sources for t in (1, 2, 3, 4)]
        ids, features, grids, hashes, offsets, mrope, delta = DRIVER(
            input_ids, sources, spec.rust_json()
        )
        # Inline handoff; test_build_native_mm pins the shm shape.
        native = NativeMmHost.build_native_mm(
            spec,
            SimpleNamespace(
                features=features,
                shm_names=None,
                grids=grids,
                hashes=hashes,
                offsets=offsets,
                mrope=mrope,
                mrope_delta=delta,
            ),
        )
        python = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=sources,
                input_text=input_ids,
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="parity"
                ),
            )
        )
        return snapshot(ids, native), snapshot(python.input_ids, python)

    def assert_parity(self, sources):
        rust, python = self.both_paths(self.native_spec(), sources)
        for field in EQUAL_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(rust[field], python[field])
        with self.subTest(field="mrope"):
            np.testing.assert_array_equal(rust["mrope"], python["mrope"])
        with self.subTest(field="features"):
            # Bytes, not allclose: the scheduler gets these float32 buffers verbatim.
            self.assertEqual(rust["features"].tobytes(), python["features"].tobytes())

    def sources_of(self, directory):
        """One image in each accepted form, plus a two-image batch."""
        first, second = image_bytes(96, 80), image_bytes(112, 88, 1)
        path = Path(directory) / "image.png"
        path.write_bytes(first)
        data_url = "data:image/png;base64," + base64.b64encode(first).decode()
        return [first], [data_url], [path.as_uri()], [first, second]

    def test_parity_across_source_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            for sources in self.sources_of(directory):
                with self.subTest(n=len(sources), form=type(sources[0]).__name__):
                    self.assert_parity(sources)

    def test_source_form_is_transport_only(self):
        """Bytes, a data: URL and a file:// path must give one identical result."""
        spec = self.native_spec()
        with tempfile.TemporaryDirectory() as directory:
            forms = self.sources_of(directory)[:3]
            expected = self.both_paths(spec, forms[0])[0]["features"].tobytes()
            for sources in forms[1:]:
                actual = self.both_paths(spec, sources)[0]["features"].tobytes()
                self.assertEqual(actual, expected, f"{sources[0]!r:.40} disagreed")


class TestQwenE2eParityTorchvision(TestQwenE2eParity):
    """The backend a default server runs."""

    image_processor = "Qwen2VLImageProcessor"

    # A property of the native path alone; running it once is enough.
    test_source_form_is_transport_only = None


if __name__ == "__main__":
    unittest.main()
