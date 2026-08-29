"""End-to-end parity at the scheduler-input boundary.

`test_preprocess.py` pins the `preprocess` binding; this drives the whole native
path — the `process_native_mm` driver, then `NativeMmHost.build_native_mm` — and
compares every field the scheduler reads against the Python `mm_processor`.
Bitwise, for both HF backends: the Rust resize clones PIL's fixed-point bicubic
and ATen's uint8 antialias kernel, so whichever one a server is configured with
is reproduced exactly.
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

# The fixture tokenizer's vocab (see `_fixtures.make_processor`):
# 1 = <|vision_start|>, 2 = <|image_pad|>, 3 = <|vision_end|>, 4 = "hello".
PROMPT_PER_IMAGE = [1, 2, 3, 4]


@unittest.skipUnless(DRIVER, "sglang-mm native Qwen driver not built")
class TestQwenE2eParity(CustomTestCase):
    """The torchvision backend a default server runs."""

    image_processor = "Qwen2VLImageProcessor"

    def setUp(self):
        import transformers.models.qwen2_vl as qwen2_vl

        self.processor = make_processor(
            self,
            PROCESSOR_CONFIGS["qwen2_5_vl"],
            getattr(qwen2_vl, self.image_processor),
        )

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    # --- the two paths under comparison ---

    def native_spec(self):
        """Resolve the spec through the production gate, so a gate that stops
        recognizing this image processor fails here too."""
        from sglang.srt.managers.multimodal_processor import import_processors

        import_processors("sglang.srt.multimodal.processors")
        # Skip __init__: it would build a processor; reuse the fixture's.
        host = NativeMmHost.__new__(NativeMmHost)
        host.model_config = SimpleNamespace(hf_config=self.processor.hf_config)
        host._processor = self.processor._processor
        host.server_args = self.processor.server_args
        spec = host.resolve_native_spec()
        self.assertIsNotNone(spec, f"gate rejected {self.image_processor}")
        return spec

    def run_native(self, spec, sources):
        """The Rust path: the `process_native_mm` driver, then the drain
        adapter — the same two steps `RustServer.drain` performs."""
        ids, features, grids, hashes, offsets, mrope, delta = DRIVER(
            PROMPT_PER_IMAGE * len(sources), sources, spec.rust_json()
        )
        # The shape of Rust's MmEncodeResult, inline transport (test_build_native_mm
        # pins the shm shape).
        handoff = SimpleNamespace(
            features=features,
            shm_names=None,
            grids=grids,
            hashes=hashes,
            offsets=offsets,
            mrope=mrope,
            mrope_delta=delta,
        )
        return snapshot(ids, NativeMmHost.build_native_mm(spec, handoff))

    def run_python(self, sources):
        """The reference path: the Python `mm_processor` the scheduler would use."""
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=sources,
                input_text=PROMPT_PER_IMAGE * len(sources),
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="parity"
                ),
            )
        )
        return snapshot(output.input_ids, output)

    def assert_parity(self, spec, sources):
        rust, python = self.run_native(spec, sources), self.run_python(sources)
        for field in ("input_ids", "grids", "offsets", "delta", "tokens"):
            with self.subTest(field=field):
                self.assertEqual(rust[field], python[field])
        with self.subTest(field="mrope"):
            np.testing.assert_array_equal(rust["mrope"], python["mrope"])
        with self.subTest(field="features"):
            # Bytes, not allclose: the scheduler gets these float32 buffers verbatim.
            self.assertEqual(rust["features"].tobytes(), python["features"].tobytes())

    # --- inputs ---

    def source_forms(self, directory):
        """One image in each accepted transport form, plus a two-image batch."""
        first, second = image_bytes(96, 80), image_bytes(112, 88, 1)
        path = Path(directory) / "image.png"
        path.write_bytes(first)
        return {
            "raw_bytes": [first],
            "data_url": ["data:image/png;base64," + base64.b64encode(first).decode()],
            "file_uri": [path.as_uri()],
            "two_image_batch": [first, second],
        }

    def test_parity_across_source_forms(self):
        spec = self.native_spec()
        with tempfile.TemporaryDirectory() as directory:
            for form, sources in self.source_forms(directory).items():
                with self.subTest(form=form):
                    self.assert_parity(spec, sources)


class TestQwenE2eParityPil(TestQwenE2eParity):
    """The PIL backend (`--disable-fast-image-processor`): the same parity
    suite, plus the transport-invariance check — a property of the native path
    alone, so running it under one backend is enough."""

    image_processor = "Qwen2VLImageProcessorPil"

    def test_source_form_is_transport_only(self):
        """Bytes, a data: URL and a file:// path must yield one identical result."""
        spec = self.native_spec()
        with tempfile.TemporaryDirectory() as directory:
            forms = self.source_forms(directory)
            features = {
                form: self.run_native(spec, forms[form])["features"].tobytes()
                for form in ("raw_bytes", "data_url", "file_uri")
            }
            self.assertEqual(len(set(features.values())), 1, sorted(features))


if __name__ == "__main__":
    unittest.main()
