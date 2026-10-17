"""End-to-end parity at the scheduler-input boundary.

``test_internvl_parity.py`` pins the tile pixels ``preprocess`` returns; this
drives the whole native path -- the ``process_mm`` driver, then
``RustMmProcessor.wrap_encoded`` -- and compares ``input_ids``, per-item
offsets and ``mm_items`` feature tensors against the Python
``InternVLProcessor``. Features compare with ``assert_allclose`` rather than
byte-equal because Python returns bf16 tiles while the native pipeline carries
f32; the vision encoder casts to its working dtype either way.
"""

import os

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import asyncio
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.rust_server.multimodal import RustMmProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _fixtures import IMAGE_SIZE, IMG_CONTEXT_ID, TEXT_ID, make_processor, snapshot
from _mm_rust_utils import image_bytes, load_core

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

CORE = load_core()
DRIVER = getattr(getattr(CORE, "internvl", None), "process_mm", None)


def rust_prompt(images):
    """The tokenized prompt a collapsed ``<image>`` slot leaves for the worker."""
    return [TEXT_ID] + [IMG_CONTEXT_ID] * images


@unittest.skipUnless(DRIVER, "sglang-mm native InternVL driver not built")
class TestInternVlE2eParity(CustomTestCase):
    def setUp(self):
        self.processor = make_processor(self)

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def native_spec(self):
        from sglang.srt.managers.multimodal_processor import import_processors

        import_processors("sglang.srt.multimodal.processors")
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(hf_config=self.processor.hf_config)
        host._processor = self.processor._processor
        host.server_args = self.processor.server_args
        spec = host.resolve_spec()
        self.assertIsNotNone(spec, "gate rejected the InternVL processor")
        return spec

    def run_native(self, spec, sources):
        ids, features, tile_counts, hashes, offsets = DRIVER(
            rust_prompt(len(sources)), sources, spec.rust_json()
        )
        meta = {
            "items": [
                {
                    "modality": "image",
                    "hash": item_hash,
                    "offsets": [list(offset)],
                }
                for item_hash, offset in zip(hashes, offsets)
            ],
            "token_ids": None,
            "mrope_delta": None,
        }
        buffers = {
            "mm.meta": np.frombuffer(msgspec.msgpack.encode(meta), dtype=np.uint8),
        }
        per = 3 * IMAGE_SIZE * IMAGE_SIZE
        row = 0
        for index, count in enumerate(tile_counts):
            n = per * count
            buffers[f"mm.feature.{index}"] = features[row : row + n].reshape(
                count, 3, IMAGE_SIZE, IMAGE_SIZE
            )
            row += n
        return snapshot(ids, RustMmProcessor.wrap_encoded(spec, buffers))

    def run_python(self, sources):
        prompt = "hello" + " <image>" * len(sources)
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=sources,
                input_text=prompt,
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="parity"
                ),
            )
        )
        return snapshot(output.input_ids, output)

    def assert_parity(self, spec, sources):
        rust, python = self.run_native(spec, sources), self.run_python(sources)
        for field in ("input_ids", "offsets", "tokens"):
            with self.subTest(field=field):
                self.assertEqual(rust[field], python[field])
        with self.subTest(field="features"):
            np.testing.assert_allclose(rust["features"], python["features"], rtol=1e-2, atol=1e-2)

    def source_forms(self, directory):
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

    def test_source_form_is_transport_only(self):
        spec = self.native_spec()
        first = image_bytes(96, 80)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(first)
            forms = {
                "raw_bytes": [first],
                "data_url": ["data:image/png;base64," + base64.b64encode(first).decode()],
                "file_uri": [path.as_uri()],
            }
            features = {
                form: self.run_native(spec, forms[form])["features"].tobytes()
                for form in forms
            }
            self.assertEqual(len(set(features.values())), 1, sorted(features))


if __name__ == "__main__":
    unittest.main()
