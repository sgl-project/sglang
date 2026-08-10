"""The mm-item hash contract of the native Qwen path.

The two paths hash different things, deliberately. Native hashes are `content_hash`
over the raw encoded source bytes, computed on an MM worker; the Python path hashes
the decoded feature tensor. Both feed `set_pad_value`, so the native drain can skip
`hash_feature` on the scheduler loop — the point of precomputing them.

Field-by-field parity of everything else is `test_e2e_parity.py`.
"""

import asyncio
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.mm_utils import hash_feature  # noqa: E402
from sglang.srt.managers.rust_server import NativeMmHost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _fixtures import make_processor  # noqa: E402
from _mm_rust_utils import PROCESSOR_CONFIGS, image_bytes, load_core  # noqa: E402

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

CORE = load_core()
DRIVER = getattr(getattr(CORE, "qwen_vl", None), "process_native_mm", None)


def raw_bytes(source):
    """The encoded bytes behind any accepted source form."""
    if isinstance(source, bytes):
        return source
    if source.startswith("data:"):
        return base64.b64decode(source.split(",", 1)[1])
    return Path(source.removeprefix("file://")).read_bytes()


@unittest.skipUnless(DRIVER, "sglang-mm native Qwen driver not built")
class TestQwenNativeMmHashes(CustomTestCase):
    def setUp(self):
        from sglang.srt.managers.multimodal_processor import import_processors

        import_processors("sglang.srt.multimodal.processors")
        self.processor = make_processor(PROCESSOR_CONFIGS["qwen2_5_vl"])

    def tearDown(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()

    def native_hashes(self, sources):
        """Per-item hashes the Rust driver returns, via the production gate."""
        host = NativeMmHost.__new__(NativeMmHost)
        host.model_config = SimpleNamespace(hf_config=self.processor.hf_config)
        host._processor = self.processor._processor
        host.server_args = self.processor.server_args
        spec = host.resolve_native_spec()
        self.assertIsNotNone(spec, "gate rejected the fixture processor")
        input_ids = [t for _ in sources for t in (1, 2, 3, 4)]
        return DRIVER(input_ids, sources, spec.rust_json())[3]

    def test_native_hashes_the_raw_source_bytes(self):
        """Same image in any source form hashes identically, because the hash is
        over the bytes and not over anything the transport changes."""
        first, second = image_bytes(96, 80), image_bytes(112, 88, 1)
        data_url = "data:image/png;base64," + base64.b64encode(first).decode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(first)
            for sources in ([first], [data_url], [path.as_uri()], [first, second]):
                with self.subTest(n=len(sources), form=type(sources[0]).__name__):
                    hashes = self.native_hashes(sources)
                    self.assertEqual(
                        list(hashes),
                        [CORE.common.content_hash(raw_bytes(s)) for s in sources],
                    )

    def test_python_hashes_the_feature(self):
        """The contrast: `set_pad_value` on the Python path derives its hash from
        the feature tensor, which is why the native path must precompute one."""
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
            expected = hash_feature(item.feature)
            item.set_pad_value()
            self.assertEqual(item.hash, expected)


if __name__ == "__main__":
    unittest.main()
