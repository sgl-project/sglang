"""Custom warmups must run without HTTP-server or processor initialization."""

import asyncio
import base64
import io
import unittest
from types import SimpleNamespace

from PIL import Image

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.warmup import execute_warmups
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCustomWarmup(CustomTestCase):
    def test_uocr_warmup_runs_without_processor_discovery(self):
        """A missing registration would silently skip the requested warmup."""
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                completed = []

                async def generate_request(req, request):
                    yield {"text": "partial"}
                    yield {"text": "complete"}
                    completed.append(req)

                asyncio.run(
                    execute_warmups(
                        mode,
                        ["unlimited_ocr_v2"],
                        SimpleNamespace(generate_request=generate_request),
                    )
                )

                self.assertEqual(len(completed), 1)
                req = completed[0]
                self.assertEqual(req.text, "<PAGE><image>Multi page parsing.")
                self.assertEqual(req.images_config, {"image_mode": "gundam"})
                self.assertEqual(
                    req.sampling_params, {"max_new_tokens": 8, "temperature": 0}
                )
                self.assertEqual(len(req.image_data), 1)
                prefix, image_data = req.image_data[0].split(",", 1)
                self.assertEqual(prefix, "data:image/png;base64")
                with Image.open(io.BytesIO(base64.b64decode(image_data))) as image:
                    self.assertEqual(image.size, (980, 1400))
                    image.verify()
                self.assertEqual(req.bootstrap_room, None if mode == "null" else 0)
                self.assertEqual(
                    req.bootstrap_host, None if mode == "null" else FAKE_BOOTSTRAP_HOST
                )

    def test_uocr_warmup_propagates_errors_after_first_response(self):
        """Startup must not succeed while the warmup generator is still failing."""

        async def generate_request(req, request):
            yield {"text": "partial"}
            raise RuntimeError("warmup inference failed")

        with self.assertRaisesRegex(RuntimeError, "warmup inference failed"):
            asyncio.run(
                execute_warmups(
                    "null",
                    ["unlimited_ocr_v2"],
                    SimpleNamespace(generate_request=generate_request),
                )
            )


if __name__ == "__main__":
    unittest.main()
