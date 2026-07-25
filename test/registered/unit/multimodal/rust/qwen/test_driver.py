"""Native driver error paths: out-of-scope and malformed inputs are rejected.

Covers ``process`` in ``rust/sglang-mm/src/driver.rs`` (via the
``_core.qwen_vl.process_native_mm`` binding). The wire-payload parsing that
feeds this driver (modality/shape rejection) lives in ``sglang-server``'s
message layer and is tested with the integration PR.

There is no Python fallback path, so the server rejects every driver error
back to the client as a 400; the message must say why (placeholder mismatch,
undecodable image, missing prompt). This pins that contract for each
rejection class.
"""

import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import (  # noqa: E402
    IMAGE_TOKEN_ID,
    PROCESSOR_CONFIGS,
    VISION_END_ID,
    VISION_START_ID,
    image_bytes,
    load_core,
    spec_json,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

QWEN_CORE = getattr(load_core(), "qwen_vl", None)
SPEC = spec_json(PROCESSOR_CONFIGS["qwen2_5_vl"])
IMAGE_IDS = [7, VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID, 8]


def gif_bytes():
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(buffer, format="GIF")
    return buffer.getvalue()


@unittest.skipUnless(
    QWEN_CORE and hasattr(QWEN_CORE, "process_native_mm"),
    "sglang-mm native Qwen driver not built",
)
class TestNativeDriverErrorPaths(CustomTestCase):
    def assert_rejected(self, input_ids, images, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            QWEN_CORE.process_native_mm(input_ids, images, SPEC)

    def test_placeholder_count_mismatches_rejected(self):
        cases = {
            "no placeholder": ([7, 8], [image_bytes(80, 80)]),
            "more images": (IMAGE_IDS, [image_bytes(80, 80), image_bytes(88, 80, 1)]),
            "more placeholders": (IMAGE_IDS + IMAGE_IDS, [image_bytes(80, 80)]),
        }
        for name, (ids, images) in cases.items():
            with self.subTest(case=name):
                self.assert_rejected(ids, images, "placeholder")

    def test_undecodable_images_rejected(self):
        # PIL-only formats (and corrupt bytes) are outside the native
        # decoder's scope; the server rejects them as a 400.
        for name, data in {"gif": gif_bytes(), "corrupt": b"junk"}.items():
            with self.subTest(image=name):
                self.assert_rejected(IMAGE_IDS, [data], "decode")

    def test_missing_text_and_input_ids_rejected(self):
        for input_ids in (None, []):
            with self.subTest(input_ids=input_ids):
                self.assert_rejected(
                    input_ids, [image_bytes(80, 80)], "without text or input_ids"
                )

    def test_image_free_request_rejected(self):
        self.assert_rejected(IMAGE_IDS, [], "image sources")


if __name__ == "__main__":
    unittest.main()
