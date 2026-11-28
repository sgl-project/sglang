"""Image-token row count of the Nemotron Shortcut multimodal processor."""

import unittest

from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.nemotron_shortcut import (  # noqa: E402
    NemotronShortcutMMProcessor,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# (width, height) -> rows of image tokens, as the checkpoint's own processor
# emits them (row breaks plus the end token) for a 1036 longest edge and
# 28-pixel merged patches.
CHECKPOINT_ROWS = {
    (640, 427): 16,
    (1, 1): 1,
    (27, 29): 2,
    (2000, 300): 6,
    (300, 2000): 37,
    (1036, 1036): 37,
    (1037, 1035): 37,
    (4096, 3072): 28,
    (55, 1100): 37,
}


class TestImageRows(CustomTestCase):
    def test_rows_match_the_checkpoint_processor(self):
        processor = NemotronShortcutMMProcessor.__new__(NemotronShortcutMMProcessor)
        processor._merged_patch = 28
        processor._longest_edge = 1036
        images = [Image.new("RGB", size) for size in CHECKPOINT_ROWS]
        self.assertEqual(
            processor._get_image_nrows(images), list(CHECKPOINT_ROWS.values())
        )


if __name__ == "__main__":
    unittest.main()
