"""Smoke coverage for the ``_core.inkling`` PyO3 bindings.

Covers the ``#[pyfunction]``s in ``rust/sglang-mm/src/inkling/mod.rs``
(``patchify_rgb`` / ``decode_patchify`` / ``decode_patchify_batch`` /
``preprocess_images`` / ``rescale_patchify_hash``) plus
``_core.common.content_hash``.

The inkling bindings are otherwise exercised only by the GPU e2e model test;
this pins the binding surface (signatures, dtypes, cross-binding consistency)
in the CPU suite so a rework of the extension can't silently break them.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import image_bytes, load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
PATCH_SIZE = 16


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestInklingBindings(CustomTestCase):
    def test_bindings_are_consistent(self):
        data = image_bytes(50, 34)
        h, w, patches = CORE.inkling.decode_patchify(data, PATCH_SIZE)
        self.assertEqual((h, w), (34, 50))
        patches = np.asarray(patches)
        self.assertEqual(patches.dtype, np.uint16)
        # One padded patch column, ceil rows (the inkling grid convention).
        expected_len = -(-h // PATCH_SIZE) * (w // PATCH_SIZE + 1) * PATCH_SIZE**2 * 3
        self.assertEqual(patches.size, expected_len)

        # patchify_rgb on the decoded array must match the fused decode path.
        dh, dw, rgb = CORE.common.image_decode_rgb(data)
        arr = np.asarray(rgb).reshape(dh, dw, 3)
        np.testing.assert_array_equal(
            np.asarray(CORE.inkling.patchify_rgb(arr, PATCH_SIZE)), patches
        )

        # Batch and hashed variants agree with the single-image call.
        [(bh, bw, batch)] = CORE.inkling.decode_patchify_batch([data], PATCH_SIZE)
        self.assertEqual((bh, bw), (h, w))
        np.testing.assert_array_equal(np.asarray(batch), patches)
        [(ph, pw, pre, phash)] = CORE.inkling.preprocess_images([data], PATCH_SIZE)
        self.assertEqual((ph, pw), (h, w))
        np.testing.assert_array_equal(np.asarray(pre), patches)
        self.assertEqual(phash, CORE.common.content_hash(data))

        rh, rw, rpatches, rhash = CORE.inkling.rescale_patchify_hash(
            arr, data, PATCH_SIZE
        )
        self.assertEqual((rh, rw), (h, w))
        np.testing.assert_array_equal(np.asarray(rpatches), patches)
        self.assertEqual(rhash, phash)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            CORE.inkling.decode_patchify(b"junk", PATCH_SIZE)
        with self.assertRaises(ValueError):
            CORE.inkling.decode_patchify(image_bytes(16, 16), 0)


if __name__ == "__main__":
    unittest.main()
