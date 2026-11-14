"""CPU-only descriptor validation and reuse checks; no GPU or transport."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SPEC = importlib.util.spec_from_file_location(
    "_mori_metadata_lifetime_fixture",
    Path(__file__).with_name("test_mori_packed_sender_lifetime.py"),
)
lifetime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lifetime
SPEC.loader.exec_module(lifetime)
packed = lifetime.packed


class TestMoriPackedMetadata(unittest.TestCase):
    def setUp(self):
        self.fixture = lifetime.TestMoriPackedSenderLifetime()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def send(self, sender, *, src=(3, 4), dst=(8, 9), tensors=1):
        return sender.send(
            src_ptrs=list(range(100, 100 + tensors)),
            dst_descs=list(range(tensors)),
            src_pages=src,
            dst_pages=dst,
            src_page_bytes=32,
            dst_page_bytes=16,
            page_size=4,
            head_offset_bytes=4,
        )

    def test_validation_normalizes_without_building_descriptor_matrices(self):
        self.assertEqual(packed._validate_page_metadata(2, ["8", 9], 16), [8, 9])
        for count, pages, size in ((0, [], 16), (1, [-1], 16), (1, [0], 0)):
            with self.assertRaises(ValueError):
                packed._validate_page_metadata(count, pages, size)

    def test_invalid_or_empty_pages_never_gather_or_build_descriptors(self):
        sender, engine = self.fixture.build()
        with patch.object(
            packed, "packed_page_offsets", side_effect=AssertionError("matrix")
        ):
            self.assertEqual(self.send(sender, src=(), dst=()), [])
            with self.assertRaises(ValueError):
                self.send(sender, src=(3,), dst=(-1,))
            with self.assertRaises(ValueError):
                self.send(sender, src=(-1,), dst=(8,))
        self.assertEqual(self.fixture.gathers, [])
        self.assertEqual(engine.submissions, [])
        self.assertTrue(sender.close())

    def test_sender_passes_normalized_pages_to_gather_and_offsets(self):
        sender, engine = self.fixture.build()
        self.send(sender, src=("3", "4"), dst=("8", "9"))
        self.assertEqual(self.fixture.gathers, [((100,), (3, 4))])
        self.assertEqual(engine.submissions[0][3], [[128, 144]])
        self.assertTrue(sender.close())


if __name__ == "__main__":
    unittest.main()
