import unittest

from sglang.srt.utils.model_file_verifier import (
    FileInfo,
    IntegrityError,
    Manifest,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestManifestFromDict(CustomTestCase):
    def test_normal_files(self):
        m = Manifest.from_dict(
            {"files": {"a.bin": {"sha256": "abc", "size": 10}}}
        )
        self.assertEqual(m.files["a.bin"].sha256, "abc")
        self.assertEqual(m.files["a.bin"].size, 10)

    def test_missing_files_key_raises(self):
        with self.assertRaises(IntegrityError):
            Manifest.from_dict({"other": 1})
        with self.assertRaises(IntegrityError):
            Manifest.from_dict({})

    def test_non_dict_input_raises(self):
        with self.assertRaises(IntegrityError):
            Manifest.from_dict(["not", "a", "dict"])

    def test_malformed_entry_raises(self):
        with self.assertRaises(IntegrityError):
            Manifest.from_dict({"files": {"a.bin": "notadict"}})
        with self.assertRaises(IntegrityError):
            Manifest.from_dict({"files": {"a.bin": {"size": 10}}})

    def test_deprecated_checksums_still_works(self):
        with self.assertWarns(DeprecationWarning):
            m = Manifest.from_dict({"checksums": {"a.bin": "deadbeef"}})
        self.assertEqual(m.files["a.bin"].sha256, "deadbeef")
        self.assertEqual(m.files["a.bin"].size, -1)


if __name__ == "__main__":
    unittest.main()
