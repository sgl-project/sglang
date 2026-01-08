import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    ModelFileVerifier,
    compute_sha256,
)

# ======== Test Utilities ========


def create_test_file(directory: str, filename: str, content: bytes) -> str:
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def flip_bit_in_file(file_path: str, byte_offset: int = 100, bit_position: int = 0):
    with open(file_path, "r+b") as f:
        f.seek(byte_offset)
        original_byte = f.read(1)
        if not original_byte:
            f.seek(0)
            original_byte = f.read(1)
            f.seek(0)
        else:
            f.seek(byte_offset)
        flipped_byte = bytes([original_byte[0] ^ (1 << bit_position)])
        f.write(flipped_byte)


# ======== Unit Tests ========


class TestModelFileVerifier(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.files = {
            "model.safetensors": b"fake safetensors content " * 100,
            "config.json": b'{"model_type": "llama"}',
            "tokenizer.json": b'{"version": "1.0"}',
        }
        for filename, content in self.files.items():
            create_test_file(self.test_dir, filename, content)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_checksums(self):
        verifier = ModelFileVerifier(self.test_dir)
        checksums = verifier.generate_checksums()

        self.assertEqual(len(checksums), 3)
        for filename in self.files:
            self.assertIn(filename, checksums)
            self.assertEqual(len(checksums[filename]), 64)

        checksums_file = os.path.join(self.test_dir, "checksums.json")
        self.assertTrue(os.path.exists(checksums_file))

        with open(checksums_file) as f:
            saved = json.load(f)
        self.assertEqual(saved, checksums)

    def test_verify_intact_files(self):
        verifier = ModelFileVerifier(self.test_dir)
        verifier.generate_checksums()

        verifier2 = ModelFileVerifier(self.test_dir)
        verifier2.verify()

    def test_detect_bit_rot(self):
        verifier = ModelFileVerifier(self.test_dir)
        verifier.generate_checksums()

        target_file = os.path.join(self.test_dir, "model.safetensors")
        flip_bit_in_file(target_file, byte_offset=50, bit_position=3)

        verifier2 = ModelFileVerifier(self.test_dir)
        with self.assertRaises(IntegrityError) as ctx:
            verifier2.verify()

        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_detect_missing_file(self):
        verifier = ModelFileVerifier(self.test_dir)
        verifier.generate_checksums()

        os.remove(os.path.join(self.test_dir, "config.json"))

        verifier2 = ModelFileVerifier(self.test_dir)
        with self.assertRaises(IntegrityError) as ctx:
            verifier2.verify()

        self.assertIn("config.json", str(ctx.exception))

    def test_verify_with_external_checksums_file(self):
        verifier = ModelFileVerifier(self.test_dir)
        checksums = verifier.generate_checksums()

        external_checksums_path = os.path.join(self.test_dir, "external_checksums.json")
        with open(external_checksums_path, "w") as f:
            json.dump(checksums, f)

        os.remove(os.path.join(self.test_dir, "checksums.json"))

        verifier2 = ModelFileVerifier(self.test_dir, external_checksums_path)
        verifier2.verify()

    def test_no_checksums_source_raises_error(self):
        verifier = ModelFileVerifier(self.test_dir)
        with self.assertRaises(IntegrityError) as ctx:
            verifier.verify()

        self.assertIn("No checksums found", str(ctx.exception))

    def test_compute_sha256(self):
        test_file = os.path.join(self.test_dir, "test.bin")
        content = b"hello world"
        with open(test_file, "wb") as f:
            f.write(content)

        result = compute_sha256(test_file)

        import hashlib

        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result, expected)

    def test_parallel_checksum_computation(self):
        for i in range(10):
            create_test_file(
                self.test_dir, f"shard_{i}.safetensors", f"content_{i}".encode() * 1000
            )

        verifier = ModelFileVerifier(self.test_dir, max_workers=4)
        checksums = verifier.generate_checksums()

        self.assertGreaterEqual(len(checksums), 10)


# ======== E2E Tests ========


class TestModelFileVerifierE2E(unittest.TestCase):

    def test_cli_generate_and_verify(self):
        import subprocess
        import sys

        test_dir = tempfile.mkdtemp()
        try:
            create_test_file(test_dir, "model.safetensors", b"test content " * 100)
            create_test_file(test_dir, "config.json", b'{"test": true}')

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.utils.model_file_verifier",
                    "generate",
                    "--model-path",
                    test_dir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Generate failed: {result.stderr}")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.utils.model_file_verifier",
                    "verify",
                    "--model-path",
                    test_dir,
                    "--model-checksum",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Verify failed: {result.stderr}")

            flip_bit_in_file(os.path.join(test_dir, "model.safetensors"))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.utils.model_file_verifier",
                    "verify",
                    "--model-path",
                    test_dir,
                    "--model-checksum",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr.lower())

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_cli_verify_with_explicit_checksum_file(self):
        import subprocess
        import sys

        test_dir = tempfile.mkdtemp()
        try:
            create_test_file(test_dir, "model.safetensors", b"test content " * 100)
            create_test_file(test_dir, "config.json", b'{"test": true}')

            checksums_path = os.path.join(test_dir, "my_checksums.json")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.utils.model_file_verifier",
                    "generate",
                    "--model-path",
                    test_dir,
                    "--output",
                    checksums_path,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Generate failed: {result.stderr}")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.utils.model_file_verifier",
                    "verify",
                    "--model-path",
                    test_dir,
                    "--model-checksum",
                    checksums_path,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Verify failed: {result.stderr}")

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
