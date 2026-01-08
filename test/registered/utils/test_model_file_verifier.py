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
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)


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
            _create_test_file(self.test_dir, filename, content)

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
        _flip_bit_in_file(target_file, byte_offset=50, bit_position=3)

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
            _create_test_file(
                self.test_dir, f"shard_{i}.safetensors", f"content_{i}".encode() * 1000
            )

        verifier = ModelFileVerifier(self.test_dir, max_workers=4)
        checksums = verifier.generate_checksums()

        self.assertGreaterEqual(len(checksums), 10)


# ======== Real Model E2E Tests ========


class TestModelFileVerifierWithRealModel(unittest.TestCase):

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download

        cls.original_model_path = snapshot_download(
            cls.MODEL_NAME,
            allow_patterns=["*.safetensors", "*.json"],
        )

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        for item in os.listdir(self.original_model_path):
            src = os.path.join(self.original_model_path, item)
            dst = os.path.join(self.test_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_server_launch_with_checksum_intact(self):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        verifier = ModelFileVerifier(self.test_dir)
        verifier.generate_checksums()

        process = popen_launch_server(
            self.test_dir,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--model-checksum"],
        )
        try:
            self.assertIsNotNone(process)
            self.assertIsNone(process.poll())
        finally:
            kill_process_tree(process.pid)

    def test_server_launch_fails_with_corrupted_weights(self):
        import subprocess
        import sys

        from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

        verifier = ModelFileVerifier(self.test_dir)
        verifier.generate_checksums()

        safetensors_files = [
            f for f in os.listdir(self.test_dir) if f.endswith(".safetensors")
        ]
        self.assertTrue(len(safetensors_files) > 0, "No safetensors files found")
        target_file = os.path.join(self.test_dir, safetensors_files[0])
        _flip_bit_in_file(target_file, byte_offset=1000, bit_position=5)

        _, host, port = DEFAULT_URL_FOR_TEST.split(":")
        host = host[2:]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.test_dir,
                "--host",
                host,
                "--port",
                port,
                "--model-checksum",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertNotEqual(result.returncode, 0)
        combined_output = result.stdout + result.stderr
        self.assertTrue(
            "IntegrityError" in combined_output
            or "mismatch" in combined_output.lower(),
            f"Expected integrity error, got: {combined_output[-500:]}",
        )


# ======== Test Utilities ========


def _create_test_file(directory: str, filename: str, content: bytes) -> str:
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _flip_bit_in_file(file_path: str, byte_offset: int = 100, bit_position: int = 0):
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


if __name__ == "__main__":
    unittest.main()
