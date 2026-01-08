import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    compute_sha256,
    generate_checksums,
    verify,
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
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        checksums = generate_checksums(self.test_dir, checksums_file)

        self.assertEqual(len(checksums), 3)
        for filename in self.files:
            self.assertIn(filename, checksums)
            self.assertEqual(len(checksums[filename]), 64)

        self.assertTrue(os.path.exists(checksums_file))

        with open(checksums_file) as f:
            saved = json.load(f)
        self.assertEqual(saved["checksums"], checksums)

    def test_verify_intact_files(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(self.test_dir, checksums_file)
        verify(self.test_dir, checksums_file)

    def test_detect_bit_rot(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(self.test_dir, checksums_file)

        target_file = os.path.join(self.test_dir, "model.safetensors")
        _flip_bit_in_file(target_file, byte_offset=50, bit_position=3)

        with self.assertRaises(IntegrityError) as ctx:
            verify(self.test_dir, checksums_file)

        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_detect_missing_file(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(self.test_dir, checksums_file)

        os.remove(os.path.join(self.test_dir, "config.json"))

        with self.assertRaises(IntegrityError) as ctx:
            verify(self.test_dir, checksums_file)

        self.assertIn("config.json", str(ctx.exception))

    def test_verify_with_external_checksums_file(self):
        external_checksums_path = os.path.join(self.test_dir, "external_checksums.json")
        generate_checksums(self.test_dir, external_checksums_path)
        verify(self.test_dir, external_checksums_path)

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

        checksums_file = os.path.join(self.test_dir, "checksums.json")
        checksums = generate_checksums(self.test_dir, checksums_file, max_workers=4)

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
        shutil.copytree(self.original_model_path, self.test_dir, dirs_exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_server_launch_with_checksum_intact(self):
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(self.test_dir, checksums_file)

        process = popen_launch_server(
            self.test_dir,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--model-checksum", checksums_file],
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

        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(self.test_dir, checksums_file)

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
                checksums_file,
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
