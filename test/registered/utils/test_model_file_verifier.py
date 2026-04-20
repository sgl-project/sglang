import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import nullcontext
from io import StringIO

import requests
from huggingface_hub import snapshot_download

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    compute_sha256,
    generate_checksums,
    verify,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# Note: AMD registration removed - test_model_file_verifier fails on AMD
register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)

MODEL_NAME = "Qwen/Qwen3-0.6B"


# ======== Base Test Classes ========


class _FakeModelTestCase(unittest.TestCase):

    FAKE_FILES = {
        "model.safetensors": b"fake safetensors content " * 100,
        "config.json": b'{"model_type": "llama"}',
        "tokenizer.json": b'{"version": "1.0"}',
    }

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        for filename, content in self.FAKE_FILES.items():
            _create_test_file(self.test_dir, filename, content)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)


class _RealModelTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_model_path = snapshot_download(MODEL_NAME)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        shutil.copytree(self.original_model_path, self.test_dir, dirs_exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)


# ======== Unit Tests ========


class TestModelFileVerifier(_FakeModelTestCase):

    def test_detect_bit_rot(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(source=self.test_dir, output_path=checksums_file)

        target_file = os.path.join(self.test_dir, "model.safetensors")
        _flip_bit_in_file(target_file, byte_offset=50, bit_position=3)

        with self.assertRaises(IntegrityError) as ctx:
            verify(model_path=self.test_dir, checksums_source=checksums_file)

        self.assertIn("model.safetensors", str(ctx.exception))
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_detect_missing_file(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(source=self.test_dir, output_path=checksums_file)

        os.remove(os.path.join(self.test_dir, "config.json"))

        with self.assertRaises(IntegrityError) as ctx:
            verify(model_path=self.test_dir, checksums_source=checksums_file)

        self.assertIn("config.json", str(ctx.exception))

    def test_compute_sha256(self):
        test_file = os.path.join(self.test_dir, "test.bin")
        content = b"hello world"
        with open(test_file, "wb") as f:
            f.write(content)

        result = compute_sha256(file_path=test_file)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(result, expected)

    def test_parallel_checksum_computation(self):
        for i in range(10):
            _create_test_file(
                self.test_dir, f"shard_{i}.safetensors", f"content_{i}".encode() * 1000
            )

        checksums_file = os.path.join(self.test_dir, "checksums.json")
        result = generate_checksums(
            source=self.test_dir, output_path=checksums_file, max_workers=4
        )

        self.assertGreaterEqual(len(result.files), 10)

    def test_generated_json_snapshot(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(source=self.test_dir, output_path=checksums_file)

        with open(checksums_file) as f:
            data = json.load(f)

        expected = {
            "files": {
                "config.json": {
                    "sha256": "81dddc8c379baae137d99d24c5fa081d3a5ce52b6a221ddc22fe364711f8beaf",
                    "size": 23,
                },
                "model.safetensors": {
                    "sha256": "eb0c73a48a89fefb6b68dd41af830d75610c885135eac99139373b04705d05f3",
                    "size": 2500,
                },
                "tokenizer.json": {
                    "sha256": "4e3043229142b64d998563bc543ce034e0a2251af5d404995e3afcb8ce8850df",
                    "size": 18,
                },
            }
        }
        self.assertEqual(data, expected)

    def test_legacy_checksums_format_deprecated(self):
        legacy_data = {
            "checksums": {
                "model.safetensors": "eb0c73a48a89fefb6b68dd41af830d75610c885135eac99139373b04705d05f3",
                "config.json": "81dddc8c379baae137d99d24c5fa081d3a5ce52b6a221ddc22fe364711f8beaf",
                "tokenizer.json": "4e3043229142b64d998563bc543ce034e0a2251af5d404995e3afcb8ce8850df",
            }
        }
        legacy_file = os.path.join(self.test_dir, "legacy_checksums.json")
        with open(legacy_file, "w") as f:
            json.dump(legacy_data, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            verify(model_path=self.test_dir, checksums_source=legacy_file)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[0].message).lower())


# ======== CLI Tests ========


class TestModelFileVerifierCLI(_FakeModelTestCase):

    def test_cli_generate(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.utils.model_file_verifier",
                "generate",
                "--model-path",
                self.test_dir,
                "--model-checksum",
                checksums_file,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertTrue(os.path.exists(checksums_file))

        with open(checksums_file) as f:
            data = json.load(f)
        self.assertIn("files", data)
        self.assertEqual(len(data["files"]), 3)

    def test_cli_verify_success(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(source=self.test_dir, output_path=checksums_file)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.utils.model_file_verifier",
                "verify",
                "--model-path",
                self.test_dir,
                "--model-checksum",
                checksums_file,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        self.assertIn("verified successfully", result.stdout)

    def test_cli_verify_fails_on_corruption(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        generate_checksums(source=self.test_dir, output_path=checksums_file)

        target_file = os.path.join(self.test_dir, "model.safetensors")
        _flip_bit_in_file(target_file, byte_offset=50, bit_position=3)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.srt.utils.model_file_verifier",
                "verify",
                "--model-path",
                self.test_dir,
                "--model-checksum",
                checksums_file,
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertTrue(
            "IntegrityError" in combined or "mismatch" in combined.lower(),
            f"Expected integrity error, got: {combined}",
        )


# ======== HuggingFace Tests ========


class TestModelFileVerifierHF(_RealModelTestCase):

    def test_generate_checksums_from_hf(self):
        checksums_file = os.path.join(self.test_dir, "checksums.json")
        result = generate_checksums(source=MODEL_NAME, output_path=checksums_file)

        self.assertTrue(os.path.exists(checksums_file))
        self.assertGreater(len(result.files), 0)
        for filename, file_info in result.files.items():
            self.assertEqual(len(file_info.sha256), 64)

    def test_verify_with_hf_checksums_source(self):
        verify(model_path=self.test_dir, checksums_source=MODEL_NAME)


# ======== Real Model E2E Tests ========


class TestModelFileVerifierWithRealModel(_RealModelTestCase):

    def _run_server_test(self, *, corrupt_weights: bool, use_hf_checksum: bool):
        if use_hf_checksum:
            checksum_arg = MODEL_NAME
        else:
            checksums_file = os.path.join(self.test_dir, "checksums.json")
            generate_checksums(source=self.test_dir, output_path=checksums_file)
            checksum_arg = checksums_file

        corrupted_file = None
        if corrupt_weights:
            safetensors_files = [
                f for f in os.listdir(self.test_dir) if f.endswith(".safetensors")
            ]
            self.assertTrue(len(safetensors_files) > 0, "No safetensors files found")
            corrupted_file = safetensors_files[0]
            _flip_bit_in_file(os.path.join(self.test_dir, corrupted_file))

        stdout_io, stderr_io = StringIO(), StringIO()
        ctx = self.assertRaises(Exception) if corrupt_weights else nullcontext()
        with ctx:
            process = popen_launch_server(
                model=self.test_dir,
                base_url=DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--model-checksum", checksum_arg],
                return_stdout_stderr=(stdout_io, stderr_io),
            )

        if corrupt_weights:
            output = stdout_io.getvalue() + stderr_io.getvalue()
            self.assertIn(corrupted_file, output)
            self.assertIn("mismatch", output.lower())
        else:
            try:
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={"text": "Hello", "sampling_params": {"max_new_tokens": 8}},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("text", response.json())
            finally:
                kill_process_tree(process.pid)

    def test_server_launch_with_checksum_intact(self):
        self._run_server_test(corrupt_weights=False, use_hf_checksum=False)

    def test_server_launch_fails_with_corrupted_weights(self):
        self._run_server_test(corrupt_weights=True, use_hf_checksum=False)

    def test_server_launch_with_hf_checksum_intact(self):
        self._run_server_test(corrupt_weights=False, use_hf_checksum=True)

    def test_server_launch_with_hf_checksum_corrupted(self):
        self._run_server_test(corrupt_weights=True, use_hf_checksum=True)


# ======== Test Utilities ========


def _create_test_file(directory: str, filename: str, content: bytes) -> str:
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _flip_bit_in_file(file_path: str, byte_offset: int = 100, bit_position: int = 0):
    file_size = os.path.getsize(file_path)
    assert (
        byte_offset < file_size
    ), f"byte_offset {byte_offset} >= file_size {file_size}"

    with open(file_path, "r+b") as f:
        f.seek(byte_offset)
        original_byte = f.read(1)[0]
        f.seek(byte_offset)
        f.write(bytes([original_byte ^ (1 << bit_position)]))


if __name__ == "__main__":
    unittest.main()
