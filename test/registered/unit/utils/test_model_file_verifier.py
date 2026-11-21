import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.model_file_verifier import generate_checksums, verify
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelFileVerifier(CustomTestCase):
    def test_regenerate_manifest_inside_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.safetensors").write_bytes(b"model weights")
            checksum_path = model_path / "checksums.json"
            first = generate_checksums(source=directory, output_path=str(checksum_path))
            second = generate_checksums(
                source=directory, output_path=str(checksum_path)
            )
            self.assertEqual(first, second)
            self.assertEqual(set(second.files), {"model.safetensors"})
            verify(model_path=directory, checksums_source=str(checksum_path))


if __name__ == "__main__":
    unittest.main()
