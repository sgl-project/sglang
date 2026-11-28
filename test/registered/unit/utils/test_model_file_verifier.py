import tempfile
import unittest
from pathlib import Path

from sglang.srt.utils.model_file_verifier import (
    IntegrityError,
    generate_checksums,
    verify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelFileVerifier(CustomTestCase):
    def test_regenerate_manifest_inside_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            weights = model_path / "model.safetensors"
            weights.write_bytes(b"model weights")
            output_path = model_path / "checksums.json"

            initial = generate_checksums(source=directory, output_path=str(output_path))
            verify(model_path=directory, checksums_source=str(output_path))
            regenerated = generate_checksums(
                source=directory, output_path=str(output_path)
            )

            self.assertEqual(initial, regenerated)
            self.assertEqual(set(regenerated.files), {weights.name})
            verify(model_path=directory, checksums_source=str(output_path))

            weights.write_bytes(b"corrupted weights")
            with self.assertRaisesRegex(IntegrityError, "model.safetensors: mismatch"):
                verify(model_path=directory, checksums_source=str(output_path))

    def test_external_manifest_preserves_same_named_model_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "manifest.json").write_text("{}")
            output_path = root / "manifest.json"
            output_path.write_text("previous manifest")

            manifest = generate_checksums(
                source=str(model_path), output_path=str(output_path)
            )

            self.assertEqual(set(manifest.files), {"manifest.json"})
            verify(model_path=str(model_path), checksums_source=str(output_path))

    def test_manifest_alone_is_an_empty_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "checksums.json"
            output_path.write_text('{"files": {}}')

            with self.assertRaisesRegex(IntegrityError, "No model files found"):
                generate_checksums(source=directory, output_path=str(output_path))


if __name__ == "__main__":
    unittest.main()
