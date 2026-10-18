import tempfile
import unittest
from pathlib import Path

from sglang.srt.model_loader.ci_weight_validation import (
    ci_validate_and_cleanup_local_snapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCIWeightValidation(unittest.TestCase):
    def test_incomplete_blob_does_not_delete_shared_model_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir) / "hub" / "models--org--model"
            snapshot_dir = model_root / "snapshots" / "revision"
            blobs_dir = model_root / "blobs"
            snapshot_dir.mkdir(parents=True)
            blobs_dir.mkdir()

            tokenizer_path = snapshot_dir / "tokenizer.json"
            config_path = snapshot_dir / "config.json"
            incomplete_path = blobs_dir / "active.incomplete"
            tokenizer_path.write_text('{"version": "1.0"}', encoding="utf-8")
            config_path.write_text("{}", encoding="utf-8")
            incomplete_path.write_bytes(b"partial")

            is_valid = ci_validate_and_cleanup_local_snapshot(
                "org/model", str(snapshot_dir), []
            )

            self.assertFalse(is_valid)
            self.assertTrue(tokenizer_path.is_file())
            self.assertTrue(config_path.is_file())
            self.assertTrue(incomplete_path.is_file())


if __name__ == "__main__":
    unittest.main()
