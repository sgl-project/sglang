import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import filelock

from sglang.srt.model_loader import ci_weight_validation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCIWeightValidation(unittest.TestCase):
    def test_incomplete_blob_is_preserved_until_locked_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_id = "org/model"
            cache_dir = Path(tmpdir) / "hub"
            model_root = cache_dir / "models--org--model"
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

            is_valid = ci_weight_validation.ci_validate_and_cleanup_local_snapshot(
                model_id, str(snapshot_dir), []
            )
            self.assertFalse(is_valid)
            self.assertTrue(tokenizer_path.is_file())
            self.assertTrue(config_path.is_file())
            self.assertTrue(incomplete_path.is_file())

            lock_path = ci_weight_validation._get_lock_file_path(
                model_id, str(cache_dir)
            )
            real_cleanup = ci_weight_validation._cleanup_incomplete_blobs
            lock_checked = False

            def recheck(*_args, **_kwargs):
                is_valid = ci_weight_validation.ci_validate_and_cleanup_local_snapshot(
                    model_id, str(snapshot_dir), []
                )
                self.assertFalse(is_valid)
                return None

            def cleanup_while_locked(*args, **kwargs):
                nonlocal lock_checked
                contender = filelock.FileLock(lock_path, timeout=0)
                with self.assertRaises(filelock.Timeout):
                    contender.acquire()
                lock_checked = True
                return real_cleanup(*args, **kwargs)

            with (
                patch(
                    "sglang.srt.model_loader.weight_utils."
                    "_find_local_hf_snapshot_dir_unlocked",
                    side_effect=recheck,
                ) as recheck_mock,
                patch(
                    "huggingface_hub.snapshot_download",
                    return_value=str(snapshot_dir),
                ) as download_mock,
                patch.object(
                    ci_weight_validation,
                    "_cleanup_incomplete_blobs",
                    side_effect=cleanup_while_locked,
                ),
                patch.object(
                    ci_weight_validation,
                    "_validate_weights_after_download",
                    return_value=True,
                ),
            ):
                result = ci_weight_validation.ci_download_with_validation_and_retry(
                    model_name_or_path=model_id,
                    allow_patterns=["*.safetensors"],
                    ignore_patterns=None,
                    cache_dir=str(cache_dir),
                    revision=None,
                    max_retries=1,
                )

            self.assertEqual(result, str(snapshot_dir))
            self.assertTrue(lock_checked)
            recheck_mock.assert_called_once()
            download_mock.assert_called_once()
            self.assertFalse(incomplete_path.exists())
            self.assertTrue(tokenizer_path.is_file())
            self.assertTrue(config_path.is_file())


if __name__ == "__main__":
    unittest.main()
