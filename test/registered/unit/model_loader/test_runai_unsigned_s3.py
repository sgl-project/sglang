"""Unit tests for RunAI object storage unsigned S3 (boto3 UNSIGNED patch).

No Run:AI wheel required — mocks only.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.utils.runai_utils as runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _reset_unsigned_patch_state() -> None:
    runai_utils._BOTO3_S3_UNSIGNED_PATCH_APPLIED = False


class TestRunaiUnsignedS3Heuristics(CustomTestCase):
    def test_should_unsigned_true_without_creds_and_metadata_disabled(self):
        with patch.object(
            runai_utils, "_has_standard_aws_credentials", return_value=False
        ):
            with patch.dict(os.environ, {"AWS_EC2_METADATA_DISABLED": "true"}):
                self.assertTrue(runai_utils._should_use_unsigned_s3_for_runai())

    def test_should_unsigned_false_without_metadata_disabled(self):
        with patch.object(
            runai_utils, "_has_standard_aws_credentials", return_value=False
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
                self.assertFalse(runai_utils._should_use_unsigned_s3_for_runai())

    def test_should_unsigned_false_when_env_has_access_keys(self):
        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "AKIATEST",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "AWS_EC2_METADATA_DISABLED": "true",
            },
        ):
            self.assertFalse(runai_utils._should_use_unsigned_s3_for_runai())


class TestRunaiUnsignedS3Patch(CustomTestCase):
    def tearDown(self):
        _reset_unsigned_patch_state()
        super().tearDown()

    @patch.object(runai_utils, "_should_use_unsigned_s3_for_runai", return_value=False)
    def test_maybe_patch_skips_when_heuristic_false(self, _):
        _reset_unsigned_patch_state()
        with patch.object(
            runai_utils, "_install_boto3_s3_unsigned_patch"
        ) as mock_install:
            runai_utils._maybe_patch_boto3_s3_unsigned()
            mock_install.assert_not_called()

    @patch.object(runai_utils, "_should_use_unsigned_s3_for_runai", return_value=True)
    def test_maybe_patch_calls_install_once(self, _):
        _reset_unsigned_patch_state()
        with patch.object(
            runai_utils, "_install_boto3_s3_unsigned_patch"
        ) as mock_install:
            runai_utils._maybe_patch_boto3_s3_unsigned()
            runai_utils._maybe_patch_boto3_s3_unsigned()
            mock_install.assert_called_once()

    def test_list_safetensors_triggers_maybe_patch(self):
        _reset_unsigned_patch_state()
        with patch.object(runai_utils, "_maybe_patch_boto3_s3_unsigned") as mock_maybe:
            with patch(
                "runai_model_streamer.list_safetensors",
                return_value=["a.safetensors"],
            ):
                out = runai_utils.list_safetensors("s3://b/p/")
        mock_maybe.assert_called_once()
        self.assertEqual(out, ["a.safetensors"])

    def test_object_storage_model_init_triggers_maybe_patch(self):
        _reset_unsigned_patch_state()
        with patch.object(runai_utils, "_maybe_patch_boto3_s3_unsigned") as mock_maybe:
            with patch.object(
                runai_utils.ObjectStorageModel,
                "get_path",
                return_value="/tmp/fake-cache",
            ):
                with patch(
                    "runai_model_streamer.ObjectStorageModel",
                    MagicMock(),
                ):
                    runai_utils.ObjectStorageModel(url="s3://b/m/")
        mock_maybe.assert_called_once()

    def test_install_merges_unsigned_for_s3_only(self):
        calls: list[tuple[str, tuple, dict]] = []

        def orig_client(*args, **kwargs):
            calls.append(("client", args, dict(kwargs)))
            return MagicMock()

        class _SessionClass:
            def client(self, *args, **kwargs):
                calls.append(("session", args, dict(kwargs)))
                return MagicMock()

        class _SessMod:
            Session = _SessionClass

        boto3_mod = MagicMock()
        boto3_mod.client = orig_client
        boto3_mod.session = _SessMod()

        UNSIGNED = object()

        class _FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def merge(self, other):
                merged = _FakeConfig()
                merged.kwargs = {**self.kwargs, **other.kwargs}
                return merged

        runai_utils._install_boto3_s3_unsigned_patch(boto3_mod, _FakeConfig, UNSIGNED)

        boto3_mod.client("ec2", region_name="us-east-1")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1][0], "ec2")
        self.assertNotIn("config", calls[0][2] or {})

        boto3_mod.client("s3", region_name="us-east-1")
        self.assertEqual(calls[-1][1][0], "s3")
        cfg = calls[-1][2]["config"]
        self.assertIsInstance(cfg, _FakeConfig)
        self.assertIs(cfg.kwargs.get("signature_version"), UNSIGNED)

        boto3_mod.session.Session().client("s3", region_name="us-east-1")
        self.assertEqual(calls[-1][1][0], "s3")
        cfg2 = calls[-1][2]["config"]
        self.assertIsInstance(cfg2, _FakeConfig)
        self.assertIs(cfg2.kwargs.get("signature_version"), UNSIGNED)


if __name__ == "__main__":
    unittest.main()
