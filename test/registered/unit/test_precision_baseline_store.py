"""Unit tests for precision_baseline_store — no server, no model loading, no HF network."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from sglang.test import precision_baseline_store as hfs
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> hfs.HfStoreConfig:
    return hfs.HfStoreConfig(repo="test/repo", revision="main")


def _make_rows(n: int, *, model: str = "org/model", base_index: int = 0) -> list[dict]:
    return [
        {
            "model": model,
            "run_path": f"org__model/2025/01/{i:02d}/run-abc123{i}",
            "date": f"2025-01-{i + base_index:02d}",
            "push_index": (i + base_index) * 1000,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHfStoreConfig(CustomTestCase):
    def test_from_env_reads_required_var(self):
        with patch.dict(
            os.environ, {"SGLANG_PRECISION_HF_REPO": "my/repo"}, clear=False
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertEqual(cfg.repo, "my/repo")
        self.assertEqual(cfg.revision, "main")

    def test_from_env_reads_optional_revision(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_PRECISION_HF_REPO": "my/repo",
                "SGLANG_PRECISION_HF_REVISION": "dev",
            },
            clear=False,
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertEqual(cfg.revision, "dev")

    def test_from_env_default_revision(self):
        with patch.dict(
            os.environ, {"SGLANG_PRECISION_HF_REPO": "my/repo"}, clear=False
        ):
            cfg = hfs.HfStoreConfig.from_env()
        self.assertEqual(cfg.revision, "main")

    def test_from_env_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                hfs.HfStoreConfig.from_env()


class TestSanitizeModelName(CustomTestCase):
    def test_slashes_and_spaces(self):
        self.assertEqual(hfs._sanitize_model_name("org/model name"), "org__model_name")

    def test_no_changes_needed(self):
        self.assertEqual(hfs._sanitize_model_name("simple"), "simple")


class TestRowRecencyKey(CustomTestCase):
    def test_uses_explicit_push_index(self):
        row = {"push_index": 100}
        self.assertEqual(hfs._row_recency_key(row, 5), (100, 5))

    def test_falls_back_to_index(self):
        row = {}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))

    def test_invalid_push_index(self):
        row = {"push_index": "bad"}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))

    def test_none_push_index(self):
        row = {"push_index": None}
        self.assertEqual(hfs._row_recency_key(row, 5), (-1, 5))


class TestSelectLatestRun(CustomTestCase):
    def test_picks_highest_recency(self):
        rows = _make_rows(3)
        result = hfs._select_latest_run(rows, model="org/model")
        # Last row has highest push_index
        self.assertEqual(result, rows[-1]["run_path"])

    def test_filters_by_model(self):
        rows = [
            {"model": "a/model", "run_path": "a", "push_index": 1},
            {"model": "b/model", "run_path": "b", "push_index": 2},
        ]
        self.assertEqual(hfs._select_latest_run(rows, model="a/model"), "a")

    def test_filters_by_capture_signature(self):
        rows = [
            {
                "model": "org/m",
                "run_path": "old",
                "capture_signature": "abc123",
                "push_index": 1,
            },
            {
                "model": "org/m",
                "run_path": "new",
                "capture_signature": "def456",
                "push_index": 2,
            },
        ]
        self.assertEqual(
            hfs._select_latest_run(rows, model="org/m", capture_signature="def456"),
            "new",
        )

    def test_returns_none_on_empty(self):
        self.assertIsNone(hfs._select_latest_run([], model="org/m"))

    def test_skips_rows_without_run_path(self):
        rows = [
            {"model": "org/m", "push_index": 1},
            {"model": "org/m", "run_path": "good", "push_index": 2},
        ]
        self.assertEqual(hfs._select_latest_run(rows, model="org/m"), "good")

    def test_returns_none_when_signature_mismatch(self):
        rows = [
            {
                "model": "org/m",
                "run_path": "old",
                "capture_signature": "abc123",
                "push_index": 1,
            },
        ]
        self.assertIsNone(
            hfs._select_latest_run(rows, model="org/m", capture_signature="zzz")
        )


class TestReadManifest(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    @patch("sglang.test.precision_baseline_store._with_retries")
    def test_parses_valid_manifest(self, mock_retries, mock_download):
        content = (
            '{"model":"a","run_path":"p1","push_index":1}\n'
            '{"model":"b","run_path":"p2","push_index":2}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_retries.return_value = tmp.name
            rows, text = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "a")
        self.assertEqual(text, content)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    @patch("sglang.test.precision_baseline_store._with_retries")
    def test_skips_blank_and_corrupt_lines(self, mock_retries, mock_download):
        content = (
            '{"model":"a","run_path":"p1"}\n\nnot-json\n{"model":"b","run_path":"p2"}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_retries.return_value = tmp.name
            rows, _ = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    @patch("sglang.test.precision_baseline_store._with_retries")
    def test_returns_empty_on_not_found(self, mock_retries, mock_download):
        from huggingface_hub.errors import EntryNotFoundError

        mock_retries.side_effect = EntryNotFoundError("not found")
        rows, text = hfs._read_manifest(_make_config())
        self.assertEqual(rows, [])
        self.assertEqual(text, "")


class TestFetchLatestBaseline(CustomTestCase):
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store._with_retries")
    @patch("sglang.test.precision_baseline_store.snapshot_download")
    def test_downloads_and_copies_tensors(
        self, mock_snapshot, mock_retries, mock_manifest
    ):
        rows = [
            {"model": "org/m", "run_path": "org__m/2025/01/01/run-abc", "push_index": 1}
        ]
        mock_manifest.return_value = (rows, "")

        # snapshot_download returns a dir with fake tensor files
        with tempfile.TemporaryDirectory() as snap_dir:
            tensors = Path(snap_dir) / "org__m/2025/01/01/run-abc/tensors"
            tensors.mkdir(parents=True)
            (tensors / "layer0.pt").write_bytes(b"\x00")
            mock_retries.return_value = snap_dir

            with tempfile.TemporaryDirectory() as target:
                result = hfs.fetch_latest_baseline(
                    config=_make_config(),
                    model="org/m",
                    target_tensors_dir=Path(target),
                )
            self.assertEqual(result, "org__m/2025/01/01/run-abc")

    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_returns_none_when_no_runs(self, mock_manifest):
        mock_manifest.return_value = ([], "")
        with tempfile.TemporaryDirectory() as target:
            result = hfs.fetch_latest_baseline(
                config=_make_config(),
                model="org/m",
                target_tensors_dir=Path(target),
            )
        self.assertIsNone(result)

    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store._with_retries")
    @patch("sglang.test.precision_baseline_store.snapshot_download")
    def test_passes_capture_signature(self, mock_snapshot, mock_retries, mock_manifest):
        rows = [
            {
                "model": "org/m",
                "run_path": "run_new",
                "capture_signature": "sig2",
                "push_index": 2,
            },
            {
                "model": "org/m",
                "run_path": "run_old",
                "capture_signature": "sig1",
                "push_index": 1,
            },
        ]
        mock_manifest.return_value = (rows, "")

        with tempfile.TemporaryDirectory() as snap_dir:
            tensors = Path(snap_dir) / "run_new/tensors"
            tensors.mkdir(parents=True)
            (tensors / "layer0.pt").write_bytes(b"\x00")
            mock_retries.return_value = snap_dir

            with tempfile.TemporaryDirectory() as target:
                result = hfs.fetch_latest_baseline(
                    config=_make_config(),
                    model="org/m",
                    target_tensors_dir=Path(target),
                    capture_signature="sig2",
                )
        self.assertEqual(result, "run_new")


class TestPushRun(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store.time_ns", return_value=42)
    def test_uploads_tensors_and_manifest(self, mock_ns, mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            meta = {"tp_size": 8, "capture_signature": "abc", "hardware": "H200"}

            run_path = hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta=meta,
            )

        mock_api.upload_folder.assert_called_once()
        mock_api.upload_file.assert_called_once()
        # Verify manifest row was appended
        upload_call = mock_api.upload_file.call_args
        manifest_path = upload_call[1]["path_or_fileobj"]
        with open(manifest_path) as f:
            last_line = f.readlines()[-1].strip()
        row = json.loads(last_line)
        self.assertEqual(row["model"], "org/m")
        self.assertEqual(row["capture_signature"], "abc")
        self.assertEqual(row["tp_size"], 8)
        self.assertTrue(run_path.startswith("org__m/"))

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store.time_ns", return_value=42)
    def test_skips_existing_tensors_unless_force(
        self, mock_ns, mock_manifest, mock_api_cls
    ):
        existing_row = {
            "model": "org/m",
            "run_path": "org__m/2025/01/01/run-abc1234",
            "date": "2025-01-01",
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
            )

        # upload_folder should still be called (for meta.json) but tensors
        # dir inside the staged run should not contain .pt files.
        # Verify by checking the staged folder_path in upload_folder call.
        folder_call = mock_api.upload_folder.call_args
        staged_path = Path(folder_call[1]["folder_path"])
        pt_files = list(staged_path.rglob("*.pt"))
        self.assertEqual(len(pt_files), 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store.time_ns", return_value=42)
    def test_force_re_uploads(self, mock_ns, mock_manifest, mock_api_cls):
        existing_row = {
            "model": "org/m",
            "run_path": "org__m/2025/01/01/run-abc1234",
            "date": "2025-01-01",
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
                force=True,
            )

        folder_call = mock_api.upload_folder.call_args
        staged_path = Path(folder_call[1]["folder_path"])
        pt_files = list(staged_path.rglob("*.pt"))
        self.assertGreater(len(pt_files), 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store.time_ns", return_value=42)
    def test_manifest_row_promotes_keys(self, mock_ns, mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            meta = {
                "tp_size": 4,
                "hardware": "H100",
                "capture_signature": "sig1",
                "num_layers_compared": 10,
                "num_layers_passed": 10,
                "num_layers_failed": 0,
                "max_rel_diff": 0.001,
                "ci_run_id": "12345",
                "extra_key_not_promoted": True,
            }
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta=meta,
            )

        upload_call = mock_api_cls.return_value.upload_file.call_args
        manifest_path = upload_call[1]["path_or_fileobj"]
        with open(manifest_path) as f:
            row = json.loads(f.readlines()[-1].strip())

        for key in hfs._MANIFEST_PROMOTE_KEYS:
            if key in meta:
                self.assertEqual(
                    row.get(key), meta[key], f"manifest missing promoted key: {key}"
                )
        self.assertNotIn("extra_key_not_promoted", row)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    @patch("sglang.test.precision_baseline_store.time_ns", return_value=42)
    def test_includes_comparator_report(self, mock_ns, mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            report_path = Path(tensor_dir) / "report.jsonl"
            report_path.write_text('{"type":"comparison_tensor"}\n')

            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
                comparator_report=report_path,
            )

        # The staged upload folder should contain comparator_report.jsonl
        folder_call = mock_api.upload_folder.call_args
        staged = Path(folder_call[1]["folder_path"])
        self.assertTrue((staged / "comparator_report.jsonl").exists())


class TestPruneOldRuns(CustomTestCase):
    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_keeps_recent_runs(self, mock_manifest):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = [
            {"model": "org/m", "run_path": "recent", "date": today},
        ]
        mock_manifest.return_value = (rows, json.dumps(rows[0]) + "\n")
        result = hfs.prune_old_runs(config=_make_config(), keep_days=30)
        self.assertIn("recent", result["kept"])
        self.assertEqual(result["pruned"], [])

    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_archives_one_per_week(self, mock_manifest):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
            {"model": "org/m", "run_path": "old3", "date": "2020-01-08"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=True, dry_run=True
        )
        # One per week should be kept (the last one)
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(result["kept"][0], "old3")
        self.assertEqual(len(result["pruned"]), 2)

    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_prune_without_archive(self, mock_manifest):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=False, dry_run=True
        )
        self.assertEqual(result["kept"], [])
        self.assertEqual(len(result["pruned"]), 2)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_dry_run_does_not_delete(self, mock_manifest, mock_api_cls):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
        ]
        mock_manifest.return_value = (rows, "")
        mock_api_cls.return_value = MagicMock()
        hfs.prune_old_runs(config=_make_config(), keep_days=0, dry_run=True)
        mock_api_cls.return_value.upload_file.assert_not_called()
        mock_api_cls.return_value.delete_folder.assert_not_called()

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch("sglang.test.precision_baseline_store._with_retries")
    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_live_mode_deletes(self, mock_manifest, mock_retries, mock_api_cls):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
        ]
        mock_manifest.return_value = (rows, "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_retries.side_effect = lambda op, **kw: op()

        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=True, dry_run=False
        )
        # One kept (weekly archive), one pruned
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(len(result["pruned"]), 1)
        mock_api.upload_file.assert_called_once()
        mock_api.delete_folder.assert_called_once()

    @patch("sglang.test.precision_baseline_store._read_manifest")
    def test_filters_by_model(self, mock_manifest):
        rows = [
            {"model": "org/m1", "run_path": "m1_old", "date": "2020-01-06"},
            {"model": "org/m2", "run_path": "m2_old", "date": "2020-01-06"},
        ]
        mock_manifest.return_value = (rows, "")
        result = hfs.prune_old_runs(
            config=_make_config(),
            model="org/m1",
            keep_days=0,
            weekly_archive=False,
            dry_run=True,
        )
        # m2 should be kept (not targeted), m1 pruned
        self.assertIn("m2_old", result["kept"])
        self.assertIn("m1_old", result["pruned"])


class TestWithRetries(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.time")
    def test_succeeds_on_first_attempt(self, mock_time):
        result = hfs._with_retries(lambda: 42, what="test")
        self.assertEqual(result, 42)
        mock_time.sleep.assert_not_called()

    @patch("sglang.test.precision_baseline_store.time")
    def test_retries_on_429(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_429 = MagicMock()
        resp_429.status_code = 429
        exc_429 = HfHubHTTPError("rate limited", response=resp_429)

        calls = iter([exc_429, "ok"])
        result = hfs._with_retries(lambda: next(calls), what="test", base_delay=0.01)
        self.assertEqual(result, "ok")
        mock_time.sleep.assert_called_once()

    @patch("sglang.test.precision_baseline_store.time")
    def test_retries_on_5xx(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_500 = MagicMock()
        resp_500.status_code = 500
        exc_500 = HfHubHTTPError("server error", response=resp_500)

        calls = iter([exc_500, "ok"])
        result = hfs._with_retries(lambda: next(calls), what="test", base_delay=0.01)
        self.assertEqual(result, "ok")
        mock_time.sleep.assert_called_once()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_on_auth_error(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_401 = MagicMock()
        resp_401.status_code = 401
        exc_401 = HfHubHTTPError("unauthorized", response=resp_401)

        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(lambda: (_ for _ in ()).throw(exc_401), what="test")
        mock_time.sleep.assert_not_called()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_after_max_attempts(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_500 = MagicMock()
        resp_500.status_code = 500
        exc = HfHubHTTPError("server error", response=resp_500)

        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(
                lambda: (_ for _ in ()).throw(exc),
                what="test",
                attempts=2,
                base_delay=0.001,
            )
        self.assertEqual(mock_time.sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
