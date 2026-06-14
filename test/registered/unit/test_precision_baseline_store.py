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
    def test_parses_valid_manifest(self, mock_download):
        content = (
            '{"model":"a","run_path":"p1","push_index":1}\n'
            '{"model":"b","run_path":"p2","push_index":2}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_download.return_value = tmp.name
            rows, text = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "a")
        self.assertEqual(text, content)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    def test_skips_blank_and_corrupt_lines(self, mock_download):
        content = (
            '{"model":"a","run_path":"p1"}\n\nnot-json\n{"model":"b","run_path":"p2"}\n'
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            tmp.write(content)
            tmp.close()
            mock_download.return_value = tmp.name
            rows, _ = hfs._read_manifest(_make_config())
        finally:
            os.unlink(tmp.name)
        self.assertEqual(len(rows), 2)

    @patch("sglang.test.precision_baseline_store.hf_hub_download")
    def test_returns_empty_on_not_found(self, mock_download):
        from huggingface_hub.errors import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("not found")
        rows, text = hfs._read_manifest(_make_config())
        self.assertEqual(rows, [])
        self.assertEqual(text, "")


class TestFetchLatestBaseline(CustomTestCase):
    @patch("sglang.test.precision_baseline_store.snapshot_download")
    @patch.object(hfs, "_read_manifest")
    def test_downloads_and_copies_tensors(self, mock_manifest, mock_snapshot):
        rows = [
            {
                "model": "org/m",
                "run_path": "org__m/2025/01/01/run-abc",
                "push_index": 1,
            }
        ]
        mock_manifest.return_value = (rows, "")

        with tempfile.TemporaryDirectory() as snap_dir:
            tensors = Path(snap_dir) / "org__m/2025/01/01/run-abc/tensors"
            tensors.mkdir(parents=True)
            (tensors / "layer0.pt").write_bytes(b"\x00")
            mock_snapshot.return_value = snap_dir

            with tempfile.TemporaryDirectory() as target:
                result = hfs.fetch_latest_baseline(
                    config=_make_config(),
                    model="org/m",
                    target_tensors_dir=Path(target),
                )
            self.assertEqual(result, "org__m/2025/01/01/run-abc")

    @patch.object(hfs, "_read_manifest")
    def test_returns_none_when_no_runs(self, mock_manifest):
        mock_manifest.return_value = ([], "")
        with tempfile.TemporaryDirectory() as target:
            result = hfs.fetch_latest_baseline(
                config=_make_config(),
                model="org/m",
                target_tensors_dir=Path(target),
            )
        self.assertIsNone(result)

    @patch("sglang.test.precision_baseline_store.snapshot_download")
    @patch.object(hfs, "_read_manifest")
    def test_passes_capture_signature(self, mock_manifest, mock_snapshot):
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
            mock_snapshot.return_value = snap_dir

            with tempfile.TemporaryDirectory() as target:
                result = hfs.fetch_latest_baseline(
                    config=_make_config(),
                    model="org/m",
                    target_tensors_dir=Path(target),
                    capture_signature="sig2",
                )
        self.assertEqual(result, "run_new")


class TestPushRun(CustomTestCase):
    """push_run deletes its temp manifest file in a finally block, so tests
    that inspect the manifest content must capture it via a side_effect on
    the mock upload_file *before* push_run cleans up."""

    @staticmethod
    def _make_push_mocks(mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture manifest text before push_run's finally block deletes it.
        captured = []
        mock_api.upload_file.side_effect = lambda *a, **kw: captured.append(
            Path(kw["path_or_fileobj"]).read_text()
        )
        return mock_api, captured

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_uploads_tensors_and_manifest(self, mock_manifest, mock_api_cls):
        mock_api, captured = self._make_push_mocks(mock_manifest, mock_api_cls)

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
        row = json.loads(captured[0].strip().splitlines()[-1])
        self.assertEqual(row["model"], "org/m")
        self.assertEqual(row["capture_signature"], "abc")
        self.assertEqual(row["tp_size"], 8)
        self.assertTrue(run_path.startswith("org__m/"))

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_skips_existing_tensors_unless_force(self, mock_manifest, mock_api_cls):
        # The run_path must match what push_run generates: model/date/sha7.
        # _today_path() returns today's date, so build the path accordingly.
        today_date, today_date_path = hfs._today_path()
        existing_run_path = f"org__m/{today_date_path}/run-abc1234"
        existing_row = {
            "model": "org/m",
            "run_path": existing_run_path,
            "date": today_date,
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture pt file count before push_run cleans up the temp staging dir.
        captured_pt_count = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_pt_count.append(
            len(list(Path(kw["folder_path"]).rglob("*.pt")))
        )

        with tempfile.TemporaryDirectory() as tensor_dir:
            (Path(tensor_dir) / "layer0.pt").write_bytes(b"\x01")
            hfs.push_run(
                config=_make_config(),
                model="org/m",
                sglang_commit="abc1234567",
                today_tensors_dir=Path(tensor_dir),
                meta={"tp_size": 8},
            )

        self.assertEqual(captured_pt_count[0], 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_force_re_uploads(self, mock_manifest, mock_api_cls):
        # Use today's date so the run_path matches what push_run generates.
        today_date, today_date_path = hfs._today_path()
        existing_run_path = f"org__m/{today_date_path}/run-abc1234"
        existing_row = {
            "model": "org/m",
            "run_path": existing_run_path,
            "date": today_date,
            "push_index": 1,
        }
        mock_manifest.return_value = ([existing_row], json.dumps(existing_row) + "\n")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture pt file count before push_run cleans up the temp staging dir.
        captured_pt_count = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_pt_count.append(
            len(list(Path(kw["folder_path"]).rglob("*.pt")))
        )

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

        self.assertGreater(captured_pt_count[0], 0)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_manifest_row_promotes_keys(self, mock_manifest, mock_api_cls):
        _mock_api, captured = self._make_push_mocks(mock_manifest, mock_api_cls)

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

        row = json.loads(captured[0].strip().splitlines()[-1])
        for key in hfs._MANIFEST_PROMOTE_KEYS:
            if key in meta:
                self.assertEqual(
                    row.get(key),
                    meta[key],
                    f"manifest missing promoted key: {key}",
                )
        self.assertNotIn("extra_key_not_promoted", row)

    @patch("sglang.test.precision_baseline_store.HfApi")
    @patch.object(hfs, "_read_manifest")
    def test_includes_comparator_report(self, mock_manifest, mock_api_cls):
        mock_manifest.return_value = ([], "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        # Capture file existence before push_run cleans up the temp staging dir.
        captured_files = []
        mock_api.upload_folder.side_effect = lambda *a, **kw: captured_files.append(
            list(Path(kw["folder_path"]).iterdir())
        )

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

        staged_names = [f.name for f in captured_files[0]]
        self.assertIn("comparator_report.jsonl", staged_names)


class TestPruneOldRuns(CustomTestCase):
    @patch.object(hfs, "_read_manifest")
    def test_keeps_recent_runs(self, mock_manifest):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = [
            {"model": "org/m", "run_path": "recent", "date": today},
        ]
        mock_manifest.return_value = (rows, json.dumps(rows[0]) + "\n")
        result = hfs.prune_old_runs(config=_make_config(), keep_days=30)
        self.assertIn("recent", result["kept"])
        self.assertEqual(result["pruned"], [])

    @patch.object(hfs, "_read_manifest")
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
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(result["kept"][0], "old3")
        self.assertEqual(len(result["pruned"]), 2)

    @patch.object(hfs, "_read_manifest")
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
    @patch.object(hfs, "_read_manifest")
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
    @patch.object(hfs, "_read_manifest")
    def test_live_mode_deletes(self, mock_manifest, mock_api_cls):
        rows = [
            {"model": "org/m", "run_path": "old1", "date": "2020-01-06"},
            {"model": "org/m", "run_path": "old2", "date": "2020-01-07"},
        ]
        mock_manifest.return_value = (rows, "")
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        result = hfs.prune_old_runs(
            config=_make_config(), keep_days=0, weekly_archive=True, dry_run=False
        )
        self.assertEqual(len(result["kept"]), 1)
        self.assertEqual(len(result["pruned"]), 1)
        mock_api.upload_file.assert_called_once()
        mock_api.delete_folder.assert_called_once()

    @patch.object(hfs, "_read_manifest")
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

        mock_op = MagicMock(side_effect=[exc_429, "ok"])
        result = hfs._with_retries(mock_op, what="test", base_delay=0.01)
        self.assertEqual(result, "ok")
        mock_time.sleep.assert_called_once()

    @patch("sglang.test.precision_baseline_store.time")
    def test_retries_on_5xx(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_500 = MagicMock()
        resp_500.status_code = 500
        exc_500 = HfHubHTTPError("server error", response=resp_500)

        mock_op = MagicMock(side_effect=[exc_500, "ok"])
        result = hfs._with_retries(mock_op, what="test", base_delay=0.01)
        self.assertEqual(result, "ok")
        mock_time.sleep.assert_called_once()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_on_auth_error(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_401 = MagicMock()
        resp_401.status_code = 401
        exc_401 = HfHubHTTPError("unauthorized", response=resp_401)

        mock_op = MagicMock(side_effect=exc_401)
        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(mock_op, what="test")
        mock_time.sleep.assert_not_called()

    @patch("sglang.test.precision_baseline_store.time")
    def test_raises_after_max_attempts(self, mock_time):
        from huggingface_hub.errors import HfHubHTTPError

        resp_500 = MagicMock()
        resp_500.status_code = 500
        exc = HfHubHTTPError("server error", response=resp_500)

        mock_op = MagicMock(side_effect=exc)
        with self.assertRaises(HfHubHTTPError):
            hfs._with_retries(mock_op, what="test", attempts=2, base_delay=0.001)
        self.assertEqual(mock_time.sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
