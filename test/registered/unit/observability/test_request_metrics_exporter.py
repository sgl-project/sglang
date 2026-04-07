"""Unit tests for request_metrics_exporter.py — no server, no model loading."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import asyncio
import json
import os
import shutil
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

# ── Test helper classes (local only, never injected into sys.modules) ──


@dataclass
class _GenerateReqInput:
    rid: Optional[str] = None
    text: Optional[str] = None
    image_data: Optional[Any] = None
    sampling_params: Optional[Dict] = None


@dataclass
class _EmbeddingReqInput:
    rid: Optional[str] = None
    text: Optional[str] = None
    image_data: Optional[Any] = None
    input_ids: Optional[List[int]] = None


class _ServerArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Deferred import of the module-under-test ──
# request_metrics_exporter.py imports io_struct and server_args at module level.
# We use patch.dict to temporarily provide lightweight stubs so the import
# succeeds without pulling in heavy transitive deps (torch, triton, …).
# The patch is started in setUpModule and stopped in tearDownModule,
# so sys.modules is never modified during pytest collection.

_patcher = None

# Module-under-test symbols, populated by setUpModule
FileRequestMetricsExporter = None
RequestMetricsExporter = None
RequestMetricsExporterManager = None
create_request_metrics_exporters = None
_ConcreteExporter = None


def setUpModule():
    global _patcher
    global FileRequestMetricsExporter, RequestMetricsExporter
    global RequestMetricsExporterManager, create_request_metrics_exporters
    global _ConcreteExporter

    stub_modules = {}
    for name in (
        "sglang.srt.managers",
        "sglang.srt.managers.io_struct",
        "sglang.srt.server_args",
    ):
        if name not in __import__("sys").modules:
            stub_modules[name] = types.ModuleType(name)

    if stub_modules:
        if "sglang.srt.managers.io_struct" in stub_modules:
            stub_modules["sglang.srt.managers.io_struct"].GenerateReqInput = (
                _GenerateReqInput
            )
            stub_modules["sglang.srt.managers.io_struct"].EmbeddingReqInput = (
                _EmbeddingReqInput
            )
        if "sglang.srt.server_args" in stub_modules:
            stub_modules["sglang.srt.server_args"].ServerArgs = _ServerArgs

        _patcher = patch.dict("sys.modules", stub_modules)
        _patcher.start()

    import sglang.srt.observability.request_metrics_exporter as _mod

    FileRequestMetricsExporter = _mod.FileRequestMetricsExporter
    RequestMetricsExporter = _mod.RequestMetricsExporter
    RequestMetricsExporterManager = _mod.RequestMetricsExporterManager
    create_request_metrics_exporters = _mod.create_request_metrics_exporters

    class ConcreteExporter(RequestMetricsExporter):
        """Minimal concrete subclass for testing base class methods."""

        async def write_record(self, obj, out_dict):
            pass

    _ConcreteExporter = ConcreteExporter


def tearDownModule():
    if _patcher is not None:
        _patcher.stop()


# ── Helpers ──


def _make_server_args(tmp_dir, enabled=True):
    return _ServerArgs(
        export_metrics_to_file=enabled,
        export_metrics_to_file_dir=tmp_dir,
    )


class TestFormatOutputData(unittest.TestCase):
    def test_basic_formatting(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(
            rid="req-1", text="hello", sampling_params={"temp": 0.5}
        )
        out_dict = {"meta_info": {"latency": 1.5, "tokens": 10}}

        result = exporter._format_output_data(obj, out_dict)

        params = json.loads(result["request_parameters"])
        self.assertEqual(params["rid"], "req-1")
        self.assertEqual(params["text"], "hello")
        self.assertIn("latency", result)
        self.assertIn("tokens", result)

    def test_excludes_always_exclude_fields(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", image_data="should_be_excluded")
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("image_data", params)

    def test_excludes_obj_skip_names(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names={"text"}, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", text="skip_me")
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("text", params)
        self.assertIn("rid", params)

    def test_excludes_none_values(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names=None
        )

        obj = _GenerateReqInput(rid="req-1", text=None)
        result = exporter._format_output_data(obj, {})

        params = json.loads(result["request_parameters"])
        self.assertNotIn("text", params)

    def test_filters_out_skip_names(self):
        server_args = _make_server_args("/tmp/unused")
        exporter = _ConcreteExporter(
            server_args, obj_skip_names=None, out_skip_names={"secret"}
        )

        obj = _GenerateReqInput(rid="req-1")
        out_dict = {"meta_info": {"latency": 1.5, "secret": "hidden"}}
        result = exporter._format_output_data(obj, out_dict)

        self.assertIn("latency", result)
        self.assertNotIn("secret", result)


class TestFileRequestMetricsExporter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_exporter(self):
        return FileRequestMetricsExporter(_make_server_args(self.tmp_dir), None, None)

    def test_init_creates_directory(self):
        sub_dir = os.path.join(self.tmp_dir, "nested", "dir")
        FileRequestMetricsExporter(_make_server_args(sub_dir), None, None)
        self.assertTrue(os.path.isdir(sub_dir))

    def test_ensure_file_handler_opens_file(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        self.assertIsNotNone(exporter._current_file_handler)
        self.assertEqual(exporter._current_hour_suffix, "20240101_12")
        exporter.close()

    def test_ensure_file_handler_rotates(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        first_handler = exporter._current_file_handler
        exporter._ensure_file_handler("20240101_13")
        self.assertTrue(first_handler.closed)
        self.assertEqual(exporter._current_hour_suffix, "20240101_13")
        exporter.close()

    def test_ensure_file_handler_close_error(self):
        """Previous handler close failure is logged but doesn't prevent rotation."""
        exporter = self._make_exporter()
        mock_handler = MagicMock()
        mock_handler.close.side_effect = OSError("disk error")
        exporter._current_file_handler = mock_handler
        exporter._current_hour_suffix = "old"

        exporter._ensure_file_handler("new")
        self.assertEqual(exporter._current_hour_suffix, "new")
        exporter.close()

    def test_ensure_file_handler_open_error(self):
        exporter = self._make_exporter()
        with patch("builtins.open", side_effect=OSError("permission denied")):
            with self.assertRaises(OSError):
                exporter._ensure_file_handler("20240101_12")
        self.assertIsNone(exporter._current_file_handler)
        self.assertIsNone(exporter._current_hour_suffix)

    def test_close(self):
        exporter = self._make_exporter()
        exporter._ensure_file_handler("20240101_12")
        exporter.close()
        self.assertIsNone(exporter._current_file_handler)
        self.assertIsNone(exporter._current_hour_suffix)

    def test_close_noop_when_no_handler(self):
        exporter = self._make_exporter()
        exporter.close()  # should not raise

    def test_close_error(self):
        """Close failure is logged but state is still reset."""
        exporter = self._make_exporter()
        mock_handler = MagicMock()
        mock_handler.close.side_effect = OSError("disk error")
        exporter._current_file_handler = mock_handler
        exporter._current_hour_suffix = "old"

        exporter.close()
        self.assertIsNone(exporter._current_file_handler)
        self.assertIsNone(exporter._current_hour_suffix)

    def test_write_record(self):
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1", text="hello")
        out_dict = {"meta_info": {"latency": 1.5}}

        asyncio.run(exporter.write_record(obj, out_dict))

        # Find the written file
        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 1)
        with open(os.path.join(self.tmp_dir, files[0])) as f:
            record = json.loads(f.readline())
        self.assertIn("request_parameters", record)
        self.assertAlmostEqual(record["latency"], 1.5)
        exporter.close()

    def test_write_record_skips_health_check(self):
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid=f"{HEALTH_CHECK_RID_PREFIX}_123", text="ping")
        asyncio.run(exporter.write_record(obj, {}))

        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 0)

    def test_write_record_handler_none(self):
        """If file handler is None after ensure, write_record returns early."""
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1")

        with patch.object(exporter, "_ensure_file_handler"):
            exporter._current_file_handler = None
            asyncio.run(exporter.write_record(obj, {}))
        # No crash, no file written

    def test_write_record_exception(self):
        """Exceptions during write are caught and logged."""
        exporter = self._make_exporter()
        obj = _GenerateReqInput(rid="req-1")

        with patch.object(
            exporter, "_ensure_file_handler", side_effect=RuntimeError("boom")
        ):
            asyncio.run(exporter.write_record(obj, {}))
        # Should not raise


class TestRequestMetricsExporterManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_exporters(self):
        server_args = _make_server_args(self.tmp_dir, enabled=False)
        manager = RequestMetricsExporterManager(server_args)
        self.assertFalse(manager.exporter_enabled())

    def test_with_file_exporter(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        manager = RequestMetricsExporterManager(server_args)
        self.assertTrue(manager.exporter_enabled())

    def test_write_record_delegates(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        manager = RequestMetricsExporterManager(server_args)

        obj = _GenerateReqInput(rid="req-1", text="hello")
        out_dict = {"meta_info": {"latency": 1.0}}
        asyncio.run(manager.write_record(obj, out_dict))

        files = os.listdir(self.tmp_dir)
        self.assertEqual(len(files), 1)


class TestCreateExporters(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_disabled(self):
        server_args = _make_server_args(self.tmp_dir, enabled=False)
        exporters = create_request_metrics_exporters(server_args)
        self.assertEqual(len(exporters), 0)

    def test_enabled(self):
        server_args = _make_server_args(self.tmp_dir, enabled=True)
        exporters = create_request_metrics_exporters(server_args)
        self.assertEqual(len(exporters), 1)
        self.assertIsInstance(exporters[0], FileRequestMetricsExporter)


if __name__ == "__main__":
    unittest.main()
