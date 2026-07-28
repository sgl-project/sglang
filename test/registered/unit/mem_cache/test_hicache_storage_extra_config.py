"""Unit tests for HiCache storage backend extra-config parsing — no server, no GPU."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sglang.srt.mem_cache.hicache_storage import (
    parse_hicache_storage_backend_extra_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

try:
    import tomllib as _tomllib  # noqa: F401

    _TOMLLIB_AVAILABLE = True
except ImportError:
    _TOMLLIB_AVAILABLE = False

try:
    import yaml as _yaml  # noqa: F401

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _fixture_path(name: str) -> Path:
    path = Path(__file__).resolve().parent / "fixtures" / "hicache_extra_config" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing test fixture: {path}")
    return path


class TestParseHicacheStorageBackendExtraConfig(CustomTestCase):
    def _assert_file_fixture(self, filename: str) -> None:
        cfg = parse_hicache_storage_backend_extra_config(f"@{_fixture_path(filename)}")
        self.assertEqual(cfg, {"prefetch_threshold": 64})

    def test_empty_and_none_return_empty_dict(self):
        self.assertEqual(parse_hicache_storage_backend_extra_config(None), {})
        self.assertEqual(parse_hicache_storage_backend_extra_config(""), {})

    def test_inline_json_string(self):
        cfg = parse_hicache_storage_backend_extra_config(
            '{"prefetch_threshold": 128, "backend_name": "custom"}'
        )
        self.assertEqual(cfg["prefetch_threshold"], 128)
        self.assertEqual(cfg["backend_name"], "custom")

    def test_invalid_inline_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_hicache_storage_backend_extra_config("{not json}")

    def test_at_json_file_path(self):
        self._assert_file_fixture("extra.json")

    @unittest.skipUnless(_TOMLLIB_AVAILABLE, "tomllib is not available, skip test")
    def test_at_toml_file_path(self):
        self._assert_file_fixture("extra.toml")

    @unittest.skipUnless(_YAML_AVAILABLE, "PyYAML is not available, skip test")
    def test_at_yaml_file_paths(self):
        for name in ("extra.yaml", "extra.yml"):
            with self.subTest(fixture=name):
                self._assert_file_fixture(name)

    @unittest.skipUnless(
        not _TOMLLIB_AVAILABLE,
        "tomllib is available, skip test",
    )
    def test_at_toml_raises_import_error_without_tomllib(self):
        with self.assertRaises(ImportError):
            parse_hicache_storage_backend_extra_config(
                f"@{_fixture_path('extra.toml')}"
            )

    @unittest.skipUnless(
        not _YAML_AVAILABLE,
        "PyYAML is available, skip test",
    )
    def test_at_yaml_raises_import_error_without_pyyaml(self):
        with self.assertRaises(ImportError):
            parse_hicache_storage_backend_extra_config(
                f"@{_fixture_path('extra.yaml')}"
            )

    def test_unsupported_file_extension_raises(self):
        path = _fixture_path("extra.ini")
        with self.assertRaisesRegex(ValueError, "Unsupported config file"):
            parse_hicache_storage_backend_extra_config(f"@{path}")

    def test_at_missing_file_raises_file_not_found(self):
        missing_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "hicache_extra_config"
            / "does_not_exist.json"
        )
        with self.assertRaises(FileNotFoundError):
            parse_hicache_storage_backend_extra_config(f"@{missing_path}")


if __name__ == "__main__":
    unittest.main()
