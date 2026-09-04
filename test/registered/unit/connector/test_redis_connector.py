# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from sglang.srt.connector.redis import RedisConnector
from sglang.srt.connector.utils import pull_files_from_db
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeFileConnector:

    def __init__(self, local_dir: str, files: dict[str, str]):
        self.local_dir = local_dir
        self.files = files
        self.read_keys = []

    def get_local_dir(self) -> str:
        return self.local_dir

    def list(self, prefix: str) -> list[str]:
        return list(self.files)

    def getstr(self, key: str) -> str:
        self.read_keys.append(key)
        return self.files[key]


class _FakeRedisConnection:

    def __init__(self, pages: list[tuple[int, list[bytes]]]):
        self.pages = iter(pages)
        self.scan_calls = []

    def scan(self, cursor: int, match: str) -> tuple[int, list[bytes]]:
        self.scan_calls.append((cursor, match))
        return next(self.pages)

    def close(self) -> None:
        pass


class TestRedisConnector(CustomTestCase):

    def test_pull_files_applies_allow_then_ignore_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "model/files/"
            connector = _FakeFileConnector(
                tmpdir,
                {
                    f"{prefix}config.json": "config",
                    f"{prefix}tokenizer.json": "tokenizer",
                    f"{prefix}weights.bin": "weights",
                },
            )

            pull_files_from_db(
                connector,
                "model",
                allow_pattern=["*.json"],
                ignore_pattern=["*tokenizer.json"],
            )

            self.assertEqual((Path(tmpdir) / "config.json").read_text(), "config")
            self.assertFalse((Path(tmpdir) / "tokenizer.json").exists())
            self.assertFalse((Path(tmpdir) / "weights.bin").exists())
            self.assertEqual(connector.read_keys, [f"{prefix}config.json"])

    def test_pull_files_keeps_download_root_for_nested_and_flat_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = "model/files/"
            connector = _FakeFileConnector(
                tmpdir,
                {
                    f"{prefix}nested/config.json": "config",
                    f"{prefix}tokenizer.json": "tokenizer",
                },
            )

            pull_files_from_db(connector, "model")

            self.assertEqual(
                (Path(tmpdir) / "nested" / "config.json").read_text(), "config"
            )
            self.assertEqual((Path(tmpdir) / "tokenizer.json").read_text(), "tokenizer")
            self.assertFalse((Path(tmpdir) / "nested" / "tokenizer.json").exists())

    def test_list_deduplicates_scan_results_in_first_seen_order(self):
        connection = _FakeRedisConnection(
            [
                (7, [b"model/keys/b", b"model/keys/a"]),
                (0, [b"model/keys/a", b"model/keys/c", b"model/keys/b"]),
            ]
        )
        connector = RedisConnector.__new__(RedisConnector)
        connector.connection = connection
        connector.closed = True

        self.assertEqual(
            connector.list("model/keys/"),
            ["model/keys/b", "model/keys/a", "model/keys/c"],
        )
        self.assertEqual(
            connection.scan_calls,
            [(0, "model/keys/*"), (7, "model/keys/*")],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
