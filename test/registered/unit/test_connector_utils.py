import os
import signal
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

from sglang.srt.connector.base_connector import BaseKVConnector
from sglang.srt.connector.utils import parse_model_name, pull_files_from_db
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


class _DummyKVConnector(BaseKVConnector):
    def __init__(self, entries: dict[str, str]):
        super().__init__("redis://127.0.0.1:6379/my-model")
        self._entries = entries

    def get(self, key: str):
        return None

    def getstr(self, key: str) -> Optional[str]:
        return self._entries.get(key)

    def set(self, key: str, obj) -> None:
        pass

    def setstr(self, key: str, txt: str) -> None:
        self._entries[key] = txt

    def list(self, prefix: str) -> List[str]:
        return [k for k in self._entries if k.startswith(prefix)]

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        pull_files_from_db(self, "my-model", allow_pattern, ignore_pattern)

    def weight_iterator(self, rank: int):
        return iter(())


class TestConnectorUtils(CustomTestCase):
    def test_parse_model_name(self):
        self.assertEqual(
            parse_model_name("redis://127.0.0.1:6379/llama-3-8b"),
            "llama-3-8b",
        )

    def test_pull_files_from_db_preserves_root_dir_across_subdirectories(self):
        entries = {
            "my-model/files/sub1/a.json": '{"a": 1}',
            "my-model/files/sub2/nested/b.json": '{"b": 2}',
            "my-model/files/config.json": '{"root": true}',
        }
        with _DummyKVConnector(entries) as conn:
            conn.pull_files()
            root = Path(conn.get_local_dir())
            self.assertEqual((root / "sub1" / "a.json").read_text(), '{"a": 1}')
            self.assertEqual(
                (root / "sub2" / "nested" / "b.json").read_text(), '{"b": 2}'
            )
            self.assertEqual((root / "config.json").read_text(), '{"root": true}')
            # Ensure subsequent files were not nested under sub1/
            self.assertFalse((root / "sub1" / "sub2").exists())
            self.assertFalse((root / "sub1" / "config.json").exists())

    def test_pull_files_from_db_allow_and_ignore_patterns(self):
        entries = {
            "my-model/files/config.json": "{}",
            "my-model/files/tokenizer.json": "{}",
            "my-model/files/README.md": "readme",
            "my-model/files/weights.bin": "weights",
        }
        with _DummyKVConnector(entries) as conn:
            conn.pull_files(
                allow_pattern=["*.json", "*.md"],
                ignore_pattern=["tokenizer*"],
            )
            root = Path(conn.get_local_dir())
            self.assertTrue((root / "config.json").exists())
            self.assertTrue((root / "README.md").exists())
            self.assertFalse((root / "tokenizer.json").exists())
            self.assertFalse((root / "weights.bin").exists())

    def test_close_by_signal_handles_sig_ign_and_callable(self):
        conn = _DummyKVConnector({})
        local_dir = conn.get_local_dir()
        self.assertTrue(os.path.exists(local_dir))

        # Should not raise TypeError when previous handler is signal.SIG_IGN (int 1)
        handler = conn._close_by_signal(signal.SIG_IGN)
        handler(signal.SIGTERM, None)
        self.assertTrue(conn.closed)
        self.assertFalse(os.path.exists(local_dir))

        # Should forward signal to a callable previous handler
        called = []
        conn2 = _DummyKVConnector({})
        handler2 = conn2._close_by_signal(lambda sig, frame: called.append(sig))
        handler2(signal.SIGINT, None)
        self.assertEqual(called, [signal.SIGINT])
        self.assertTrue(conn2.closed)


if __name__ == "__main__":
    unittest.main()
