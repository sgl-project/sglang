from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = ROOT / "examples" / "runtime" / "kimi_k3" / "benchmark_kimi_k3_5090.py"
SPEC = importlib.util.spec_from_file_location("benchmark_kimi_k3_5090", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class TestKimiBenchmark(unittest.TestCase):
    def _argv(self, root: Path) -> list[str]:
        gguf_dir = root / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "KIMI-K3-00001-of-00001.gguf").write_bytes(b"gguf")
        return [
            "--gguf",
            str(gguf_dir / "KIMI-K3-00001-of-00001.gguf"),
        ]

    def test_all_generated_paths_are_under_external_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kimi-benchmark-") as value:
            root = Path(value)
            args = benchmark.parse_args(self._argv(root))
            self.assertEqual(args.server_log.parent, args.artifact_dir)
            self.assertEqual(args.stats_path.parent, args.artifact_dir)
            self.assertEqual(args.report_path.parent, args.artifact_dir)
            self.assertNotIn(ROOT, args.artifact_dir.parents)
            self.assertEqual(args.gguf.parent, root / "gguf")

    def test_artifact_directory_is_derived_and_external(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kimi-benchmark-") as value:
            args = benchmark.parse_args(self._argv(Path(value)))
            self.assertNotIn(ROOT, args.artifact_dir.parents)

    def test_cli_only_exposes_required_paths_and_token_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kimi-benchmark-") as value:
            argv = self._argv(Path(value))
            args = benchmark.parse_args([*argv, "--max-new-tokens", "17"])
            self.assertEqual(args.max_new_tokens, 17)
            self.assertTrue(args.direct_io)
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                benchmark.parse_args([*argv, "--direct-io"])

    def test_server_command_leaves_artifact_preparation_to_sglang(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kimi-benchmark-") as value:
            args = benchmark.parse_args(self._argv(Path(value)))
            command = benchmark.build_server_command(args)
            self.assertEqual(command[command.index("--model-path") + 1], str(args.gguf))
            self.assertNotIn("--tokenizer-path", command)
            encoded = command[command.index("--model-loader-extra-config") + 1]
            loader_config = json.loads(encoded)
            self.assertNotIn("pack_path", loader_config)
            self.assertNotIn("manifest_path", loader_config)
            self.assertNotIn("source_path", loader_config)
            self.assertEqual(loader_config["read_splits"], 1)
            self.assertTrue(loader_config["direct_io"])


if __name__ == "__main__":
    unittest.main()
