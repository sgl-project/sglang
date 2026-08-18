from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = (
    ROOT / "examples" / "runtime" / "deepseek_v4" / "benchmark_deepseek_5090.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_deepseek_5090", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class TestDeepseekBenchmark(unittest.TestCase):
    def _artifacts(self, root: Path) -> list[str]:
        source = root / "source.gguf"
        source.write_bytes(b"source")
        pack = root / "DeepSeek-V4-Flash.expert-pack"
        pack.write_bytes(b"pack")
        manifest = root / "DeepSeek-V4-Flash.expert-pack.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "format": "SGLANG-EXPERTPACK-v1",
                    "complete": True,
                    "pack_size": pack.stat().st_size,
                    "pack_sha256": benchmark._sha256(pack),
                    "source": {
                        "size": source.stat().st_size,
                        "sha256": benchmark._sha256(source),
                    },
                    "model": {
                        "ollama_manifest_sha256": "2" * 64,
                        "config_sha256": "3" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        return ["--gguf", str(source)]

    def test_cli_only_exposes_gguf_and_max_new_tokens(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepseek-benchmark-") as value:
            argv = self._artifacts(Path(value)) + ["--max-new-tokens", "17"]
            args = benchmark.parse_args(argv)
            self.assertEqual(args.max_new_tokens, 17)
            self.assertTrue(args.direct_io)
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                benchmark.parse_args(argv + ["--direct-io"])

    def test_validate_only_and_server_command_are_self_contained(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepseek-benchmark-") as value:
            root = Path(value)
            argv = self._artifacts(root)
            args = benchmark.parse_args(argv)
            args.model_path.mkdir(parents=True)
            (args.model_path / "config.json").write_text("{}\n", encoding="utf-8")
            artifacts = benchmark.validate_artifacts(args)
            extra = artifacts["loader_extra_config"]
            command = benchmark.build_server_command(args, extra)

            self.assertEqual(command[command.index("--load-format") + 1], "expert_pack")
            encoded = command[command.index("--model-loader-extra-config") + 1]
            self.assertEqual(json.loads(encoded), extra)
            self.assertEqual(extra["pack_path"], str(args.pack_path))
            self.assertEqual(extra["manifest_path"], str(args.manifest_path))
            self.assertEqual(extra["source_path"], str(args.source_path))
            self.assertTrue(extra["direct_io"])
            self.assertTrue(extra["verify_source_sha256"])
            self.assertEqual(args.report_path.parent, args.artifact_dir)

    def test_artifact_directory_inside_checkout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepseek-benchmark-") as value:
            argv = self._artifacts(Path(value))
            args = benchmark.parse_args(argv)
            args.model_path.mkdir(parents=True)
            args.artifact_dir = ROOT / ".forbidden-deepseek-artifacts"
            with self.assertRaisesRegex(ValueError, "outside the SGLang checkout"):
                benchmark.validate_artifacts(args)
            self.assertFalse(args.artifact_dir.exists())

    def test_run_benchmark_always_stops_server_on_generation_failure(self) -> None:
        args = SimpleNamespace(
            warmup=False,
            prompt="prompt",
            max_new_tokens=10,
        )
        process = object()
        with (
            mock.patch.object(benchmark, "start_server", return_value=process),
            mock.patch.object(
                benchmark, "generate", side_effect=RuntimeError("generation failed")
            ),
            mock.patch.object(benchmark, "stop_server") as stop_server,
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                benchmark.run_benchmark(args, {})
        stop_server.assert_called_once_with(process, args)

    def test_manifest_identity_and_sizes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepseek-benchmark-") as value:
            root = Path(value)
            argv = self._artifacts(root)
            args = benchmark.parse_args(argv)
            args.model_path.mkdir(parents=True)
            (args.model_path / "config.json").write_text("{}\n", encoding="utf-8")
            manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
            manifest["pack_size"] += 1
            args.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size does not match"):
                benchmark.validate_artifacts(args)

    def test_validate_only_checks_requested_pack_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepseek-benchmark-") as value:
            root = Path(value)
            args = benchmark.parse_args(self._artifacts(root))
            args.validate_only = True
            args.verify_pack_sha256 = True
            args.model_path.mkdir(parents=True)
            (args.model_path / "config.json").write_text("{}\n", encoding="utf-8")
            args.pack_path.write_bytes(b"same-size-bad-pack")
            manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
            manifest["pack_size"] = args.pack_path.stat().st_size
            args.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Pack SHA-256"):
                benchmark.validate_artifacts(args)


if __name__ == "__main__":
    unittest.main()
