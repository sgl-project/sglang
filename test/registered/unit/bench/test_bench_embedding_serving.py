import json
import unittest

from sglang.bench_embedding_serving import build_commands, create_argument_parser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBenchEmbeddingServing(CustomTestCase):
    def _parse(self, argv):
        return create_argument_parser().parse_args(argv)

    def test_default_dry_run_command_uses_embedding_backend(self):
        args = self._parse([])
        commands = build_commands(args)

        self.assertIn("sglang.launch_server", commands.launch_server)
        self.assertIn("--is-embedding", commands.launch_server)
        self.assertIn("sglang.bench_serving", commands.benchmark)
        self.assertIn("sglang-embedding", commands.benchmark)
        self.assertIn("random-ids", commands.benchmark)
        self.assertIn("--random-output-len", commands.benchmark)
        output_len_index = commands.benchmark.index("--random-output-len") + 1
        self.assertEqual(commands.benchmark[output_len_index], "0")

    def test_served_model_name_keeps_model_path_for_tokenizer(self):
        args = self._parse(
            [
                "--model-path",
                "Qwen/Qwen3-Embedding-0.6B",
                "--served-model-name",
                "embedding-model",
                "--base-url",
                "http://127.0.0.1:30000",
            ]
        )
        commands = build_commands(args)

        self.assertEqual(
            commands.benchmark[commands.benchmark.index("--model") + 1],
            "Qwen/Qwen3-Embedding-0.6B",
        )
        self.assertEqual(
            commands.benchmark[commands.benchmark.index("--served-model-name") + 1],
            "embedding-model",
        )
        self.assertIn("--base-url", commands.benchmark)
        self.assertNotIn("--host", commands.benchmark)

    def test_dimensions_are_merged_into_extra_request_body(self):
        args = self._parse(
            [
                "--dimensions",
                "128",
                "--extra-request-body",
                '{"encoding_format":"float"}',
            ]
        )
        commands = build_commands(args)

        body = json.loads(
            commands.benchmark[commands.benchmark.index("--extra-request-body") + 1]
        )
        self.assertEqual(body, {"dimensions": 128, "encoding_format": "float"})

    def test_rejects_conflicting_dimensions(self):
        args = self._parse(
            [
                "--dimensions",
                "128",
                "--extra-request-body",
                '{"dimensions":256}',
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            build_commands(args)

    def test_extra_server_args_are_appended_after_separator(self):
        args = self._parse(["--launch-server", "--", "--tp-size", "2"])
        commands = build_commands(args)

        self.assertEqual(commands.launch_server[-2:], ["--tp-size", "2"])

    def test_rejects_literal_ellipsis_passthrough(self):
        args = self._parse(["--dry-run", "..."])

        with self.assertRaisesRegex(ValueError, "literal"):
            build_commands(args)


if __name__ == "__main__":
    unittest.main()
