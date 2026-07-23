"""Tests for ServerArgs CLI argument auto-derivation from A[T, Arg(...)] annotations.

Each test covers a distinct edge case in how add_cli_args_from_dataclass
translates field annotations into argparse arguments.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestServerArgsAnnotatedCli(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse(self, args_list):
        args = self.parser.parse_args(["--model", "dummy"] + args_list)
        return ServerArgs.from_cli_args(args)

    def test_aliases_and_dest(self):
        """Field name drives dest; long forms and short aliases both work."""
        cases = [
            ("--tp-size", "--tensor-parallel-size", "tp_size"),
            ("--pp-size", "--pipeline-parallel-size", "pp_size"),
            ("--dp-size", "--data-parallel-size", "dp_size"),
            ("--ep-size", "--expert-parallel-size", "ep_size"),
            ("--attn-cp-size", "--attention-context-parallel-size", "attn_cp_size"),
            ("--moe-dp-size", "--moe-data-parallel-size", "moe_dp_size"),
        ]
        for short, long, field in cases:
            with self.subTest(field=field):
                sa_short = self._parse([short, "4"])
                sa_long = self._parse([long, "4"])
                self.assertEqual(getattr(sa_short, field), 4)
                self.assertEqual(getattr(sa_long, field), 4)

    def test_cli_name_differs_from_field_name(self):
        """cli_name maps a different CLI flag to the dataclass field via dest."""
        sa = self._parse(
            ["--fp8-gemm-backend", "triton", "--fp4-gemm-backend", "marlin"]
        )
        self.assertEqual(sa.fp8_gemm_runner_backend, "triton")
        self.assertEqual(sa.fp4_gemm_runner_backend, "marlin")

    def test_nargs_question_with_const(self):
        """nargs='?' + const='' for --model-checksum."""
        self.assertIsNone(self._parse([]).model_checksum)
        self.assertEqual(self._parse(["--model-checksum"]).model_checksum, "")
        self.assertEqual(self._parse(["--model-checksum", "abc"]).model_checksum, "abc")

    def test_boolean_optional_action(self):
        """BooleanOptionalAction supports --flag and --no-flag."""
        self.assertIsNone(self._parse([]).experts_shared_outer_loras)
        self.assertTrue(
            self._parse(["--experts-shared-outer-loras"]).experts_shared_outer_loras
        )
        self.assertFalse(
            self._parse(["--no-experts-shared-outer-loras"]).experts_shared_outer_loras
        )

    def test_json_type_parsers(self):
        """json.loads and json_list_type parse single-string arguments."""
        sa = self._parse(
            [
                "--extra-metric-labels",
                '{"k": "v"}',
                "--forward-hooks",
                '[{"type": "test"}]',
            ]
        )
        self.assertEqual(sa.extra_metric_labels, {"k": "v"})
        self.assertEqual(sa.forward_hooks, [{"type": "test"}])

    def test_literal_auto_derives_choices(self):
        """Literal type annotations produce argparse choices automatically."""
        sa = self._parse(
            ["--deepep-mode", "low_latency", "--elastic-ep-backend", "none"]
        )
        self.assertEqual(sa.deepep_mode, "low_latency")
        self.assertEqual(sa.elastic_ep_backend, "none")

    def test_deprecated_flags_still_work(self):
        """Deprecated flags set the correct dest field."""
        sa = self._parse(["--stream-output"])
        self.assertTrue(sa.incremental_streaming_output)

    def test_combined_parse(self):
        """Multiple option types parsed together in one invocation."""
        sa = self._parse(
            [
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--data-parallel-size",
                "2",
                "--enable-lora",
                "--watchdog-timeout",
                "600",
                "--bucket-time-to-first-token",
                "0.5",
                "1.0",
                "--lora-paths",
                "p1",
                "p2",
            ]
        )
        self.assertEqual(sa.dtype, "bfloat16")
        self.assertEqual(sa.tp_size, 4)
        self.assertEqual(sa.dp_size, 2)
        self.assertTrue(sa.enable_lora)
        self.assertEqual(sa.watchdog_timeout, 600.0)
        self.assertEqual(sa.bucket_time_to_first_token, [0.5, 1.0])
        self.assertIsNotNone(sa.lora_paths)


if __name__ == "__main__":
    unittest.main()
