"""Tests for ServerArgs CLI argument auto-derivation from A[T, Arg(...)] annotations.

Each test covers a distinct edge case in how add_cli_args_from_dataclass
translates field annotations into argparse arguments.
"""

import argparse
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import get_model, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import configure_media_url_security
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
        server_args = ServerArgs.from_cli_args(args)
        # Parsing hands back the raw record; the cases below read values that
        # resolution normalises, so resolve here the way a launcher would.
        server_args.resolve_once()
        return server_args

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

    def test_media_url_security_args(self):
        try:
            sa = self._parse(
                [
                    "--allowed-media-domains",
                    "Media.Example.com.",
                    "127.0.0.1",
                    "--media-url-max-file-size-mb",
                    "32",
                ]
            )
            # The normalization is a declaration.
            self.assertEqual(
                resolution_result(sa, "allowed_media_domains"),
                ["127.0.0.1", "media.example.com"],
            )
            self.assertEqual(sa.media_url_max_file_size_mb, 32)
        finally:
            configure_media_url_security([], max_file_size_mb=64)

    def test_media_url_security_args_reject_invalid_values(self):
        try:
            with self.assertRaises(ValueError):
                self._parse(["--allowed-media-domains", "https://media.example.com"])
            with self.assertRaises(ValueError):
                self._parse(["--media-url-max-file-size-mb", "-1"])
        finally:
            configure_media_url_security([], max_file_size_mb=64)

    def test_literal_auto_derives_choices(self):
        """Literal type annotations produce argparse choices automatically."""
        sa = self._parse(
            ["--deepep-mode", "low_latency", "--elastic-ep-backend", "none"]
        )
        self.assertEqual(sa.deepep_mode, "low_latency")
        self.assertEqual(sa.elastic_ep_backend, "none")

    def test_image_processor_backend_choices(self):
        for backend in ("auto", "torchvision", "pil"):
            with self.subTest(backend=backend):
                sa = self._parse(["--image-processor-backend", backend])
                self.assertEqual(sa.image_processor_backend, backend)

    def test_startup_weight_load_mode(self):
        """The startup loading mode keeps serial as the safe default."""
        serial = self._parse([])
        overlap = self._parse(["--startup-weight-load-mode", "overlap"])
        self.assertEqual(serial.startup_weight_load_mode, "serial")
        self.assertEqual(overlap.startup_weight_load_mode, "overlap")
        # The predicate over that leaf is a bag leaf now, computed at publish.
        for record, expected in ((serial, False), (overlap, True)):
            reset_context()
            self.addCleanup(reset_context)
            publish(record, role="test")
            self.assertIs(get_model().is_startup_weight_load_overlap, expected)

        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--startup-weight-load-mode",
                    "unsupported",
                ]
            )

    def test_the_deprecated_flag_still_reaches_its_replacement(self):
        """`--disable-cuda-graph` is the one deprecated alias left standing.

        It is worth a test because it is not a rename. The flag writes
        `disable_cuda_graph`, which is `no_cli=True` -- so this spelling is its
        only way in -- and the cuda-graph hook is what turns that into the two
        phase backends the replacement flag sets directly. Asserting the
        backends, against the baseline of not passing it, covers that second
        half; asserting the dest alone would not.
        """
        from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            default_prefill_backend,
        )

        def backends(argv):
            sa = self._parse(argv)
            parse_cuda_graph_config(sa)
            config = resolution_result(sa, "cuda_graph_config")
            return sa.disable_cuda_graph, config.decode.backend, config.prefill.backend

        # Not a literal: the prefill default is BREAKABLE on CUDA and
        # TC_PIECEWISE elsewhere, and this file runs on the CPU runner.
        self.assertEqual(backends([]), (False, Backend.FULL, default_prefill_backend()))
        self.assertEqual(
            backends(["--disable-cuda-graph"]),
            (True, Backend.DISABLED, Backend.DISABLED),
        )

    def test_each_deprecation_shape_does_its_job(self):
        """The four `Deprecated*Action` shapes, on a throwaway parser.

        Only `DeprecatedStoreTrueAction` has a registration in `ServerArgs`
        today (`--disable-cuda-graph`, covered above). The other three are kept
        for the next flag that needs them, which makes this the only thing
        standing between them and silent rot -- and it doubles as the worked
        example of which shape to reach for.
        """
        from sglang.srt.arg_groups.argparse_actions import (
            DeprecatedAction,
            DeprecatedAliasStoreAction,
            DeprecatedStoreConstAction,
            DeprecatedStoreTrueAction,
        )

        def parser_with(**kwargs):
            parser = argparse.ArgumentParser()
            parser.add_argument("--old", **kwargs)
            return parser

        # An old boolean whose field survives.
        args = parser_with(
            action=DeprecatedStoreTrueAction, dest="new", new_flag="--new"
        ).parse_args(["--old"])
        self.assertIs(args.new, True)

        # An old boolean replaced by one value of a valued flag.
        args = parser_with(
            action=DeprecatedStoreConstAction,
            dest="backend",
            const_value="disabled",
            new_flag="--backend",
        ).parse_args(["--old"])
        self.assertEqual(args.backend, "disabled")

        # An old valued flag, renamed: the value survives the move.
        args = parser_with(
            action=DeprecatedAliasStoreAction, dest="new", new_flag="--new"
        ).parse_args(["--old", "fa3"])
        self.assertEqual(args.new, "fa3")

        # Retired outright: stop, and say what to use instead.
        parser = parser_with(
            action=DeprecatedAction, dest="gone", error_message="use --new instead"
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--old"])

        # Without a message it is a no-op that warns rather than a hard stop.
        args = parser_with(action=DeprecatedAction, dest="gone").parse_args(["--old"])
        self.assertIsNone(args.gone)

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
