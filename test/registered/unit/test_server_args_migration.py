"""Tests for ServerArgs CLI argument auto-derivation from A[T, Arg(...)] annotations.

Each test covers a distinct edge case in how add_cli_args_from_dataclass
translates field annotations into argparse arguments.
"""

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.choices import REMOVED_CHOICES
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.multimodal.media_io import (
    configure_media_url_security,
    media_url_security_config,
    media_url_timeout,
    read_local_media_file,
)
from sglang.srt.runtime_context import get_model, publish, reset_context
from sglang.srt.server_args import REMOVED_FLAGS, ServerArgs, reject_removed_kwargs
from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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

    def test_media_url_timeout_and_local_files_args(self):
        try:
            sa = self._parse([])
            self.assertIsNone(sa.media_url_timeout)
            self.assertTrue(sa.allow_local_media_files)
            sa = self._parse(
                ["--media-url-timeout", "2.5", "--no-allow-local-media-files"]
            )
            self.assertEqual(sa.media_url_timeout, 2.5)
            self.assertFalse(sa.allow_local_media_files)
            # Parsing applies the policy to this process.
            self.assertEqual(media_url_timeout(3), 2.5)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                read_local_media_file(__file__)
        finally:
            configure_media_url_security([], max_file_size_mb=64)

    def test_allow_private_media_hosts_arg(self):
        try:
            self._parse(["--allow-private-media-hosts"])
            self.assertEqual(media_url_security_config()[4], True)
        finally:
            configure_media_url_security([], max_file_size_mb=64)

    def test_media_url_security_args_reject_invalid_values(self):
        try:
            with self.assertRaises(ValueError):
                self._parse(["--allowed-media-domains", "https://media.example.com"])
            with self.assertRaises(ValueError):
                self._parse(["--media-url-max-file-size-mb", "-1"])
            with self.assertRaises(ValueError):
                self._parse(["--media-url-timeout", "0"])
        finally:
            configure_media_url_security([], max_file_size_mb=64)

    def test_literal_auto_derives_choices(self):
        """Literal type annotations produce argparse choices automatically."""
        sa = self._parse(["--deepep-mode", "low_latency"])
        self.assertEqual(sa.deepep_mode, "low_latency")

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

    def test_continuous_decode_steps_is_a_deprecated_noop(self):
        from sglang.srt.entrypoints.engine import Engine

        flag = "--num-continuous-decode-steps"
        field = "num_continuous_decode_steps"
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            with open(config, "w") as handle:
                handle.write("num-continuous-decode-steps: 2\n")
            yaml_args = ConfigArgumentMerger(self.parser).merge_config_with_args(
                ["--config", config]
            )
            for source, argv in (("CLI", [flag, "2"]), ("YAML", yaml_args)):
                with self.subTest(source=source):
                    with self.assertLogs() as logs:
                        args = self._parse(argv)
                    self.assertIn(f"'{flag}' is deprecated", "\n".join(logs.output))
                    self.assertIsNone(resolution_result(args, field))

        # Stop the offline Engine at the subprocess boundary.
        with (
            patch("sglang.srt.entrypoints.engine.atexit.register"),
            patch.object(
                Engine, "_launch_subprocesses", side_effect=StopIteration
            ) as launch,
            self.assertRaises(StopIteration),
        ):
            Engine(model_path="dummy", num_continuous_decode_steps=2)
        args = launch.call_args.kwargs["server_args"]
        with self.assertLogs() as logs:
            args.resolve_once()
        self.assertIn(f"{field} is deprecated", "\n".join(logs.output))
        self.assertIsNone(resolution_result(args, field))

    def test_removed_flags_name_their_replacement(self):
        """A flag removed in SGLang 1.0 stops the launch with a message naming
        what to pass instead, not argparse's generic "unrecognized arguments";
        so does its key in a YAML `--config` file and its keyword to `Engine`."""
        # A bare flag, and one that took a value.
        for argv in (["--show-time-cost"], ["--monokernel-backend", "amd"]):
            flag = argv[0]
            replacement = REMOVED_FLAGS[flag]
            with self.subTest(flag=flag):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        self.parser.parse_args(["--model", "dummy", *argv])
                self.assertIn(
                    f"{flag} was removed in SGLang 1.0; {replacement}.",
                    " ".join(stderr.getvalue().split()),
                )

        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            with open(config, "w") as handle:
                handle.write("model-path: dummy\nshow-time-cost: true\n")
            with self.assertRaises(ValueError) as raised:
                ConfigArgumentMerger(self.parser).merge_config_with_args(
                    ["--config", config]
                )
            self.assertEqual(
                str(raised.exception),
                "--show-time-cost was removed in SGLang 1.0; "
                f"{REMOVED_FLAGS['--show-time-cost']}.",
            )

        with self.assertRaisesRegex(
            TypeError,
            "enable_dp_attention was removed in SGLang 1.0; use --attn-dp-size",
        ):
            reject_removed_kwargs(ServerArgs, {"enable_dp_attention": True})
        # `disable_cuda_graph` is still a field; only its flag was removed.
        reject_removed_kwargs(ServerArgs, {"disable_cuda_graph": True})

    def test_removed_choices_name_their_replacement(self):
        """A removed choice value stops the launch with a message naming the
        value to pass instead, not argparse's generic invalid-choice list.
        Each field wires its own parser type, so each is checked once."""
        for field, removed in REMOVED_CHOICES.items():
            flag = "--" + field.replace("_", "-")
            value, replacement = next(iter(removed.items()))
            with self.subTest(flag=flag, value=value):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit):
                        self.parser.parse_args(["--model", "dummy", flag, value])
                self.assertIn(
                    f"{value} was removed in SGLang 1.0; use {replacement}",
                    " ".join(stderr.getvalue().split()),
                )

    def test_each_deprecation_shape_does_its_job(self):
        """Four `Deprecated*Action` shapes, on a throwaway parser.

        `ServerArgs` registers `DeprecatedAction` with `error_message=` (the
        removed flags, covered above) and `DeprecatedAliasStoreAction`
        (`--mamba-max-states-per-path`). The other shapes are kept for the
        next flag that needs them, which makes this the only thing standing
        between them and silent rot -- and it doubles as the worked example of
        which shape to reach for.
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
