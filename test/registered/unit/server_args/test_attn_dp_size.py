"""--attn-dp-size sets the attention data-parallel width; the deprecated
`--dp-size N --enable-dp-attention` spelling resolves to `--attn-dp-size N`."""

import argparse
import json
import logging
import os
import shutil
import tempfile
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# A real config.json, so resolution runs past the dummy-model early return.
_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class TestAttnDpSize(CustomTestCase):
    def setUp(self):
        super().setUp()
        environment = dict(os.environ)

        def restore():
            os.environ.clear()
            os.environ.update(environment)

        self.addCleanup(restore)
        self.path = tempfile.mkdtemp(prefix="attn_dp_size_")
        self.addCleanup(shutil.rmtree, self.path, ignore_errors=True)
        with open(os.path.join(self.path, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)

    def resolve(self, **fields):
        server_args = ServerArgs(
            model_path=self.path, device="cuda", random_seed=42, **fields
        )
        server_args.resolve_once()
        return server_args

    def layout(self, server_args):
        return tuple(
            resolution_result(server_args, name)
            for name in ("dp_size", "attn_dp_size", "enable_dp_attention")
        )

    def test_the_deprecated_spelling_moves_the_width_to_attn_dp_size(self):
        for tp_size, width in ((2, 2), (4, 4), (4, 2), (8, 4)):
            with self.subTest(tp_size=tp_size, width=width):
                legacy = self.resolve(
                    tp_size=tp_size, dp_size=width, enable_dp_attention=True
                )
                new = self.resolve(tp_size=tp_size, attn_dp_size=width)
                self.assertEqual(self.layout(legacy), (1, width, False))
                self.assertEqual(legacy.resolved_dict(), new.resolved_dict())

    def test_without_attention_dp_the_width_is_one(self):
        for fields in (
            {"tp_size": 2},
            {"tp_size": 2, "dp_size": 2},
            # One DP group under the deprecated flag is no attention DP.
            {"tp_size": 2, "dp_size": 1, "enable_dp_attention": True},
        ):
            with self.subTest(**fields):
                server_args = self.resolve(**fields)
                self.assertEqual(
                    self.layout(server_args), (fields.get("dp_size", 1), 1, False)
                )

    def test_rejected_layouts(self):
        for fields, message in (
            (
                {"tp_size": 4, "dp_size": 2, "attn_dp_size": 2},
                "replicas combined with attention data parallelism",
            ),
            (
                {
                    "tp_size": 4,
                    "dp_size": 2,
                    "attn_dp_size": 2,
                    "enable_dp_attention": True,
                },
                "replicas combined with attention data parallelism",
            ),
            ({"tp_size": 2, "attn_dp_size": 0}, "must be positive"),
        ):
            with self.subTest(**fields):
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve(**fields)

    def test_the_deprecated_flag_next_to_attn_dp_size_is_redundant(self):
        both = self.resolve(tp_size=4, attn_dp_size=2, enable_dp_attention=True)
        new = self.resolve(tp_size=4, attn_dp_size=2)
        self.assertEqual(self.layout(both), (1, 2, False))
        self.assertEqual(both.resolved_dict(), new.resolved_dict())

    def test_the_readback_reports_the_deprecated_field_as_attention_dp(self):
        for fields, attention_dp in (
            ({"tp_size": 4, "attn_dp_size": 2}, True),
            ({"tp_size": 4, "dp_size": 2, "enable_dp_attention": True}, True),
            ({"tp_size": 4, "dp_size": 2}, False),
            ({"tp_size": 4}, False),
        ):
            with self.subTest(**fields):
                server_args = self.resolve(**fields)
                self.assertFalse(resolution_result(server_args, "enable_dp_attention"))
                readback = server_args.resolved_dict()
                self.assertIs(readback["enable_dp_attention"], attention_dp)
                # A read-back config resolves to the same layout.
                again = self.resolve(
                    **{
                        name: readback[name]
                        for name in (
                            "tp_size",
                            "dp_size",
                            "attn_dp_size",
                            "enable_dp_attention",
                        )
                    }
                )
                self.assertEqual(self.layout(again), self.layout(server_args))

    def test_the_readback_reports_the_width_not_the_joiner_arm(self):
        """`attn_dp_enabled` is also true for an elastic scale joiner at width
        one. The readback reports the configured width instead, so a joiner's
        replicas are not folded into attention-DP groups when it is read back."""
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang.srt.runtime_context import attn_dp_enabled_of

        joiner = ServerArgs(
            model_path="dummy", tp_size=2, dp_size=2, ep_join_mode="scale"
        )
        joiner.resolve_once()
        self.assertTrue(attn_dp_enabled_of(resolving_view(joiner)))
        readback = joiner.resolved_dict()
        self.assertFalse(readback["enable_dp_attention"])
        self.assertEqual(readback["dp_size"], 2)
        self.assertEqual(readback["attn_dp_size"], 1)

    def test_both_cli_spellings_parse(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flag in ("--attn-dp-size", "--attention-data-parallel-size"):
            with self.subTest(flag=flag):
                raw = parser.parse_args(["--model-path", self.path, flag, "2"])
                self.assertEqual(ServerArgs.from_cli_args(raw).attn_dp_size, 2)

    def test_the_deprecated_flag_warns_with_its_replacement(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = ["--model-path", self.path, "--dp-size", "2", "--enable-dp-attention"]
        with self.assertLogs(
            "sglang.srt.arg_groups.argparse_actions", "WARNING"
        ) as logs:
            raw = parser.parse_args(argv)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("--attn-dp-size", logs.output[0])
        self.assertTrue(ServerArgs.from_cli_args(raw).enable_dp_attention)


class TestPublishedAttnDpSize(CustomTestCase):
    """What the runtime reads: the attention-DP switch and the DP rank count."""

    def test_the_published_layout(self):
        for fields, (attn_dp_size, num_dp_ranks) in (
            ({"tp_size": 4, "attn_dp_size": 2}, (2, 2)),
            ({"tp_size": 4, "dp_size": 2, "enable_dp_attention": True}, (2, 2)),
            ({"tp_size": 2, "dp_size": 2}, (1, 2)),
            ({"tp_size": 2}, (1, 1)),
        ):
            with self.subTest(**fields):
                with published_topology(**fields):
                    parallel = get_parallel()
                    self.assertEqual(parallel.attn_dp_size, attn_dp_size)
                    self.assertEqual(parallel.num_dp_ranks, num_dp_ranks)
                    self.assertEqual(parallel.attn_dp_enabled, attn_dp_size > 1)
                    self.assertEqual(
                        parallel.attn_tp_size, fields["tp_size"] // attn_dp_size
                    )

    def test_a_scale_joiner_runs_attention_dp_at_width_one(self):
        with published_topology(tp_size=1, attn_dp_size=1, ep_join_mode="scale"):
            parallel = get_parallel()
            self.assertEqual(parallel.attn_dp_size, 1)
            self.assertEqual(parallel.num_dp_ranks, 1)
            self.assertTrue(parallel.attn_dp_enabled)

    def test_an_overridden_layout_publishes_the_same_leaves(self):
        for fields, width in (
            ({"tp_size": 4, "attn_dp_size": 4}, 4),
            ({"tp_size": 4, "attn_dp_size": 2}, 2),
            ({"tp_size": 4, "dp_size": 2}, 1),
        ):
            with self.subTest(**fields):
                with get_context().override_server_args(**fields):
                    self.assertEqual(get_parallel().attn_dp_size, width)
                    self.assertEqual(get_parallel().attn_dp_enabled, width > 1)
                    self.assertEqual(
                        get_parallel().num_dp_ranks, fields.get("dp_size", width)
                    )


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
