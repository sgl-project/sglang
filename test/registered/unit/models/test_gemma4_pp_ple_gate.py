"""Gemma4 PLE variants under pipeline parallelism refuse CUDA graphs."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import Gemma4TextConfig

from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
from sglang.srt.models import gemma4_causal
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _GatePassed(Exception):
    pass


def _ple_config():
    return Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=8,
        max_position_embeddings=32,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
    )


class TestGemma4PipelinePleGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _build_second_stage_model(self, argv):
        server_args = ServerArgs.from_cli_args(
            self.parser.parse_args(["--model", "dummy"] + argv)
        )
        server_args.resolve_once()
        # The dummy model returns from resolution before the CUDA graph hook.
        parse_cuda_graph_config(server_args)
        reset_context()
        self.addCleanup(reset_context)
        publish(server_args, role="test")

        pp_group = SimpleNamespace(world_size=2, is_first_rank=True)
        with (
            patch.object(
                gemma4_causal,
                "get_parallel",
                return_value=SimpleNamespace(pp_group=pp_group),
            ),
            patch.object(
                gemma4_causal,
                "Gemma4TextScaledWordEmbedding",
                side_effect=_GatePassed,
            ),
        ):
            gemma4_causal.Gemma4TextModel(_ple_config())

    def test_graphs_disabled_on_both_phases_admit_pp(self):
        """Graphs disabled through the per-phase backend flags admit PP, not
        only the deprecated --disable-cuda-graph spelling."""
        for argv in (
            [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ],
            ["--disable-cuda-graph"],
        ):
            with self.subTest(argv=argv), self.assertRaises(_GatePassed):
                self._build_second_stage_model(argv)

    def test_default_graphs_reject_pp(self):
        with self.assertRaisesRegex(ValueError, "per-layer-input"):
            self._build_second_stage_model([])


if __name__ == "__main__":
    unittest.main()
