"""CPU control-flow coverage for Bailing's router kernel opt-out."""

import argparse
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt import runtime_context
from sglang.srt.models import bailing_moe_v3
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBailingRouter(unittest.TestCase):
    def test_cli_controls_router_kernel(self):
        """The CLI opt-out reaches the gate while preserving default matvec."""
        runtime_context.reset_context()
        self.addCleanup(runtime_context.reset_context)
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        config = SimpleNamespace(num_experts=2, hidden_size=4)
        hidden_states = mock.Mock(is_cuda=True, shape=(2, 4))
        with (
            mock.patch.object(
                bailing_moe_v3,
                "router_gate_matvec",
                return_value=mock.sentinel.matvec_output,
            ),
            mock.patch.object(
                bailing_moe_v3.F, "linear", return_value=mock.sentinel.linear_output
            ),
        ):
            for flags, expected in (
                ([], mock.sentinel.matvec_output),
                (["--disable-router-gate-matvec"], mock.sentinel.linear_output),
            ):
                with self.subTest(flags=flags):
                    server_args = ServerArgs.from_cli_args(
                        parser.parse_args(
                            ["--model-path", "dummy", "--device", "cpu", *flags]
                        )
                    )
                    runtime_context.publish(server_args, role="test")
                    gate = bailing_moe_v3.BailingMoEGate(
                        config, params_dtype=torch.float32
                    )
                    self.assertIs(gate(hidden_states), expected)


if __name__ == "__main__":
    unittest.main()
