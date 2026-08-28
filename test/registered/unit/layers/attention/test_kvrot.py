"""Unit tests for Q/K block Hadamard rotation helpers — CPU only."""

import argparse
import unittest

import torch

from sglang.srt.layers.attention.kvrot import apply_block_hadamard_rotation
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestApplyBlockHadamardRotation(CustomTestCase):
    def test_rejects_non_positive_block_size(self):
        x = torch.randn(2, 4, 16)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            apply_block_hadamard_rotation(x, block_size=0)

    def test_rejects_block_size_that_does_not_divide_head_dim(self):
        x = torch.randn(2, 4, 16)
        with self.assertRaisesRegex(ValueError, "must divide head_dim"):
            apply_block_hadamard_rotation(x, block_size=12)

    def test_cli_parses_qkrot_block(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model", "dummy", "--qkrot-block", "16"])
        self.assertEqual(args.qkrot_block, 16)


if __name__ == "__main__":
    unittest.main()
