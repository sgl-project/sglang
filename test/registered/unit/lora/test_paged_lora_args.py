"""Unit tests for paged LoRA server arguments.

Usage:
    python -m pytest test/registered/unit/lora/test_paged_lora_args.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.test_utils import CustomTestCase


class TestPagedLoRAArgs(CustomTestCase):
    def test_lora_page_rank_size_default_disabled(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertEqual(server_args.lora_page_rank_size, 0)

    def test_lora_pages_default_auto(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertEqual(server_args.lora_pages, 0)

    def test_lora_page_rank_size_from_cli(self):
        server_args = prepare_server_args(
            ["--model-path", "dummy", "--lora-page-rank-size", "8"]
        )
        self.assertEqual(server_args.lora_page_rank_size, 8)

    def test_lora_pages_from_cli(self):
        server_args = prepare_server_args(
            ["--model-path", "dummy", "--lora-pages", "64"]
        )
        self.assertEqual(server_args.lora_pages, 64)


if __name__ == "__main__":
    unittest.main()
