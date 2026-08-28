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

    def test_rejects_negative_page_rank_size(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_lora=True,
            max_lora_rank=8,
            lora_target_modules=["q_proj"],
            lora_page_rank_size=-1,
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            server_args.check_lora_server_args()

    def test_rejects_negative_page_count(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_lora=True,
            max_lora_rank=8,
            lora_target_modules=["q_proj"],
            lora_pages=-1,
        )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            server_args.check_lora_server_args()

    def test_explicit_pages_require_page_rank_size(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_lora=True,
            max_lora_rank=8,
            lora_target_modules=["q_proj"],
            lora_pages=4,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            server_args.check_lora_server_args()

    def test_paged_lora_rejects_virtual_experts(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_lora=True,
            max_lora_rank=8,
            lora_target_modules=["q_proj"],
            lora_page_rank_size=8,
            lora_use_virtual_experts=True,
        )
        with self.assertRaisesRegex(ValueError, "classic fused_moe_lora"):
            server_args.check_lora_server_args()


if __name__ == "__main__":
    unittest.main()
