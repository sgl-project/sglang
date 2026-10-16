"""Rank-stabilized adapters must preserve the PEFT scaling rule."""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRsLoRAScaling(unittest.TestCase):
    def scaling(self, rank, **options):
        config = LoRAConfig.from_dict(
            dict(
                peft_type="LORA",
                r=rank,
                lora_alpha=8,
                target_modules=["q_proj"],
                **options,
            )
        )
        return LoRAAdapter(
            "test", config, SimpleNamespace(num_hidden_layers=1), None, None
        ).scaling

    def test_rslora_uses_square_root_rank(self):
        for rank in [4, 8, 64]:
            with self.subTest(rank=rank):
                self.assertAlmostEqual(
                    self.scaling(rank, use_rslora=True), 8 / math.sqrt(rank)
                )

    def test_standard_lora_scaling_is_unchanged(self):
        self.assertEqual(self.scaling(64), 0.125)
        self.assertEqual(self.scaling(64, use_rslora=False), 0.125)


if __name__ == "__main__":
    unittest.main()
