"""Pipeline-stage embedding ownership for DeepSeek/GLM NextN."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v2 import pp_stage_needs_embedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pp_group(*, rank: int, size: int) -> SimpleNamespace:
    return SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == size - 1)


class TestDeepseekPPStageEmbedding(CustomTestCase):
    def test_without_speculative_decoding_only_the_first_stage_embeds(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self.assertEqual(
                    pp_stage_needs_embedding(_pp_group(rank=rank, size=4), None),
                    rank == 0,
                )

    def test_speculative_decoding_adds_the_last_stage(self):
        wants = [
            pp_stage_needs_embedding(_pp_group(rank=rank, size=4), "EAGLE")
            for rank in range(4)
        ]
        self.assertEqual(wants, [True, False, False, True])

    def test_middle_stages_never_embed(self):
        self.assertFalse(pp_stage_needs_embedding(_pp_group(rank=2, size=4), "EAGLE"))

    def test_without_pipeline_parallelism_the_single_stage_embeds_either_way(self):
        for spec in (None, "EAGLE"):
            with self.subTest(spec=spec):
                self.assertTrue(
                    pp_stage_needs_embedding(_pp_group(rank=0, size=1), spec)
                )


if __name__ == "__main__":
    unittest.main()
