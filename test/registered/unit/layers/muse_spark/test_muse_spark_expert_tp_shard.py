"""Expert-TP checkpoint sharding must tile each expert's intermediate dim exactly."""

import unittest

import torch

from sglang.srt.models.muse_spark import MuseSparkTextForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

_shard = MuseSparkTextForCausalLM._shard_expert_intermediate


class TestMuseSparkExpertTPShard(CustomTestCase):
    def test_gate_up_halves_are_sharded_separately(self):
        experts, inter, hidden, size = 4, 64, 16, 4
        for name in (".experts.gate_up_proj", ".experts.gate_up_proj_weight_scale"):
            full = torch.randn(experts, 2 * inter, hidden)
            shards = [_shard(full, "l.mlp" + name, r, size) for r in range(size)]
            for shard in shards:
                self.assertEqual(shard.shape, (experts, 2 * inter // size, hidden))
            gate = torch.cat([s.chunk(2, dim=1)[0] for s in shards], dim=1)
            up = torch.cat([s.chunk(2, dim=1)[1] for s in shards], dim=1)
            torch.testing.assert_close(torch.cat([gate, up], dim=1), full)

    def test_down_is_sharded_on_last_dim(self):
        experts, inter, hidden, size = 4, 64, 16, 8
        for name in (".experts.down_proj", ".experts.down_proj_weight_scale"):
            full = torch.randn(experts, hidden, inter)
            shards = [_shard(full, "l.mlp" + name, r, size) for r in range(size)]
            torch.testing.assert_close(torch.cat(shards, dim=-1), full)

    def test_per_expert_scalars_are_replicated(self):
        for name in (
            ".experts.gate_up_proj_weight_scale_2",
            ".experts.down_proj_weight_scale_2",
            ".experts.gate_up_proj_input_scale",
            ".experts.down_proj_input_scale",
        ):
            full = torch.randn(4, 2)
            self.assertIs(_shard(full, "l.mlp" + name, 1, 8), full)


if __name__ == "__main__":
    unittest.main()
