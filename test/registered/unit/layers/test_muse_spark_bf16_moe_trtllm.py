"""Streamed weight updates into Muse Spark BF16 experts after the TRT-LLM shuffle.

On SM10x the experts are shuffled into the TRT-LLM BlockMajorK layout after the
first load. An RL weight update then writes each expert again; writing it in the
checkpoint layout crashed (``start (0) + length (128) exceeds dimension size``),
and a permutation cache kept on the GPU was zeroed by a memory-saver pause/resume
and scrambled the update.
"""

import unittest

import torch

from sglang.srt.layers.muse_spark_bf16_moe import (
    MuseSpark1xBf16Experts,
    _trtllm_bf16_moe_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")


def _kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return _trtllm_bf16_moe_supported(torch.device("cuda", 0))


@unittest.skipUnless(
    _kernel_available(), "needs SM10x and FlashInfer's TRT-LLM BF16 MoE"
)
class TestMuseSparkBf16ExpertsTrtllmUpdate(CustomTestCase):
    def test_online_update_after_trtllm_shuffle(self):
        """A streamed expert update after the first load must not crash and must
        leave the shuffled weights equal to a cold load of the new weights."""
        torch.manual_seed(0)
        device = torch.device("cuda", 0)
        num_experts, hidden, intermediate = 4, 512, 256

        def build(gate_up, down):
            experts = MuseSpark1xBf16Experts(
                num_experts=2 * num_experts,
                num_local_experts=num_experts,
                local_expert_offset=num_experts,
                expert_hidden_size=hidden,
                intermediate_size=intermediate,
                top_k=2,
            ).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                experts.gate_up_proj.copy_(gate_up)
                experts.down_proj.copy_(down)
            experts.prepare_trtllm_weights(None)
            return experts

        def randn(*shape):
            return torch.randn(*shape, device=device, dtype=torch.bfloat16)

        gate_up = randn(num_experts, 2 * intermediate, hidden)
        down = randn(num_experts, hidden, intermediate)
        gate, up, new_down = (
            randn(intermediate, hidden),
            randn(intermediate, hidden),
            randn(hidden, intermediate),
        )
        local_id, untouched_global_id = 1, 0

        for shards in (("w1", "w2", "w3"), ("w1",)):
            with self.subTest(shards=shards):
                experts = build(gate_up, down)
                expected_gate_up, expected_down = gate_up.clone(), down.clone()
                for shard_id in shards:
                    param, weight = {
                        "w1": (experts.gate_up_proj, gate),
                        "w2": (experts.down_proj, new_down),
                        "w3": (experts.gate_up_proj, up),
                    }[shard_id]
                    experts.weight_loader(
                        param,
                        weight,
                        f"experts.{shard_id}",
                        shard_id=shard_id,
                        expert_id=num_experts + local_id,
                    )
                    # Global experts outside this rank's slice are ignored.
                    experts.weight_loader(
                        param,
                        weight,
                        f"experts.{shard_id}",
                        shard_id=shard_id,
                        expert_id=untouched_global_id,
                    )
                    if shard_id == "w2":
                        expected_down[local_id] = new_down
                    else:
                        offset = 0 if shard_id == "w1" else intermediate
                        expected_gate_up[local_id, offset : offset + intermediate] = (
                            weight
                        )

                cold = build(expected_gate_up, expected_down)
                self.assertTrue(torch.equal(experts.gate_up_proj, cold.gate_up_proj))
                self.assertTrue(torch.equal(experts.down_proj, cold.down_proj))
                # A colocated engine's memory-saver resume zeroes GPU memory allocated
                # while loading weights, so the cached permutations must live on the host.
                for indices in experts._trtllm_permute_cache.values():
                    self.assertEqual(indices.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
