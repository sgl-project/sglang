# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageArchConfig
from sglang.multimodal_gen.runtime.loader.checkpoint_name_resolver import (
    CheckpointNameResolver,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)


class TestCheckpointNameResolver(unittest.TestCase):
    def _qwen_mapping(self):
        return get_param_names_mapping(QwenImageArchConfig().param_names_mapping)

    def test_qwenimage_to_out_direct_rename(self):
        mapped, _, _ = self._qwen_mapping()("transformer_blocks.0.attn.to_out.weight")

        self.assertEqual(mapped, "transformer_blocks.0.attn.to_out.0.weight")

    def test_qwenimage_added_qkv_fused_mapping(self):
        state_dict = {
            "transformer_blocks.0.attn.add_q_proj.weight": torch.ones(1, 2),
            "transformer_blocks.0.attn.add_k_proj.weight": torch.ones(1, 2) * 2,
            "transformer_blocks.0.attn.add_v_proj.weight": torch.ones(1, 2) * 3,
        }

        custom_sd, _, resolver = hf_to_custom_state_dict(
            state_dict, self._qwen_mapping(), return_resolver=True
        )

        fused = custom_sd["transformer_blocks.0.attn.to_added_qkv.weight"]
        self.assertTrue(
            torch.equal(fused, torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
        )
        shards = resolver.source_param_shards_by_target[
            "transformer_blocks.0.attn.to_added_qkv.weight"
        ]
        self.assertEqual([shard.source_name for shard in shards], list(state_dict))

    def test_incomplete_fused_mapping_raises(self):
        state_dict = {
            "transformer_blocks.0.attn.add_q_proj.weight": torch.ones(1, 2),
            "transformer_blocks.0.attn.add_k_proj.weight": torch.ones(1, 2),
        }

        with self.assertRaisesRegex(ValueError, "missing shard"):
            hf_to_custom_state_dict(state_dict, self._qwen_mapping())

    def test_metadata_key_map_uses_checkpoint_side_candidates(self):
        resolver = CheckpointNameResolver(self._qwen_mapping())

        metadata_map = resolver.metadata_key_map(
            ["transformer_blocks.0.attn.add_q_proj.wtscale"], ("wtscale",)
        )

        self.assertEqual(
            metadata_map["wtscale"]["transformer_blocks.0.attn.to_added_qkv"],
            "transformer_blocks.0.attn.add_q_proj.wtscale",
        )

    def test_ignored_tensor_stays_ignored(self):
        custom_sd, _ = hf_to_custom_state_dict(
            {"transformer_blocks.0.attn.to_qkv.wtscale": torch.ones(())},
            self._qwen_mapping(),
        )

        self.assertEqual(custom_sd, {})


if __name__ == "__main__":
    unittest.main()
