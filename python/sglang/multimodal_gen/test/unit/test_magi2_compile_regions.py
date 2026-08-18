# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    Magi2BlockGridAttention,
)
from sglang.multimodal_gen.runtime.layers.moe_multihead import Magi2MultiHeadExperts
from sglang.multimodal_gen.runtime.models.dits.magi2_preview import Magi2PreviewDiT
from sglang.multimodal_gen.runtime.models.dits.magi2_refiner import Magi2RefinerDiT


class TestCompileRegions(unittest.TestCase):
    def test_preview_compiles_whole_blocks_only(self):
        conditions = Magi2PreviewDiT._compile_conditions
        self.assertTrue(conditions)
        self.assertTrue(any(c("blocks.0", object()) for c in conditions))
        # A submodule would compile a region nested inside an outer one.
        self.assertFalse(any(c("blocks.0.attention", object()) for c in conditions))
        self.assertFalse(any(c("blocks", object()) for c in conditions))

    def test_expert_forward_is_opaque_to_dynamo(self):
        # Autotuning these GEMMs asks more shared memory than sm90 has, so a
        # compiled region containing them fails to build.
        self.assertTrue(Magi2MultiHeadExperts.forward._torchdynamo_disable)

    def test_refiner_compiles_whole_blocks_only(self):
        # Both DiTs are compiled, so an empty condition list would make the
        # framework's regional path raise on the refiner pass.
        conditions = Magi2RefinerDiT._compile_conditions
        self.assertTrue(conditions)
        self.assertTrue(any(c("blocks.0", object()) for c in conditions))
        self.assertFalse(any(c("blocks.0.attention", object()) for c in conditions))

    def test_block_grid_attention_is_opaque_to_dynamo(self):
        # It compiles its own mask per bucket; an outer region would recompile it
        # per sequence length instead.
        self.assertTrue(Magi2BlockGridAttention.forward._torchdynamo_disable)


if __name__ == "__main__":
    unittest.main()
