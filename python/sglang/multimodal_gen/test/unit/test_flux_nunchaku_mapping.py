# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.flux import FluxTransformer2DModel


class TestFluxNunchakuMapping(unittest.TestCase):
    def test_flux1_single_block_projections(self):
        mapping = get_param_names_mapping(FluxTransformer2DModel.param_names_mapping)
        cases = {
            "single_transformer_blocks.3.qkv_proj.lora_up": "single_transformer_blocks.3.attn.to_qkv.proj_up",
            "single_transformer_blocks.3.qkv_proj.qweight": "single_transformer_blocks.3.attn.to_qkv.qweight",
            "single_transformer_blocks.3.out_proj.smooth": "single_transformer_blocks.3.attn.to_out.0.smooth_factor",
            "transformer_blocks.0.qkv_proj.lora_down": "transformer_blocks.0.attn.to_qkv.proj_down",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual(mapping(source)[0], target)


if __name__ == "__main__":
    unittest.main()
