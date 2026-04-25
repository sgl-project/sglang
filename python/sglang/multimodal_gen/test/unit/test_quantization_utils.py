import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.flux import FluxArchConfig
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
)


class TestQuantizationUtils(unittest.TestCase):
    def test_nvfp4_excludes_flux2_bf16_fallback_layers_after_name_mapping(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "format_version": 1,
                    "layers": {
                        "double_blocks.0.img_mlp.0": {"format": "nvfp4"},
                    },
                }
            )
        }
        tensors = {
            "double_blocks.0.img_mlp.0.weight": torch.zeros(
                (36864, 3072), dtype=torch.uint8
            ),
            "double_blocks.0.img_mlp.0.weight_scale": torch.zeros(
                (36864, 384), dtype=torch.float8_e4m3fn
            ),
            "final_layer.linear.weight": torch.zeros((128, 6144), dtype=torch.bfloat16),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "flux2-dev-nvfp4-mixed.safetensors"
            save_file(tensors, str(path), metadata=metadata)

            quant_config = build_nvfp4_config_from_safetensors_list(
                [str(path)],
                param_names_mapping_dict=FluxArchConfig().param_names_mapping,
            )

        self.assertIsNotNone(quant_config)
        assert quant_config is not None
        self.assertIn("proj_out", quant_config.exclude_modules)
        self.assertNotIn("final_layer.linear", quant_config.exclude_modules)


if __name__ == "__main__":
    unittest.main()
