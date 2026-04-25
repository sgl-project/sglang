import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines.flux_2_nvfp4 import (
    _build_supplemental_safetensors_dir,
)


class TestFlux2Nvfp4Pipeline(unittest.TestCase):
    def test_rebuilds_stale_supplemental_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mixed_file = tmp_path / "flux2-dev-nvfp4-mixed.safetensors"
            non_mixed_file = tmp_path / "flux2-dev-nvfp4.safetensors"
            temp_root = tmp_path / "temp"
            temp_root.mkdir()

            save_file(
                {"foo.weight": torch.zeros((1,), dtype=torch.uint8)},
                str(mixed_file),
            )
            save_file(
                {
                    "foo.weight": torch.zeros((1,), dtype=torch.uint8),
                    "double_blocks.0.txt_mlp.0.input_scale": torch.tensor(
                        3.0, dtype=torch.float32
                    ),
                },
                str(non_mixed_file),
            )

            with patch(
                "sglang.multimodal_gen.runtime.pipelines.flux_2_nvfp4.tempfile.gettempdir",
                return_value=str(temp_root),
            ):
                supp_dir = Path(_build_supplemental_safetensors_dir(str(mixed_file)))
                self.assertTrue((supp_dir / "supplemental-mixed.safetensors").is_file())

                # Simulate a stale cache produced by an older buggy builder.
                save_file({}, str(supp_dir / "supplemental-mixed.safetensors"))

                rebuilt_dir = Path(_build_supplemental_safetensors_dir(str(mixed_file)))

            self.assertEqual(supp_dir, rebuilt_dir)
            with safe_open(
                rebuilt_dir / "supplemental-mixed.safetensors",
                framework="pt",
                device="cpu",
            ) as f:
                self.assertIn("double_blocks.0.txt_mlp.0.input_scale", set(f.keys()))


if __name__ == "__main__":
    unittest.main()
