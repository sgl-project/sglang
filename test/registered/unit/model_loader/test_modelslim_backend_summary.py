import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_loader.checkpoint_quantization import is_modelslim_backend_summary
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimSummary(unittest.TestCase):
    def test_backend_only_and_inline_descriptions(self):
        self.assertTrue(is_modelslim_backend_summary({"quant_method": "ascend"}))
        self.assertTrue(is_modelslim_backend_summary({"quant_method": "modelslim"}))
        self.assertFalse(
            is_modelslim_backend_summary(
                {
                    "quant_method": "ascend",
                    "layers.0.ffn.experts.0.w1.weight": "W4A8_DYNAMIC",
                }
            )
        )
        self.assertFalse(is_modelslim_backend_summary({"quant_method": "fp8"}))

    def test_backend_marker_reads_per_layer_description(self):
        from sglang.srt.model_loader import weight_utils

        weight = "model.layers.0.mlp.experts.0.gate_proj.weight"

        class Descriptor:
            get_config_filenames = staticmethod(
                lambda: ["quant_model_description.json"]
            )
            from_config = staticmethod(dict)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(
                weight_utils, "get_quantization_config", return_value=Descriptor
            ),
            patch.object(weight_utils, "_maybe_attach_checkpoint_tensor_names"),
        ):
            args = SimpleNamespace(
                model_path=folder,
                is_draft_model=False,
                quantization="modelslim",
                hf_config={"quantization_config": {"quant_method": "ascend"}},
            )
            desc = Path(folder) / "quant_model_description.json"
            desc.write_text(json.dumps({weight: "W4A8_DYNAMIC"}))
            load = SimpleNamespace(model_loader_extra_config=None)
            self.assertEqual(
                weight_utils.get_quant_config(args, load, {})[weight], "W4A8_DYNAMIC"
            )
            desc.unlink()
            with self.assertRaisesRegex(ValueError, "Cannot find"):
                weight_utils.get_quant_config(args, load, {})
            args.hf_config["quantization_config"][weight] = "FLOAT"
            self.assertEqual(
                weight_utils.get_quant_config(args, load, {})[weight], "FLOAT"
            )


if __name__ == "__main__":
    unittest.main()
