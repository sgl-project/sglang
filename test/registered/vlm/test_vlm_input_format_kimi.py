"""Kimi-VL case of test_vlm_input_format.py, kept on H100: Kimi-VL-A3B plus its
HF vision tower does not fit a 32 GB card."""

import unittest

import test_vlm_input_format as base
from transformers import AutoModel

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=75, stage="base-b", runner_config="1-gpu-large")


class TestKimiVLImageUnderstandsImage(base.VLMInputTestBase, CustomTestCase):
    model_path = "moonshotai/Kimi-VL-A3B-Instruct"
    chat_template = "kimi-vl"

    @classmethod
    def _init_visual(cls):
        import inspect

        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config = AutoConfig.from_pretrained(cls.model_path, trust_remote_code=True)

        # Transformers v5 auto-populates rope_scaling with
        # {"rope_theta": ..., "rope_type": "default"} even when the original
        # config had rope_scaling: null. The remote KimiVL code branches on
        # `if self.config.rope_scaling is None` so we must reset it.
        tc = getattr(config, "text_config", None)
        if tc is not None:
            rs = getattr(tc, "rope_scaling", None)
            if isinstance(rs, dict) and rs.get("rope_type") == "default":
                tc.rope_scaling = None

        # Transformers v5 calls tie_weights(recompute_mapping=False) in
        # post_init, but KimiVL's tie_weights doesn't accept that kwarg.
        auto_map = getattr(config, "auto_map", {})
        model_ref = auto_map.get("AutoModel")
        if model_ref:
            model_cls = get_class_from_dynamic_module(model_ref, cls.model_path)
            orig_tie = model_cls.tie_weights
            if "recompute_mapping" not in inspect.signature(orig_tie).parameters:

                def _patched_tie(self, **kwargs):
                    return orig_tie(self)

                model_cls.tie_weights = _patched_tie

        model = AutoModel.from_pretrained(
            cls.model_path, config=config, trust_remote_code=True
        )
        cls.vision_tower = model.vision_tower.eval().to(cls.device)
        cls.mm_projector = model.multi_modal_projector.eval().to(cls.device)
        _vt_dtype = next(cls.vision_tower.parameters()).dtype

        cls.visual = lambda tokenizer_output: cls.mm_projector(
            cls.vision_tower(
                pixel_values=tokenizer_output["pixel_values"].to(_vt_dtype),
                grid_hws=tokenizer_output["image_grid_hws"],
            )
        )

    def _processor_output_image_data(self, processor_output):
        return dict(processor_output, format="processor_output")


if __name__ == "__main__":
    unittest.main()
