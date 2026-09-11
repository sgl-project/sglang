"""Weight-name mapping GLM-5.3-Flash needs for Quark checkpoints — CPU only."""

import unittest

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MAPPER = Glm5NextForConditionalGeneration.hf_to_sglang_mapper


class TestHfToSglangMapper(CustomTestCase):
    def test_language_tower_prefix_is_stripped(self):
        self.assertEqual(
            _MAPPER.apply_list(["model.language_model.layers.0.mlp.down_proj"]),
            ["model.layers.0.mlp.down_proj"],
        )

    def test_vision_tower_prefix_is_rewritten(self):
        self.assertEqual(
            _MAPPER.apply_list(["model.visual.blocks.0.attn.proj"]),
            ["visual.blocks.0.attn.proj"],
        )

    def test_fused_vision_qkv_maps_to_runtime_module_name(self):
        # The checkpoint ships a pre-fused `qkv`; the runtime module is
        # `qkv_proj`. Quantization configs address layers by module name, so an
        # unmapped exclusion would silently fail to match.
        self.assertEqual(
            _MAPPER.apply_list(["model.visual.blocks.0.attn.qkv"]),
            ["visual.blocks.0.attn.qkv_proj"],
        )

    def test_mapper_rewrites_dict_keys(self):
        mapped = _MAPPER.apply_dict(
            {"model.language_model.layers.7.self_attn.q_a_proj": {"marker": 1}}
        )
        self.assertEqual(mapped, {"model.layers.7.self_attn.q_a_proj": {"marker": 1}})


class TestBlockFp8ScaleNameMapping(CustomTestCase):
    """Quark writes `.weight_scale`; the FP8 linear method wants `.weight_scale_inv`."""

    @staticmethod
    def _mapper(params_dict):
        def maybe_map_fp8_block_scale_name(name: str) -> str:
            if name.endswith(".weight_scale"):
                candidate = name.removesuffix(".weight_scale") + ".weight_scale_inv"
                if candidate in params_dict:
                    return candidate
            return name

        return maybe_map_fp8_block_scale_name

    def test_scale_is_renamed_when_runtime_param_exists(self):
        fn = self._mapper({"model.layers.7.self_attn.q_a_proj.weight_scale_inv": None})
        self.assertEqual(
            fn("model.layers.7.self_attn.q_a_proj.weight_scale"),
            "model.layers.7.self_attn.q_a_proj.weight_scale_inv",
        )

    def test_scale_is_left_alone_when_runtime_param_absent(self):
        # MXFP4 layers keep `.weight_scale`; renaming them would break loading.
        fn = self._mapper({"model.layers.3.mlp.experts.0.up_proj.weight_scale": None})
        self.assertEqual(
            fn("model.layers.3.mlp.experts.0.up_proj.weight_scale"),
            "model.layers.3.mlp.experts.0.up_proj.weight_scale",
        )

    def test_non_scale_names_are_untouched(self):
        fn = self._mapper({"model.layers.7.self_attn.q_a_proj.weight_scale_inv": None})
        self.assertEqual(
            fn("model.layers.7.self_attn.q_a_proj.weight"),
            "model.layers.7.self_attn.q_a_proj.weight",
        )


if __name__ == "__main__":
    unittest.main()
