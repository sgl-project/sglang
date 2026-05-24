import unittest

from sglang.srt.models.gemma3_causal import (
    get_gemma3_rope_parameters,
    get_gemma3_text_rope_parameters,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestGemma3RopeParameters(unittest.TestCase):
    def test_nested_default_full_attention_does_not_require_factor(self):
        rope_params = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        }

        full_attention_params = get_gemma3_rope_parameters(
            rope_params, "full_attention", 1000000.0
        )

        self.assertEqual(
            full_attention_params,
            {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        )
        self.assertNotIn("factor", full_attention_params)

    def test_nested_default_text_model_rope_params_do_not_require_factor(self):
        global_rope_params, local_rope_params = get_gemma3_text_rope_parameters(
            {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                },
            },
            rope_local_base_freq=10000.0,
        )

        self.assertEqual(
            global_rope_params,
            {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        )
        self.assertEqual(
            local_rope_params,
            {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        )

    def test_nested_scaled_full_attention_preserves_factor(self):
        rope_params = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            },
        }

        full_attention_params = get_gemma3_rope_parameters(
            rope_params, "full_attention", 1000000.0
        )

        self.assertEqual(
            full_attention_params,
            {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            },
        )

    def test_nested_legacy_type_preserves_scaling_type(self):
        rope_params = {
            "full_attention": {
                "type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            },
        }

        full_attention_params = get_gemma3_rope_parameters(
            rope_params, "full_attention", 1000000.0
        )

        self.assertEqual(full_attention_params["rope_type"], "linear")
        self.assertEqual(full_attention_params["type"], "linear")
        self.assertEqual(full_attention_params["factor"], 8.0)

    def test_flat_legacy_type_preserves_scaling_type(self):
        rope_params = {
            "type": "linear",
            "rope_theta": 1000000.0,
            "factor": 8.0,
        }

        full_attention_params = get_gemma3_rope_parameters(
            rope_params, "full_attention", 1000000.0
        )

        self.assertEqual(full_attention_params["rope_type"], "linear")
        self.assertEqual(full_attention_params["type"], "linear")
        self.assertEqual(full_attention_params["factor"], 8.0)

    def test_missing_layer_type_uses_defaults(self):
        rope_params = {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        }

        sliding_attention_params = get_gemma3_rope_parameters(
            rope_params, "sliding_attention", 10000.0
        )

        self.assertEqual(
            sliding_attention_params,
            {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        )

    def test_none_uses_defaults(self):
        full_attention_params = get_gemma3_rope_parameters(
            None, "full_attention", 1000000.0
        )

        self.assertEqual(
            full_attention_params,
            {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
