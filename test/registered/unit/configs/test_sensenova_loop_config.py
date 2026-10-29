import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.model_config import is_multimodal_model
from sglang.srt.configs.sensenova_loop import SenseNovaLoopConfig
from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def canonical_config():
    return {
        "model_type": "sensenova_loop",
        "architectures": ["Qwen3_5LoopForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 32,
            "layer_types": [
                "full_attention" if i % 4 == 3 else "linear_attention"
                for i in range(32)
            ],
            "full_attention_interval": 4,
        },
        "loop_settings": {
            "start": 4,
            "end": 28,
            "eval_count": 2,
            "counts": [2],
            "probabilities": [1.0],
            "use_gate": False,
        },
        "mhc": {
            "start": 4,
            "end": 28,
            "streams": 2,
            "iterations": 20,
            "fixed_identity": False,
        },
    }


class TestSenseNovaLoopConfig(CustomTestCase):
    def test_visit_cache_layout(self):
        config = SenseNovaLoopConfig(**canonical_config())
        text = config.get_text_config()
        plan = tuple(range(4)) + tuple(range(4, 28)) * 2 + tuple(range(28, 32))
        self.assertEqual(config.execution_plan, plan)
        self.assertEqual(text.physical_num_hidden_layers, 32)
        self.assertEqual(text.num_hidden_layers, 56)
        self.assertEqual(text.full_attention_layer_ids, list(range(3, 56, 4)))
        self.assertEqual(len(text.linear_layer_ids), 42)
        self.assertTrue(
            set(text.linear_layer_ids).isdisjoint(text.full_attention_layer_ids)
        )
        self.assertEqual(
            sorted(text.linear_layer_ids + text.full_attention_layer_ids),
            list(range(56)),
        )

    def test_layer_types_follow_the_checkpoint(self):
        raw = canonical_config()
        raw["text_config"]["layer_types"][4] = "full_attention"
        config = SenseNovaLoopConfig(**raw)
        self.assertIn(4, config.text_config.full_attention_layer_ids)
        self.assertIn(28, config.text_config.full_attention_layer_ids)
        self.assertNotIn(4, config.text_config.linear_layer_ids)

    def test_roundtrip_does_not_expand_twice(self):
        config = SenseNovaLoopConfig(**canonical_config())
        restored = SenseNovaLoopConfig.from_dict(config.to_dict())
        self.assertEqual(restored.execution_plan, config.execution_plan)
        self.assertEqual(restored.text_config.num_hidden_layers, 56)
        self.assertEqual(restored.text_config.physical_num_hidden_layers, 32)

    def test_gdn_state_and_prefix_cache_are_registered(self):
        config = SenseNovaLoopConfig(**canonical_config())
        spec, text_config = get_linear_attn_config(config)
        self.assertIs(text_config, config.text_config)
        self.assertTrue(spec.uses_mamba_radix_cache)
        self.assertTrue(spec.support_mamba_cache_extra_buffer)
        self.assertEqual(spec.arch_names, ["SenseNovaLoopForConditionalGeneration"])

    def test_registered_config_does_not_need_remote_code(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(canonical_config()))
            config = get_config(directory, trust_remote_code=False)
        self.assertIsInstance(config, SenseNovaLoopConfig)
        self.assertEqual(
            config.architectures, ["SenseNovaLoopForConditionalGeneration"]
        )

    def test_invalid_loop_contracts(self):
        for key, value in (
            ("use_gate", True),
            ("eval_count", 0),
            ("eval_count", True),
            ("counts", [1, 2]),
            ("probabilities", [0.5]),
            ("start", -1),
            ("end", 33),
        ):
            with self.subTest(key=key, value=value):
                raw = canonical_config()
                raw["loop_settings"][key] = value
                with self.assertRaises(ValueError):
                    SenseNovaLoopConfig(**raw)

    def test_invalid_mhc_contracts(self):
        for updates in (
            {"start": 3},
            {"streams": 3},
            {"iterations": 0},
            {"fixed_identity": 1},
            {"pre_scale": float("nan")},
            {"pre_scale": 0},
            {"unknown": True},
        ):
            with self.subTest(updates=updates):
                raw = canonical_config()
                raw["mhc"].update(updates)
                with self.assertRaises(ValueError):
                    SenseNovaLoopConfig(**raw)

    def test_expanded_checkpoint_is_rejected(self):
        raw = canonical_config()
        raw["loop_expanded"] = True
        with self.assertRaises(ValueError):
            SenseNovaLoopConfig(**raw)

    def test_input_config_is_not_modified(self):
        raw = canonical_config()
        raw["text_config"]["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10000000,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        }
        original = copy.deepcopy(raw)
        config = SenseNovaLoopConfig(**raw)
        self.assertEqual(raw, original)
        self.assertFalse(getattr(config, "language_model_only", False))
        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [11, 11, 10]
        )
        self.assertTrue(config.text_config.rope_parameters["mrope_interleaved"])
        self.assertEqual(config.text_config.rope_parameters["rope_theta"], 10000000)

    def test_multimodal_architecture_is_registered(self):
        config = SenseNovaLoopConfig(**canonical_config())
        self.assertTrue(is_multimodal_model(config.architectures))

    def test_image_and_video_positions_follow_qwen3_5(self):
        # Each temporal video patch is represented by its own vision span.
        # Without the Qwen3.5 video-grid dispatch, the second span is misindexed.
        for video in (False, True):
            with self.subTest(video=video):
                span = [10, 102 if video else 101, 102 if video else 101, 11]
                ids = [1] + span + ([2] + span if video else []) + [3]
                kwargs = dict(
                    spatial_merge_size=2,
                    image_token_id=101,
                    video_token_id=102,
                    vision_start_token_id=10,
                    input_ids=torch.tensor([ids]),
                )
                kwargs["video_grid_thw" if video else "image_grid_thw"] = torch.tensor(
                    [[2 if video else 1, 2, 4]]
                )
                actual = get_rope_index(model_type="sensenova_loop", **kwargs)
                expected = get_rope_index(model_type="qwen3_5", **kwargs)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference)
                self.assertEqual(actual[0].shape, (3, 1, len(ids)))
                self.assertFalse(torch.equal(actual[0][0], actual[0][2]))


if __name__ == "__main__":
    unittest.main()
