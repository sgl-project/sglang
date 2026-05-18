# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.omni.configs.registry import resolve_omni_model_key
from sglang.omni.configs.sensenova_u1 import (
    SenseNovaU1OmniPlugin,
    _build_default_generation_backend,
    _build_diffusion_server_kwargs,
    _parse_diffusion_server_args,
    _resolve_omni_max_concurrent_generations,
)
from sglang.omni.core.protocol import OmniInputSegment, OmniRequest


class TestSenseNovaU1OmniConfig(unittest.TestCase):
    def test_registry_resolves_sensenova_u1_aliases(self):
        self.assertEqual("sensenova-u1", resolve_omni_model_key(None))
        self.assertEqual(
            "sensenova-u1",
            resolve_omni_model_key("sensenova/SenseNova-U1-8B-MoT"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported omni model"):
            resolve_omni_model_key("other-model")

    def test_sampling_payload_request_metadata_is_split_before_sampling_build(self):
        plugin = SenseNovaU1OmniPlugin()
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="describe"),),
            sampling_params={
                "task": "vlm",
                "max_new_tokens": 4,
                "max_interleave_images": 2,
                "think": "true",
                "num_steps": 3,
            },
        )

        normalized = plugin.normalize_request(request)

        self.assertEqual("vlm", normalized.mode)
        self.assertEqual(4, normalized.metadata["max_new_tokens"])
        self.assertEqual(2, normalized.max_images)
        self.assertIsNone(normalized.max_text_segments_after_media)
        self.assertTrue(normalized.think)
        self.assertEqual("vlm", normalized.sampling_params.omni_generation_mode)
        self.assertTrue(normalized.sampling_params.think_mode)
        self.assertEqual(3, normalized.sampling_params.num_inference_steps)

    def test_interleave_defaults_to_one_visible_text_after_media(self):
        plugin = SenseNovaU1OmniPlugin()
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="hi"),),
            mode="interleave",
            max_text_segments=3,
            sampling_params={},
        )

        normalized = plugin.normalize_request(request)

        self.assertEqual("interleave", normalized.mode)
        self.assertEqual(3, normalized.max_text_segments)
        self.assertEqual(1, normalized.max_text_segments_after_media)

    def test_sampling_defaults_follow_u1_official_image_generation(self):
        plugin = SenseNovaU1OmniPlugin()
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            sampling_params={},
        )

        normalized = plugin.normalize_request(request)
        params = normalized.sampling_params

        self.assertEqual(50, params.num_inference_steps)
        self.assertEqual(4.0, params.cfg_text_scale)
        self.assertEqual(1.0, params.cfg_img_scale)
        self.assertEqual([0.0, 1.0], params.cfg_interval)
        self.assertEqual("none", params.cfg_renorm_type)
        self.assertEqual(0.02, params.t_eps)

    def test_sampling_params_must_be_object(self):
        plugin = SenseNovaU1OmniPlugin()
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            sampling_params="bad",
        )

        with self.assertRaisesRegex(ValueError, "sampling_params must be an object"):
            plugin.normalize_request(request)

    def test_diffusion_source_engine_args_accept_json_and_cli_string(self):
        cases = [
            '{"num_gpus": 1, "tp_size": 1, "attention_backend": "fa"}',
            "--num-gpus 1 --tp-size 1 --attention-backend fa",
        ]

        for raw_args in cases:
            with self.subTest(raw_args=raw_args):
                parsed = _parse_diffusion_server_args(raw_args)

                self.assertEqual(1, parsed["num_gpus"])
                self.assertEqual(1, parsed["tp_size"])
                self.assertEqual("fa", parsed["attention_backend"])

    def test_diffusion_source_engine_args_keep_u1_pipeline(self):
        kwargs = _build_diffusion_server_kwargs(
            SimpleNamespace(
                model_path="sensenova-u1",
                diffusion_server_args='{"pipeline_class_name": "OtherPipeline"}',
            )
        )

        self.assertEqual("sensenova-u1", kwargs["model_path"])
        self.assertEqual("SenseNovaU1Pipeline", kwargs["pipeline_class_name"])

    def test_default_generation_backend_imports_u1_pipeline(self):
        backend = _build_default_generation_backend(
            SimpleNamespace(model_path="sensenova-u1", diffusion_server_args=None)
        )

        self.assertEqual("sensenova-u1", backend.server_args.model_path)
        self.assertEqual(
            "SenseNovaU1Pipeline",
            backend.server_args.pipeline_class_name,
        )

    def test_omni_generation_admission_limit_defaults_to_one(self):
        self.assertEqual(1, _resolve_omni_max_concurrent_generations(None))
        self.assertEqual(
            2,
            _resolve_omni_max_concurrent_generations(
                SimpleNamespace(omni_max_concurrent_generations=2)
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolve_omni_max_concurrent_generations(
                SimpleNamespace(omni_max_concurrent_generations=0)
            )


if __name__ == "__main__":
    unittest.main()
