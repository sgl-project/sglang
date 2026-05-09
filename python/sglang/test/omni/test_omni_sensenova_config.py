# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.omni.configs.sensenova_u1 import (
    SenseNovaU1OmniPlugin,
    _build_diffusion_server_kwargs,
    _parse_diffusion_server_args,
)
from sglang.omni.protocol import OmniInputSegment, OmniRequest


class TestSenseNovaU1OmniConfig(unittest.TestCase):
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
        self.assertTrue(normalized.think)
        self.assertEqual(3, normalized.sampling_params.num_inference_steps)

    def test_diffusion_source_engine_args_accepts_json(self):
        parsed = _parse_diffusion_server_args(
            '{"num_gpus": 1, "tp_size": 1, "attention_backend": "fa"}'
        )

        self.assertEqual(1, parsed["num_gpus"])
        self.assertEqual(1, parsed["tp_size"])
        self.assertEqual("fa", parsed["attention_backend"])

    def test_diffusion_source_engine_args_accepts_cli_string(self):
        parsed = _parse_diffusion_server_args(
            "--num-gpus 1 --tp-size 1 --attention-backend fa"
        )

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


if __name__ == "__main__":
    unittest.main()
