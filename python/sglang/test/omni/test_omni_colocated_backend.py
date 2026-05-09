# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    build_sensenova_u1_sampling_params,
)
from sglang.omni.backends.colocated import ColocatedPipelineBackend
from sglang.omni.protocol import GeneratedSegment, OmniInputSegment, OmniRequest


class TestOmniColocatedBackend(unittest.TestCase):
    def test_pipeline_backend_passes_context_ops_through_req_extra(self):
        pipeline = _FakePipeline()
        server_args = SimpleNamespace()
        backend = ColocatedPipelineBackend(
            pipeline=pipeline,
            server_args=server_args,
            context_ops_extra_key="ctx",
            output_extra_key="generated",
        )
        context_ops = SimpleNamespace(metadata={"session": "s0"})
        request = OmniRequest(
            messages=(
                OmniInputSegment(type="text", text="draw"),
                OmniInputSegment(type="text", text="then describe"),
            ),
            sampling_params=build_sensenova_u1_sampling_params(
                {"num_inference_steps": 2}
            ),
        )

        segment = backend.generate_segment(request, context_ops)

        self.assertEqual("image", segment.type)
        self.assertEqual("image-bytes", segment.image)
        self.assertEqual("commit-image", segment.commit_payload)
        self.assertEqual(server_args, pipeline.server_args)
        self.assertIs(context_ops, pipeline.batch.extra["ctx"])
        self.assertEqual(
            [
                {"type": "text", "text": "draw"},
                {"type": "text", "text": "then describe"},
            ],
            pipeline.batch.extra["omni_messages"],
        )
        self.assertEqual(
            {"session": "s0"},
            pipeline.batch.extra["omni_context_metadata"],
        )
        self.assertEqual("draw\nthen describe", pipeline.batch.prompt)

    def test_pipeline_backend_rejects_missing_generated_segment(self):
        backend = ColocatedPipelineBackend(
            pipeline=_EmptyPipeline(),
            server_args=SimpleNamespace(),
            context_ops_extra_key="ctx",
            output_extra_key="generated",
        )
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="draw"),),
            sampling_params=build_sensenova_u1_sampling_params(
                {"num_inference_steps": 2}
            ),
        )

        with self.assertRaisesRegex(ValueError, "did not set extra"):
            backend.generate_segment(request, SimpleNamespace(metadata={}))


class _FakePipeline:
    def __init__(self):
        self.batch = None
        self.server_args = None

    def forward(self, batch, server_args):
        self.batch = batch
        self.server_args = server_args
        batch.extra["generated"] = GeneratedSegment(
            type="image",
            image="image-bytes",
            commit_payload="commit-image",
        )


class _EmptyPipeline:
    def forward(self, batch, server_args):
        del batch, server_args


if __name__ == "__main__":
    unittest.main()
