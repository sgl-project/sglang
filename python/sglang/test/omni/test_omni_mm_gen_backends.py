# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    build_sensenova_u1_sampling_params,
)
from sglang.omni.backends.mm_gen.pipeline_executor_backend import (
    PipelineExecutorBackend,
)
from sglang.omni.backends.mm_gen.pipeline_forward_backend import (
    DirectPipelineForwardBackend,
    require_generated_segment,
)
from sglang.omni.core.protocol import GeneratedSegment, OmniInputSegment, OmniRequest


class TestOmniMMGenBackends(unittest.TestCase):
    def test_pipeline_backend_passes_context_ops_through_req_field(self):
        pipeline = _FakePipeline()
        server_args = SimpleNamespace()
        backend = DirectPipelineForwardBackend(
            pipeline=pipeline,
            server_args=server_args,
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
        self.assertIs(context_ops, pipeline.batch.omni_context_ops)
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

    def test_pipeline_executor_backend_drives_stages_directly(self):
        executor = _FakeExecutor()
        server_args = SimpleNamespace()
        stages = [object()]
        backend = PipelineExecutorBackend(
            executor=executor,
            stages=stages,
            server_args=server_args,
        )
        context_ops = SimpleNamespace(metadata={"session": "s1"})
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="revise"),),
            sampling_params=build_sensenova_u1_sampling_params(
                {"num_inference_steps": 2}
            ),
        )

        segment = backend.generate_segment(request, context_ops)

        self.assertEqual("image", segment.type)
        self.assertEqual("executor-image", segment.image)
        self.assertEqual(stages, executor.stages)
        self.assertEqual(server_args, executor.server_args)
        self.assertIs(context_ops, executor.batch.omni_context_ops)

    def test_pipeline_backend_requires_generated_segment_contract(self):
        with self.assertRaisesRegex(TypeError, "GeneratedSegment"):
            require_generated_segment(SimpleNamespace(type="image", image="legacy"))


class _FakePipeline:
    def __init__(self):
        self.batch = None
        self.server_args = None

    def forward(self, batch, server_args):
        self.batch = batch
        self.server_args = server_args
        batch.generated_segment = GeneratedSegment(
            type="image",
            image="image-bytes",
            commit_payload="commit-image",
        )
        return batch


class _FakeExecutor:
    def __init__(self):
        self.stages = None
        self.batch = None
        self.server_args = None

    def execute_with_profiling(self, stages, batch, server_args):
        self.stages = stages
        self.batch = batch
        self.server_args = server_args
        batch.generated_segment = GeneratedSegment(
            type="image",
            image="executor-image",
        )
        return batch


if __name__ == "__main__":
    unittest.main()
