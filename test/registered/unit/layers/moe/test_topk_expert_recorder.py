"""Keep TRT-LLM routing observable when expert recording is enabled."""

import unittest

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.runtime_context import get_context, get_exec, get_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTopKExpertRecorder(CustomTestCase):
    def test_trtllm_materializes_routes_for_expert_recording(self):
        for recorder_mode in (None, "stat", "stat_approx", "per_pass", "per_token"):
            with self.subTest(recorder_mode=recorder_mode):
                with (
                    get_context().override_server_args(
                        model_path="dummy", device="cpu"
                    ),
                    get_model().override(shared_experts=SharedExpertsConfig()),
                    get_exec().moe.override(
                        expert_distribution_recorder_mode=recorder_mode
                    ),
                ):
                    router = TopK(top_k=2, layer_id=0)
                    router.bind_runner_backend(
                        MoeRunnerBackend.FLASHINFER_TRTLLM,
                        routing_method_type=RoutingMethodType.Renormalize,
                    )
                    expected = (
                        TopKOutputFormat.BYPASSED
                        if recorder_mode is None
                        else TopKOutputFormat.STANDARD
                    )
                    self.assertEqual(router.topk_config.output_format, expected)


if __name__ == "__main__":
    unittest.main()
