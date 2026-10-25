# SPDX-License-Identifier: Apache-2.0
import unittest

import requests
import test_dsa_glm52_nvfp4_tp_mtp as base

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=400,
    stage="base-c",
    runner_config="4-gpu-b200",
)


class TestGLM52NVFP4TPMTPStepGraph(base.TestGLM52NVFP4TPMTP):
    """GLM-5.2-NVFP4 TP4 MTP with --enable-speculative-step-cuda-graph: the same
    accuracy and accept-length thresholds, and the step graph must engage."""

    extra_server_args = base.TestGLM52NVFP4TPMTP.extra_server_args + [
        "--enable-speculative-step-cuda-graph",
    ]

    def test_step_cuda_graph_engaged(self):
        requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 64},
            },
        ).raise_for_status()
        info = requests.get(self.base_url + "/get_server_info").json()
        stats = info["internal_states"][0]["spec_step_cuda_graph"]
        self.assertGreater(stats["replays"], 0, stats)
        self.assertEqual(stats["capture_failures"], 0, stats)
        self.assertIsNone(stats["disabled_reason"], stats)


if __name__ == "__main__":
    unittest.main()
