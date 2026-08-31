from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=147, stage="base-b", runner_config="1-gpu-large")

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

PROMPTS = [
    "Question: Natalia sold clips to 48 friends in April, and half as many in "
    "May. How many clips did she sell altogether? Answer:",
    "The capital of France is",
    "Q: What is 12 times 13? A:",
]


class TestBatchingFDFOJointThreshold(CustomTestCase):
    """At a single in-flight request, FDFO and synchronous execution run identical
    forward shapes, so a correct stateful (``dllm_algo_state``) carry must produce
    byte-identical multi-block output.
    """

    model = "inclusionAI/LLaDA2.1-mini"
    base_url = DEFAULT_URL_FOR_TEST

    def _collect_outputs(self, fdfo: bool):
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--attention-backend",
            "flashinfer",
            "--dllm-algorithm",
            "JointThreshold",
            "--cuda-graph-bs",
            "1",
        ]
        # FDFO is the default; the sync arm must opt out explicitly.
        other_args.append("--dllm-fdfo" if fdfo else "--no-dllm-fdfo")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            outputs = []
            for prompt in PROMPTS:
                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 128,
                        "temperature": 0,
                    },
                    timeout=120,
                )
                outputs.append(response.json()["choices"][0]["text"])
            return outputs
        finally:
            kill_process_tree(process.pid)

    def test_fdfo_matches_sync(self):
        sync_outputs = self._collect_outputs(fdfo=False)
        fdfo_outputs = self._collect_outputs(fdfo=True)
        self.assertEqual(
            fdfo_outputs,
            sync_outputs,
            "JointThreshold FDFO output must match synchronous output, which "
            "validates the cross-step dllm_algo_state carry across blocks.",
        )


if __name__ == "__main__":
    unittest.main()
