import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class _Eagle3ParityBase(Eagle3Base):
    """Shared configuration for EAGLE3 parity tests; no test logic."""

    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)
    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128


def _greedy(url, text, max_new_tokens=48):
    return requests.post(
        url + "/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    ).json()["text"]


class SpecParityKitNPU:
    """Lossless output parity between speculative and non-speculative decoding.

    Launches a non-speculative reference server first, captures greedy outputs,
    shuts it down, then launches an EAGLE3 speculative server on the same port.
    This sequential setup avoids running two large models concurrently.

    Mix this kit first in the base list so its setUpClass runs before the
    server fixture: ``class T(SpecParityKitNPU, Eagle3Base)``.
    """

    parity_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "The three primary colors are",
        "def fibonacci(n):",
    ]

    @classmethod
    def setUpClass(cls):
        ref_url = DEFAULT_URL_FOR_TEST
        ref_proc = popen_launch_server(
            cls.model,
            ref_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                cls.attention_backend,
                "--page-size",
                "128",
                "--dtype",
                cls.dtype,
                *(["--trust-remote-code"] if cls.trust_remote_code else []),
            ],
        )
        try:
            cls.parity_ref_outputs = {
                p: _greedy(ref_url, p) for p in cls.parity_prompts
            }
        finally:
            kill_process_tree(ref_proc.pid, wait_timeout=60)

        super().setUpClass()

    def test_parity_vs_reference(self):
        """Greedy outputs from EAGLE3 speculative decoding match the non-speculative reference."""
        for prompt in self.parity_prompts:
            spec_out = _greedy(self.base_url, prompt)
            self.assertEqual(
                spec_out,
                self.parity_ref_outputs[prompt],
                f"spec != ref for prompt {prompt!r}",
            )


class TestEagle3ParityNPU(SpecParityKitNPU, _Eagle3ParityBase):
    """Test Case: Verify EAGLE3 speculative decoding greedy output matches non-speculative reference.

    [Test Category] Functionality
    [Test Target] EAGLE3 speculative decoding (lossless output parity)
    """

    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
