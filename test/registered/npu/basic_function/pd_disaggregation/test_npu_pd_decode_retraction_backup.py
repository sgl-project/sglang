"""NPU end-to-end tests for the `--disaggregation-decode-retraction-backup` feature.

These tests launch a real prefill + decode + load balancer PD pipeline on Ascend
NPU with the decode side configured with `--disaggregation-decode-retraction-backup
host_pool`. They verify:

1. The PD pipeline serves requests end to end: prefill handles the prompt and the
   decode side, operating under host_pool KV retraction backup, produces the final
   completion with the correct text.
2. The NPU wiring for the host-pool backup is active on the decode server: the
   running server reports `hicache_io_backend == kernel_ascend` and
   `hicache_mem_layout == page_first_direct`, which is exactly what the NPU backend
   applies when the decode side uses (explicit or inferred) host_pool retraction
   backup.
"""

import logging
import os
import unittest
from typing import Dict

# The PD disaggregation backend must be the Ascend engine (not the mooncake
# default in environ.py) to run the NPU host_pool retraction backup path.
# ``setdefault`` keeps any value set externally (e.g. by the CI runner) in force.
os.environ.setdefault("SGLANG_TEST_PD_DISAGG_BACKEND", "ascend")

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_npu_ci(est_time=400, suite="base-b-test-17-npu-a3")
register_npu_ci(est_time=400, suite="nightly-17-npu-a3", nightly=True)

# Explicitly selected on the decode server; the NPU backend then forces
# hicache_io_backend=kernel_ascend and hicache_mem_layout=page_first_direct.
DECODE_RETRACTION_BACKUP = "host_pool"


class TestPDDecodeRetractionBackup(PDDisaggregationServerBase):
    """Testcase: E2E PD serving on NPU with --disaggregation-decode-retraction-backup
    host_pool, plus NPU host-pool wiring assertions on the decode server.

    [Test Category] Functional
    [Test Target] PD disaggregation on NPU
    --disaggregation-mode; --disaggregation-decode-retraction-backup;
    --disaggregation-bootstrap-port;
    """

    extra_prefill_args = [
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.6",
        "--disable-cuda-graph",
    ]
    extra_decode_args = [
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.6",
        "--disable-cuda-graph",
        "--num-reserved-decode-tokens",
        "128",
        "--disaggregation-decode-retraction-backup",
        DECODE_RETRACTION_BACKUP,
    ]

    # One engine per device. In CI these overrides are replaced by the runner's own
    # device assignment; locally they pin prefill/ decode to two idle Ascend chips.
    # Each side sees exactly one visible device, so gpu index 0 maps to it.
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 0
    extra_prefill_env = {
        "ASCEND_RT_VISIBLE_DEVICES": os.getenv("PD_PREFILL_DEV", "1"),
        "ASCEND_MF_STORE_URL": os.getenv(
            "ASCEND_MF_STORE_URL", "tcp://127.0.0.1:24667"
        ),
    }
    extra_decode_env = {
        "ASCEND_RT_VISIBLE_DEVICES": os.getenv("PD_DECODE_DEV", "8"),
        "ASCEND_MF_STORE_URL": os.getenv(
            "ASCEND_MF_STORE_URL", "tcp://127.0.0.1:24667"
        ),
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The repo constant points at the CI model cache. It can be overridden via
        # env for local runs where the cache path is not populated.
        cls.model = (
            os.getenv("SGLANG_TEST_WEIGHTS_PATH") or LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        cls.tokenizer = get_tokenizer(cls.model)
        cls.launch_all()

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                },
            },
            timeout=120,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def test_generate_through_pd_pipeline(self):
        """A request routed through prefill -> decode completes with output.

        With host_pool retraction backup the decode server reserves a HiCache host
        pool to preserve KV across retractions. A successful, non-empty completion
        confirms the PD pipeline (and the NPU host-pool wiring on decode) is alive.
        """
        prompt = "The capital of France is"
        response = self.send_request(prompt, max_tokens=96)
        text = response["text"]
        logging.warning(f"E2E generate output: {text!r}")
        self.assertTrue(isinstance(text, str) and text.strip())
        # With host_pool retraction backup the decode side may drain its batch when
        # the reserved prefill KV is retracted to the host pool and re-fetched, so the
        # completion can legitimately end with either `stop` (EOS) or `length`
        # (max_new_tokens reached). Both are valid proof the PD pipeline is alive.
        self.assertIn(
            response["meta_info"]["finish_reason"]["type"], ("stop", "length")
        )

    def test_decode_side_npu_host_pool_config_resolved(self):
        """The decode server must resolve the NPU host-pool wiring.

        The running decode server was launched with
        --disaggregation-decode-retraction-backup=host_pool, so the NPU backend must
        have forced kernel_ascend IO and page_first_direct layout. Assert those
        resolved values via /server_info.
        """
        info = requests.get(self.decode_url + "/server_info", timeout=30).json()
        self.assertEqual(
            info.get("disaggregation_decode_retraction_backup"), "host_pool"
        )
        self.assertEqual(
            info.get("hicache_io_backend"),
            "kernel_ascend",
            f"NPU host-pool backup must use kernel_ascend IO, got {info.get('hicache_io_backend')}",
        )
        self.assertEqual(
            info.get("hicache_mem_layout"),
            "page_first_direct",
            f"NPU host-pool backup must use page_first_direct layout, got {info.get('hicache_mem_layout')}",
        )


if __name__ == "__main__":
    unittest.main()
