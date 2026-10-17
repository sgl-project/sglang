"""
Disaggregation integration test for the NIXL transfer backend on Intel XPU.

Launches a prefill server, a decode server, and a load-balancer, then verifies
that basic text completion works end-to-end. Exercises the np.uint64
pointer-arithmetic fix in python/sglang/srt/disaggregation/nixl/conn.py, which
is required on Intel XPU where device addresses have bit 63 set (e.g.
0xffff81ab54e01000) and would overflow np.int64.

Manual, not CI: UCX treats SYCL device allocations as host memory, so NIXL
cannot register XPU KV buffers until https://github.com/ai-dynamo/nixl/pull/1536
and https://github.com/openucx/ucx/pull/11218 land. Run against NIXL and UCX
builds that carry both patches:

    python3 test/manual/kv_transfer/test_disaggregation_nixl_xpu.py

Mirrors the Mooncake scenarios in
test/registered/disaggregation/test_disaggregation_xpu.py. Needs 2 XPUs.
"""

import unittest

import requests
import torch

from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

_XPU_AVAILABLE = torch.xpu.is_available()
# PD disaggregation needs two devices: the fixture puts decode on
# decode_base_gpu_id=1 while prefill holds device 0.
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"PD disaggregation needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
)

try:
    import nixl._api  # noqa: F401

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False
_NIXL_SKIP = "nixl not installed"


@unittest.skipUnless(_NIXL_AVAILABLE, _NIXL_SKIP)
@unittest.skipUnless(_XPU_DEVICE_COUNT >= 2, _SKIP_REASON)
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.extra_prefill_args = ["--device", "xpu"]
        # host_pool retraction backup calls cudaHostRegister, which is CUDA-only.
        cls.extra_decode_args = [
            "--device",
            "xpu",
            "--disaggregation-decode-retraction-backup",
            "cpu_tensor",
        ]
        cls.launch_all()

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"]

    def test_completion_returns_text(self):
        text = self._generate("The capital of France is", max_new_tokens=16)
        self.assertGreater(len(text), 0, "Generated text should not be empty")

    def test_completion_correct_output(self):
        generated = self._generate("1 + 1 =", max_new_tokens=4)
        self.assertIn("2", generated, f"Expected '2' in output, got: {generated!r}")


if __name__ == "__main__":
    unittest.main()
