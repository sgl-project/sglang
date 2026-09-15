"""KV-canary end-to-end on Intel XPU.

Exercises ``--kv-canary`` on ``--device xpu``, where the write / verify /
plan-entries kernels are CUDA-JIT only, so they route to their torch references
via ``kv_canary._dispatch.use_torch_reference`` and the D2H stream/event
machinery runs through ``torch.xpu``.

Both directions are needed: a dispatch shim that silently no-oped would pass the
baseline too, so only an injected corruption going *undetected* separates a
working fallback from a dead one.
"""

from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")

_XPU_SERVER_ARGS = ("--device", "xpu")


class _XPUCanaryE2EBase(CanaryE2EBase):
    """Shared XPU server config for the cases below.

    No workload sizes here on purpose: ``--device xpu`` puts the canary on its torch
    reference, which ``CanaryE2EBase`` already clamps the workload for.
    """

    model_mode = "mha"
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _XPU_SERVER_ARGS


class TestXPUCanaryBaseline(_XPUCanaryE2EBase):
    """Clean XPU canary run: no violations, all requests succeed."""

    def test_no_violation(self) -> None:
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)


class TestXPUCanaryPerturbDetected(_XPUCanaryE2EBase):
    """Injected req_to_token corruption must be detected on XPU."""

    extra_env = {
        # 0.1 per forward fires well inside the clamped workload.
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "0.1",
        "SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS": "0",
        # Corrupting the slot mapping looks like a pool leak to the on-idle checker.
        # Expected here, so strict mode stays off or the scheduler crashes before we
        # can assert.
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
    }

    def test_req_to_token_perturbation_reports_chain_hash_violation(self) -> None:
        self.send_parallel_requests()
        self.assert_per_forward_violation_reported(fail_reason="verify_chain_hash")


if __name__ == "__main__":
    unittest.main()
