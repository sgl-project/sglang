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

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

# --disable-cuda-graph is mandatory, not tuning: install_canary refuses a captured decode
# on a device that routes to the torch reference (host work and D2H, so replay checks nothing).
_XPU_SERVER_ARGS = ("--device", "xpu", "--disable-cuda-graph")


class _XPUCanaryE2EBase(CanaryE2EBase):
    """Shared XPU server config for the cases below.

    The torch reference folds the chain slot-by-slot on the host, so it runs orders
    of magnitude slower than the CUDA kernels; this subclass shrinks the workload
    rather than the shared base, which stays on its CUDA-tuned defaults.
    """

    model_mode = "mha"
    kv_canary_mode = CanaryMode.LOG
    extra_server_args = _XPU_SERVER_ARGS
    # Enough decode steps for the chain to span several forwards; measured at roughly
    # 3 tok/s on the reference path, so the timeout is generous rather than tight.
    default_parallel_n = 2
    default_max_new_tokens = 32
    default_request_timeout = 120.0


class TestXPUCanaryBaseline(_XPUCanaryE2EBase):
    """Clean XPU canary run: no violations, all requests succeed."""

    def test_no_violation(self) -> None:
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)


class TestXPUCanaryRealKvBaseline(_XPUCanaryE2EBase):
    """Clean run with real-KV fingerprinting on, the reference's other fold path.

    ``--kv-canary-real-data partial`` is what makes verify/write read the KV pool
    itself; without a case that sets it, the reference's real-KV gather stays
    unexecuted on XPU no matter how many chain-only cases pass.
    """

    extra_server_args = (*_XPU_SERVER_ARGS, "--kv-canary-real-data", "partial")

    def test_no_violation(self) -> None:
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)


class TestXPUCanaryPerturbDetected(_XPUCanaryE2EBase):
    """Injected req_to_token corruption must be detected on XPU."""

    extra_env = {
        # Every forward, so the short reference workload cannot end before it fires.
        "SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB": "1.0",
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
