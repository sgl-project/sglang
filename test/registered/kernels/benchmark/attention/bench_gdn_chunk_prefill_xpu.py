"""Speed guard for the Gated DeltaNet chunk prefill kernels on Intel Arc B60 / B580.

Regression guarded: masked pointer loads in these Triton kernels are not lowered to
2D block loads on XPU, which made the prefill 2.6x slower with identical outputs, so
only a speed check catches it.
"""

import unittest
from collections import defaultdict

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="nightly-xpu-kernel-main-1-gpu", nightly=True)

# PPLX-Decider-v1-27B (Qwen3.5 hybrid) per-rank shape at TP=4: one 4K-token chunk.
SEQ_LEN = 4096
NUM_QK_HEADS = 4
NUM_V_HEADS = 12
HEAD_DIM = 128
# About 21 MB of inputs per copy; rotating 4 copies keeps every call L2-cold (18 MB L2).
NUM_INPUT_COPIES = 4
NUM_TIMED_CALLS = 20
# The fastest of several trials, so a transient neighbour on the host does not fail it.
NUM_TRIALS = 5
# Ceilings in ms per call, between the B60 time with 2D block I/O and with masked
# pointer loads (two B60 hosts): 0.21-0.26 vs 2.22-2.30, 0.20-0.22 vs 1.55-1.59,
# 0.87-0.91 vs 1.22-1.24, 0.36-0.37 vs 0.45-0.51.
KERNEL_CEILING_MS = {
    "chunk_fwd_kernel_o": 0.45,
    "recompute_w_u_fwd_kernel": 0.45,
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop": 1.10,
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg": 0.42,
}
SUPPORTED_DEVICES = ("B60", "B580")


def _is_supported_device() -> bool:
    if not torch.xpu.is_available():
        return False
    name = torch.xpu.get_device_name(0)
    return any(device in name for device in SUPPORTED_DEVICES)


def _make_inputs(seed: int) -> dict:
    # CPU generator + copy: torch.Generator(device="xpu") is not portable across the
    # XPU torch builds CI runs.
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to("xpu", torch.bfloat16)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator).to("xpu")

    return {
        "q": randn(1, SEQ_LEN, NUM_QK_HEADS, HEAD_DIM),
        "k": randn(1, SEQ_LEN, NUM_QK_HEADS, HEAD_DIM),
        "v": randn(1, SEQ_LEN, NUM_V_HEADS, HEAD_DIM),
        "g": rand(1, SEQ_LEN, NUM_V_HEADS).mul_(-0.1),
        "beta": rand(1, SEQ_LEN, NUM_V_HEADS).to(torch.bfloat16),
        "initial_state": torch.zeros(
            1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="xpu", dtype=torch.float32
        ),
        "initial_state_indices": torch.zeros(1, device="xpu", dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, SEQ_LEN], device="xpu", dtype=torch.long),
    }


def _run(inputs: dict) -> None:
    chunk_gated_delta_rule(**inputs, scale=HEAD_DIM**-0.5, use_qk_l2norm_in_kernel=True)


def _profile_trial(inputs: list[dict]) -> dict[str, float]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.XPU]
    ) as profile:
        for call in range(NUM_TIMED_CALLS):
            _run(inputs[call % NUM_INPUT_COPIES])
        torch.xpu.synchronize()
    device_time_us = defaultdict(float)
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.XPU:
            device_time_us[event.name] += event.device_time_total
    return {
        kernel: total_us / NUM_TIMED_CALLS / 1e3
        for kernel, total_us in device_time_us.items()
    }


@unittest.skipUnless(
    _is_supported_device(), "ceilings are measured on Intel Arc B60 / B580"
)
class TestGdnChunkPrefillXpuSpeed(CustomTestCase):
    def test_prefill_kernels_stay_under_ceiling(self):
        inputs = [_make_inputs(seed) for seed in range(NUM_INPUT_COPIES)]
        for copy in inputs:
            _run(copy)
        torch.xpu.synchronize()

        best_ms: dict[str, float] = {}
        for _ in range(NUM_TRIALS):
            for kernel, per_call_ms in _profile_trial(inputs).items():
                best_ms[kernel] = min(best_ms.get(kernel, per_call_ms), per_call_ms)

        for kernel, ceiling_ms in KERNEL_CEILING_MS.items():
            per_call_ms = best_ms.get(kernel, 0.0)
            print(f"{kernel}: {per_call_ms:.3f} ms per call (ceiling {ceiling_ms})")
            with self.subTest(kernel=kernel):
                self.assertGreater(per_call_ms, 0.0, f"{kernel} did not run")
                self.assertLess(
                    per_call_ms,
                    ceiling_ms,
                    f"{kernel}: {per_call_ms:.3f} ms per call, ceiling {ceiling_ms} ms",
                )


if __name__ == "__main__":
    unittest.main()
