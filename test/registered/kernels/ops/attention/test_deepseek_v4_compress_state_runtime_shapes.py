"""DeepSeek-V4 compress with a bf16 state pool must track the fp32 state pool.

Covers Flash/Pro C4/C128 decode and prefill presets plus 84 compress shapes
captured from DeepSeek-V4-Flash serving (some short C128 cases emit no rows).
The timing benchmark over the same shapes lives in
``test/manual/kernels/benchmark/attention/bench_deepseek_v4_compress_state.py``.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.deepseek_v4.compress_state import (
    CAPTURED_FLASH_RUNTIME_SPECS,
    CompressSpec,
    make_compress_pair,
)

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=25, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PRESET_SPECS = [
    CompressSpec("flash", 4, "decode", 2, 512, 128),
    CompressSpec("flash", 4, "prefill", 1, 512, 16),
    CompressSpec("pro", 128, "decode", 2, 512, 128),
    CompressSpec("pro", 128, "prefill", 1, 512, 128),
]


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.numel() == 0:
        return 0.0
    return (lhs.float() - rhs.float()).abs().max().item()


@pytest.mark.parametrize("spec", PRESET_SPECS + CAPTURED_FLASH_RUNTIME_SPECS)
def test_bf16_state_matches_fp32_state(spec: CompressSpec) -> None:
    fp32_case, bf16_case = make_compress_pair(spec, seed=20260603)
    fp32_out = fp32_case.run()
    bf16_out = bf16_case.run()
    torch.cuda.synchronize()

    assert _max_abs_diff(fp32_out, bf16_out) < 0.1
    assert _max_abs_diff(fp32_case.state_pool, bf16_case.state_pool) < 0.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
