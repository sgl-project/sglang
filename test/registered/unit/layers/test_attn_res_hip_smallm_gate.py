"""use_hip_smallm admits exactly the inputs attn_res_smallm_hip accepts."""

import pytest
import torch

from sglang.kernels.ops.attention import attn_res_smallm_hip as smallm
from sglang.srt.layers import attn_residual
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hip_shape_gate(monkeypatch):
    monkeypatch.setattr(attn_residual, "_use_hip_fused", lambda h, nvb: True)
    monkeypatch.setattr(attn_residual, "smallm_max_t", lambda: 16)


def test_eligible_bf16_selects_the_hip_kernel():
    assert attn_residual.use_hip_smallm(4, 7168, 3, torch.bfloat16)


@pytest.mark.parametrize(
    "num_tokens,hidden,nvb,dtype",
    [
        (4, 7168, 3, torch.float16),
        (4, 7168, 3, torch.float32),
        (4, 7164, 3, torch.bfloat16),
        (4, 7168, smallm.MAX_NVB + 1, torch.bfloat16),
        (4, 7168, 0, torch.bfloat16),
        (0, 7168, 3, torch.bfloat16),
        (17, 7168, 3, torch.bfloat16),
    ],
)
def test_unsupported_inputs_decline(num_tokens, hidden, nvb, dtype):
    assert not attn_residual.use_hip_smallm(num_tokens, hidden, nvb, dtype)


def test_gate_matches_kernel_contract():
    for hidden in (1024, 4096, 7168, 7176, 16384, 65536):
        for nvb in range(1, smallm.MAX_NVB + 2):
            assert attn_residual.use_hip_smallm(
                4, hidden, nvb, torch.bfloat16
            ) == smallm.supported(hidden, nvb)
