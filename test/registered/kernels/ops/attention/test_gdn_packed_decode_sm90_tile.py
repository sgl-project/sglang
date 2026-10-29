"""SM90 packed GDN decode with an FP32 state uses BV=4, bit-identical to BV=32.

``fused_recurrent_gated_delta_rule_packed_decode`` picks a narrower value tile on
SM90 when the recurrent state is FP32. Each program still reduces over the full K
for its own V rows, so the output and the updated state must match the default
BV=32 launch bit for bit. BF16 states keep BV=32.
"""

import pytest
import torch

from sglang.kernels.ops.attention.fla import fused_recurrent as fr
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90-only value-tile selection",
)

_K = _V = 128
_KERNEL = fr.fused_recurrent_gated_delta_rule_packed_decode_kernel


class _LaunchSpy:
    """Records the BV of each packed-decode launch and forwards to the real kernel."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.bvs = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            self.bvs.append(kwargs["BV"])
            return launch(*args, **kwargs)

        return run


def _run(monkeypatch, case, state_dtype, sm90):
    H, HV, B = case
    gen = torch.Generator(device="cuda").manual_seed(H * 1000 + HV * 10 + B)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    mixed_qkv = randn(B, 2 * H * _K + HV * _V).to(torch.bfloat16)
    a = randn(B, HV).to(torch.bfloat16)
    b = randn(B, HV).to(torch.bfloat16)
    A_log = randn(HV) * 0.1
    dt_bias = randn(HV) * 0.1
    # Slot 0 is the null slot; real sequences use slots 1..B.
    state = (randn(B + 1, HV, _V, _K) * 0.02).to(state_dtype)
    out = torch.empty(B, 1, HV, _V, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, B + 1, device="cuda", dtype=torch.int32)

    spy = _LaunchSpy(_KERNEL)
    monkeypatch.setattr(
        fr, "fused_recurrent_gated_delta_rule_packed_decode_kernel", spy
    )
    monkeypatch.setattr(fr, "is_sm90_supported", lambda: sm90)
    fr.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv, a, b, A_log, dt_bias, _K**-0.5, state, out, indices, True
    )
    torch.cuda.synchronize()
    return out, state, spy.bvs


@pytest.mark.parametrize("case", [(16, 48, 1), (16, 48, 64), (2, 6, 1), (4, 16, 257)])
def test_fp32_state_uses_bv4_and_matches_bv32(monkeypatch, case):
    out, state, bvs = _run(monkeypatch, case, torch.float32, sm90=True)
    ref_out, ref_state, ref_bvs = _run(monkeypatch, case, torch.float32, sm90=False)
    assert bvs == [4] and ref_bvs == [32]
    assert torch.equal(out, ref_out)
    assert torch.equal(state, ref_state)


def test_bf16_state_keeps_bv32(monkeypatch):
    _, _, bvs = _run(monkeypatch, (16, 48, 8), torch.bfloat16, sm90=True)
    assert bvs == [32]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
