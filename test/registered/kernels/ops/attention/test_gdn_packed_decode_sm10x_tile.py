"""SM10x packed GDN decode with an FP32 state uses BV=4, bit-identical to BV=32.

Each program of ``fused_recurrent_gated_delta_rule_packed_decode_kernel`` owns
BV rows of the value dim and reduces over the full K for each of them, so the
tile only changes how rows are split across programs: output and updated state
must match a BV=32 launch bit for bit. BF16 states keep their current tile.
"""

import pytest
import torch

from sglang.kernels.ops.attention.fla import fused_recurrent as fr
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM10x-only value-tile selection",
)

_K = _V = 128
_KERNEL = fr.fused_recurrent_gated_delta_rule_packed_decode_kernel


class _LaunchSpy:
    """Records the BV of each packed-decode launch and forwards to the real kernel."""

    def __init__(self, kernel, force_bv=None):
        self.kernel = kernel
        self.force_bv = force_bv
        self.bvs = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.bvs.append(kwargs["BV"])
            if self.force_bv is not None:
                kwargs["BV"] = self.force_bv
                grid_ = (kwargs["V"] // self.force_bv, grid[1])
            else:
                grid_ = grid
            return self.kernel[grid_](*args, **kwargs)

        return run


def _run(monkeypatch, case, state_dtype, force_bv=None):
    H, HV, B = case
    gen = torch.Generator(device="cuda").manual_seed(H * 1000 + HV * 10 + B)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    mixed_qkv = randn(B, 2 * H * _K + HV * _V).to(torch.bfloat16)
    a = randn(B, HV).to(torch.bfloat16)
    b = randn(B, HV).to(torch.bfloat16)
    A_log = randn(HV) * 0.1
    dt_bias = randn(HV) * 0.1
    # Slot 0 is the null slot; real sequences use slots 1..B, one padded row
    # points at -1 and must leave the pool untouched.
    state = (randn(B + 1, HV, _V, _K) * 0.02).to(state_dtype)
    out = torch.empty(B, 1, HV, _V, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, B + 1, device="cuda", dtype=torch.int32)
    if B > 1:
        indices[-1] = -1

    spy = _LaunchSpy(_KERNEL, force_bv)
    monkeypatch.setattr(
        fr, "fused_recurrent_gated_delta_rule_packed_decode_kernel", spy
    )
    fr.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv, a, b, A_log, dt_bias, _K**-0.5, state, out, indices, True
    )
    torch.cuda.synchronize()
    return out, state, spy.bvs


# (H, HV, B): Qwen3.8-27B at TP1/TP4 (incl. the old TP4 B=1 special case),
# a 16/32-head layout, and a large ragged batch.
CASES = [(16, 48, 1), (16, 48, 64), (4, 12, 1), (16, 32, 8), (4, 16, 257)]


@pytest.mark.parametrize("case", CASES)
def test_fp32_state_uses_bv4_and_matches_bv32(monkeypatch, case):
    out, state, bvs = _run(monkeypatch, case, torch.float32)
    ref_out, ref_state, _ = _run(monkeypatch, case, torch.float32, force_bv=32)
    assert bvs == [4]
    assert torch.equal(out, ref_out)
    assert torch.equal(state, ref_state)


def test_bf16_state_keeps_bv32(monkeypatch):
    _, _, bvs = _run(monkeypatch, (16, 48, 8), torch.bfloat16)
    assert bvs == [32]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
