import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.bitexact_gate import (
    BitExactFusionGate,
    tensors_equal,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_bitexact_gate_once_mode_verifies_then_reuses():
    gate = BitExactFusionGate("once")
    calls = {"fused": 0, "ref": 0}

    def fused():
        calls["fused"] += 1
        return torch.tensor([1.0])

    def ref():
        calls["ref"] += 1
        return torch.tensor([1.0])

    assert torch.equal(gate.run(fused, ref), torch.tensor([1.0]))
    assert gate.verified and not gate.disabled and calls == {"fused": 1, "ref": 1}
    assert torch.equal(gate.run(fused, ref), torch.tensor([1.0]))
    assert calls == {"fused": 2, "ref": 1}


def test_bitexact_gate_mismatch_disables_permanently():
    gate = BitExactFusionGate("mismatch")

    out = gate.accept_or_fallback(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        mismatch_msg="mismatch",
    )
    assert torch.equal(out, torch.tensor([2.0]))
    assert gate.disabled and not gate.verified


def test_bitexact_gate_per_signature_tracks_each_sig():
    gate = BitExactFusionGate("sig", per_signature=True)
    a = torch.tensor([1.0])
    assert torch.equal(gate.accept_or_fallback(a, a, sig=("a",)), a)
    assert gate.is_verified(("a",))
    assert not gate.is_verified(("b",))
    assert torch.equal(gate.accept_or_fallback(a, a, sig=("b",)), a)
    assert gate.verified_sigs == {("a",), ("b",)}


def test_tensors_equal_supports_sequences():
    assert tensors_equal(
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([1.0]), torch.tensor([2.0])),
    )
    assert not tensors_equal(
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([1.0]), torch.tensor([3.0])),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
