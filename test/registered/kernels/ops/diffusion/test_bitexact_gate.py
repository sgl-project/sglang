import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.kernels.ops.diffusion.bitexact_gate import (
    BitExactFusionGate,
    flashinfer_rmsnorm_diagnostic_hint,
    tensors_equal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def test_bitexact_gate_once_mode_verifies_then_reuses():
    gate = BitExactFusionGate("once")
    calls = {"fused": 0, "ref": 0}

    def fused():
        calls["fused"] += 1
        return torch.tensor([1.0])

    def ref():
        calls["ref"] += 1
        return torch.tensor([1.0])

    assert torch.equal(gate.accept_or_fallback(fused(), ref()), torch.tensor([1.0]))
    assert gate.verified and not gate.disabled and calls == {"fused": 1, "ref": 1}
    assert torch.equal(fused(), torch.tensor([1.0]))
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


def test_bitexact_gate_skips_first_sight_during_graph_capture(monkeypatch):
    # Negative-branch contract: an unverified gate must not attempt first-sight
    # verification inside CUDA graph capture — the eager-reference host sync
    # would abort the capture (and BCG would permanently block the signature).
    gate = BitExactFusionGate("capture")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not gate.can_attempt_once()
    # A verified gate replays the fused kernel alone, which is capture-safe.
    gate.mark_verified()
    assert gate.can_attempt_once()


def test_tensors_equal_supports_sequences():
    assert tensors_equal(
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([1.0]), torch.tensor([2.0])),
    )
    assert not tensors_equal(
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([1.0]), torch.tensor([3.0])),
    )


class TestBitExactFallbackDiagnostics(CustomTestCase):
    def test_mismatch_warning_is_actionable_and_diagnostic_is_lazy(self):
        logger = MagicMock()
        diagnostic = MagicMock(return_value="backend=CuTe DSL")
        gate = BitExactFusionGate("diagnostic")

        matched = gate.accept_or_fallback(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            logger=logger,
            diagnostic_hint=diagnostic,
        )
        self.assertTrue(torch.equal(matched, torch.tensor([1.0])))
        diagnostic.assert_not_called()
        logger.warning_once.assert_not_called()

        gate = BitExactFusionGate("diagnostic")
        fallback = gate.accept_or_fallback(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            logger=logger,
            diagnostic_hint=diagnostic,
        )

        self.assertTrue(torch.equal(fallback, torch.tensor([2.0])))
        diagnostic.assert_called_once_with()
        warning = logger.warning_once.call_args.args[0]
        self.assertIn("Correctness is preserved", warning)
        self.assertIn("reference kernel or reduction-order change", warning)
        self.assertIn("backend=CuTe DSL", warning)

    def test_diagnostic_failure_cannot_break_the_eager_fallback(self):
        logger = MagicMock()

        def broken_diagnostic():
            raise RuntimeError("diagnostics unavailable")

        gate = BitExactFusionGate("diagnostic")
        fallback = gate.accept_or_fallback(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            logger=logger,
            diagnostic_hint=broken_diagnostic,
        )

        self.assertTrue(torch.equal(fallback, torch.tensor([2.0])))
        self.assertTrue(gate.disabled)
        self.assertIn("Correctness is preserved", logger.warning_once.call_args.args[0])

    def test_flashinfer_rmsnorm_hint_reports_backend_and_versions(self):
        flashinfer = ModuleType("flashinfer")
        flashinfer_norm = ModuleType("flashinfer.norm")
        flashinfer_norm._USE_CUDA_NORM = False
        versions = {
            "flashinfer-python": "0.6.12",
            "flashinfer-cubin": "0.6.12",
            "flashinfer-jit-cache": "0.6.12+cu130",
        }

        with (
            patch.dict(
                sys.modules,
                {"flashinfer": flashinfer, "flashinfer.norm": flashinfer_norm},
            ),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
            patch.dict("os.environ", {"FLASHINFER_USE_CUDA_NORM": "0"}),
        ):
            hint = flashinfer_rmsnorm_diagnostic_hint()

        self.assertIn("backend=CuTe DSL", hint)
        self.assertIn("FLASHINFER_USE_CUDA_NORM=0", hint)
        for package, version in versions.items():
            self.assertIn(f"{package}={version}", hint)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
