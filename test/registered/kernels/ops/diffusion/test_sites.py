"""``diffusion.sites``: the two gate protocols and the mount lifecycle.

Nothing here is a kernel.  ``sites`` decides *whether* a fusion is allowed to
run for a given request and model:

- :class:`QualityGatedFusion` -- for fusions that are **not** bit-exact.  They
  mount onto marked ``nn.Module`` sites only for ``quality="high"``, at batch
  boundaries, all-or-nothing per transformer.
- :class:`BitExactFusionGate` -- for fusions that **are** bit-exact.  They run
  by default but verify themselves against the live eager chain on first
  sight and disable permanently on any mismatch.

The protocol tests are pure-CPU.  The mount-lifecycle tests below use
synthetic sites (a bare ``nn.Module`` with the marker attribute) so they test
the protocol rather than any one model; the real model wrappers live in
``test_model_fast_paths.py``.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.kernels.ops.diffusion.sites.fused_gate_rmsnorm_site as gate_rmsnorm
import sglang.kernels.ops.diffusion.sites.fused_linear_gelu_site as linear_gelu
import sglang.kernels.ops.diffusion.sites.sana_video_linear_attention_site as sana_video_linear_attention
from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    QualityGatedFusion,
    can_use_ln_modulate,
    flashinfer_rmsnorm_diagnostic_hint,
    fused_ln_modulate,
    fused_ln_modulate_active,
    mark_fused_ln_modulate_site,
    mount_fused_ln_modulate,
    tensors_equal,
    unmount_fused_ln_modulate,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=38, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# ---------------------------------------------------------------------------
# QualityGatedFusion protocol (CPU)
# ---------------------------------------------------------------------------


def test_quality_gate_mounts_and_unmounts_all_sites():
    fusion = QualityGatedFusion(
        name="test fusion",
        marker_attr="_test_fusion_site",
        enabled_attr="_test_fusion_enabled",
    )
    root = nn.ModuleList([nn.Module(), nn.Module()])
    for index, site in enumerate(root):
        fusion.mark(site, index)

    assert [fusion.metadata(site) for site in root] == [0, 1]
    assert fusion.mount(root)
    assert all(fusion.is_enabled(site) for site in root)
    fusion.unmount(root)
    assert not any(fusion.is_enabled(site) for site in root)


def test_quality_gate_rejection_is_all_or_nothing():
    fusion = QualityGatedFusion(
        name="test fusion",
        marker_attr="_test_fusion_site",
        enabled_attr="_test_fusion_enabled",
    )
    root = nn.ModuleList([nn.Module(), nn.Module()])
    for index, site in enumerate(root):
        fusion.mark(site, index)

    assert not fusion.mount(
        root, reject_reason=lambda site: "rejected" if fusion.metadata(site) else None
    )
    assert not any(fusion.is_enabled(site) for site in root)
    assert not fusion.mount(nn.Module())


# -------------------------------------------------------------------------
# BitExactFusionGate protocol (CPU)
# -------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mount lifecycle on synthetic sites (CUDA)
# ---------------------------------------------------------------------------

LN_DIM = 3072  # FLUX.1 hidden size


@requires_cuda
@pytest.mark.parametrize("seq_len", [4096, 512])
def test_fused_ln_modulate_matches_reference(seq_len):
    torch.cuda.manual_seed(0)
    x = torch.randn((1, seq_len, LN_DIM), device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((1, LN_DIM), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    assert can_use_ln_modulate(x, scale, shift)

    norm = nn.LayerNorm(LN_DIM, eps=1e-6, elementwise_affine=False).cuda()
    ref = norm(x) * (1 + scale[:, None]) + shift[:, None]
    # Contract: bf16 rounding-order-level difference only, not bit-exact --
    # which is exactly why this fusion is quality-gated.
    torch.testing.assert_close(
        fused_ln_modulate(x, scale, shift, eps=1e-6), ref, atol=0.0625, rtol=0.05
    )


@requires_cuda
def test_fused_ln_modulate_guards_and_mount_protocol():
    x = torch.randn((2, 64, LN_DIM), device="cuda", dtype=torch.bfloat16)
    row = torch.randn((2, LN_DIM), device="cuda", dtype=torch.bfloat16)
    assert not can_use_ln_modulate(x, row, row)  # folded affine needs B == 1

    root = nn.Module()
    root.child = nn.Module()
    mark_fused_ln_modulate_site(root.child)
    assert not fused_ln_modulate_active(root.child)
    assert mount_fused_ln_modulate(root)
    assert fused_ln_modulate_active(root.child)
    unmount_fused_ln_modulate(root)
    assert not fused_ln_modulate_active(root.child)
    assert not mount_fused_ln_modulate(nn.Module())  # no marked sites


@requires_cuda
@torch.no_grad()
def test_mounted_ln_modulate_site_compiles_fullgraph():
    class Site(nn.Module):
        def __init__(self):
            super().__init__()
            mark_fused_ln_modulate_site(self)

        def forward(self, x, scale, shift):
            if fused_ln_modulate_active(self) and can_use_ln_modulate(x, scale, shift):
                return fused_ln_modulate(x, scale, shift, eps=1e-6)
            return (
                F.layer_norm(x, (x.shape[-1],), eps=1e-6) * (1 + scale[:, None])
                + shift[:, None]
            )

    site = Site()
    assert mount_fused_ln_modulate(site)
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    expected = site(x, scale, shift)
    torch.testing.assert_close(
        torch.compile(site, fullgraph=True)(x, scale, shift),
        expected,
        atol=0.0625,
        rtol=0.05,
    )


GATE_RMSNORM_DIM, GATE_RMSNORM_EPS = 4608, 1e-5  # Ideogram 4 hidden / norm_eps


class _GateRMSNormSite(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.norm = nn.RMSNorm(
            GATE_RMSNORM_DIM, eps=GATE_RMSNORM_EPS, device="cuda", dtype=dtype
        )
        gate_rmsnorm.mark_fused_gate_rmsnorm_site(self, ("norm",))


@requires_cuda
def test_fused_gate_rmsnorm_matches_ideogram_reference():
    torch.manual_seed(0)
    site = _GateRMSNormSite()
    w = site.norm.weight.data
    dim = GATE_RMSNORM_DIM
    x = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    # adaln-style strided chunks, as produced by Ideogram's modulation .chunk()
    mods = torch.randn(1, 1, 2 * dim, device="cuda", dtype=torch.bfloat16)
    scale, gate = mods.chunk(2, dim=-1)

    assert gate_rmsnorm.mount_fused_gate_rmsnorm(site)
    got_scale = gate_rmsnorm.fused_rmsnorm_scale(x, w, 1.0 + scale, GATE_RMSNORM_EPS)
    got_gate = gate_rmsnorm.fused_rmsnorm_tanh_residual(
        x, gate, residual, w, GATE_RMSNORM_EPS
    )
    norm = F.rms_norm(x, (dim,), w, GATE_RMSNORM_EPS)
    # The fused path uses bf16-native norm statistics: close, not bit-exact.
    torch.testing.assert_close(got_scale, norm * (1.0 + scale), atol=8e-2, rtol=4e-2)
    torch.testing.assert_close(
        got_gate, residual + torch.tanh(gate) * norm, atol=8e-2, rtol=4e-2
    )


@requires_cuda
def test_fused_gate_rmsnorm_mount_is_all_or_nothing():
    good, bad = _GateRMSNormSite(), _GateRMSNormSite(torch.float32)
    # One fp32 norm anywhere in the tree keeps *every* site on the reference.
    assert not gate_rmsnorm.mount_fused_gate_rmsnorm(nn.ModuleList([good, bad]))
    assert not gate_rmsnorm.fused_gate_rmsnorm_active(good)
    assert gate_rmsnorm.mount_fused_gate_rmsnorm(good)
    gate_rmsnorm.unmount_fused_gate_rmsnorm(good)
    assert not gate_rmsnorm.fused_gate_rmsnorm_active(good)


class _GeluSite(nn.Module):
    def __init__(self, dtype=torch.bfloat16, bias=True):
        super().__init__()
        self.proj = nn.Linear(64, 256, bias=bias, device="cuda", dtype=dtype)
        linear_gelu.mark_fused_gelu_site(self, "proj")

    def forward(self, x):
        if linear_gelu.fused_gelu_active(self) and linear_gelu.can_use_linear_gelu(
            self.proj, x
        ):
            return linear_gelu.fused_linear_gelu_tanh(
                x, self.proj.weight, self.proj.bias
            )
        return F.gelu(self.proj(x), approximate="tanh")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_linear_gelu_matches_reference(dtype):
    torch.manual_seed(0)
    site = _GeluSite(dtype)
    x = torch.randn(512, 64, device="cuda", dtype=dtype)
    ref = site(x)  # unmounted: the reference chain
    assert linear_gelu.mount_fused_linear_gelu(site)
    atol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(site(x), ref, atol=atol, rtol=2e-2)


@requires_cuda
def test_fused_linear_gelu_guards_and_lossless_path():
    torch.manual_seed(0)
    good, bad = _GeluSite(), _GeluSite(torch.float32)
    assert not linear_gelu.mount_fused_linear_gelu(nn.ModuleList([good, bad]))
    assert not linear_gelu.fused_gelu_active(good)

    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    ref = good(x)
    assert linear_gelu.mount_fused_linear_gelu(good)
    linear_gelu.unmount_fused_linear_gelu(good)
    # Unmounting must restore the reference path bit-for-bit.
    assert torch.equal(good(x), ref)

    no_bias = nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    assert not linear_gelu.can_use_linear_gelu_static(no_bias)
    assert not linear_gelu.can_use_linear_gelu(good.proj, x.float())


@requires_cuda
@torch.no_grad()
def test_mounted_gelu_site_compiles_fullgraph():
    site = _GeluSite()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    assert linear_gelu.mount_fused_linear_gelu(site)
    expected = site(x)
    torch.testing.assert_close(
        torch.compile(site, fullgraph=True)(x), expected, atol=2e-2, rtol=2e-2
    )


@requires_cuda
@torch.no_grad()
def test_sana_video_linear_attention_quality_path_and_guards():
    torch.manual_seed(0)
    site = nn.Module()
    sana_video_linear_attention.mark_sana_video_linear_attention_site(site)
    shape = (1, 4, 16, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    normalizer = torch.randn(
        shape[0], shape[1], 1, shape[-1], device="cuda", dtype=torch.bfloat16
    )

    assert (
        sana_video_linear_attention.try_sana_video_linear_attention(
            site, query, key, value, normalizer
        )
        is None
    )
    assert sana_video_linear_attention.mount_sana_video_linear_attention(site)
    output = sana_video_linear_attention.try_sana_video_linear_attention(
        site, query, key, value, normalizer
    )
    reference = ((value.float() @ key.float().transpose(-1, -2)) @ query.float()) * (
        normalizer
    )
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)

    assert (
        sana_video_linear_attention.try_sana_video_linear_attention(
            site,
            query.expand(2, -1, -1, -1),
            key.expand(2, -1, -1, -1),
            value.expand(2, -1, -1, -1),
            normalizer.expand(2, -1, -1, -1),
        )
        is None
    )
    sana_video_linear_attention.unmount_sana_video_linear_attention(site)
    assert not sana_video_linear_attention.sana_video_linear_attention_active(site)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
