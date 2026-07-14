"""Kernel-level correctness test for seg_la state_layout (kv vs vk).

Calls ``seg_la_fwd`` directly (no server, no model), comparing
``state_layout="kv"`` and ``"vk"`` across prefill (zero / nonzero init),
decode, spec, and MTP.  The primary assertion is vk == kv (identical by
construction under refined-B) and, where state is written, vk final state
== kv final state transposed.

State tensors are passed to the kernel *uncloned* so the in-place state
write-back is actually verified (not the original inputs).
"""

import unittest

import torch

from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_amd_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_decays(H, D, device):
    """slope ∈ (0.3/H, 0.3) → decay = exp(-slope)."""
    slope = 0.3 * torch.arange(1, H + 1, device=device, dtype=torch.float32) / H
    return slope.view(H, 1, 1)


def _make_packed_inputs(B, S, H, D, device="cuda", dtype=torch.bfloat16):
    total = B * S
    q = torch.randn(total, H, D, device=device, dtype=dtype)
    k = torch.randn(total, H, D, device=device, dtype=dtype)
    v = torch.randn(total, H, D, device=device, dtype=dtype)
    q_offsets = torch.arange(0, total + 1, S, device=device, dtype=torch.int32)
    return q, k, v, q_offsets


def _make_seg_meta(B, S, q_offsets, s_scales_val, s_offsets, mask=None):
    return SegLaMeta(
        batch_size=B,
        max_q_length=S,
        q_offsets=q_offsets,
        s_offsets=s_offsets,
        q_lengths=q_offsets.diff(),
        s_scales=torch.full(
            (B,), s_scales_val, device=q_offsets.device, dtype=torch.int32
        ),
        mask=mask,
    )


def _new_state(B, H, D, device):
    """Return a random state pool [B, H, D, D] fp32 in kv layout."""
    return torch.randn(B, H, D, D, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSegLaStateLayout(CustomTestCase):
    H = 2
    D = 128  # decode needs K_SPLIT_DIM=128 → HEAD_DIM=128
    atol = 5e-2 if is_in_amd_ci() else 3e-2
    rtol = 5e-2 if is_in_amd_ci() else 3e-2

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # ---- helpers ----
    def _run_both_layouts(
        self,
        B,
        S,
        state_kv,
        decay,
        softmax_scale,
        q,
        k,
        v,
        q_offsets,
        caches_kv=None,
        caches_vk=None,
        cache_indices=None,
        mask=None,
        has_init=True,
    ):
        """Run seg_la_fwd for kv and vk.

        ``state_kv`` / ``caches_kv`` are passed *uncloned* so the kernel's
        in-place state write is observable on return. ``has_init`` controls
        s_scales (1 = read existing state, 0 = zero-init).
        Returns (o_kv, o_vk, s_kv_final, s_vk_final, caches_kv, caches_vk).
        """
        s_offsets = torch.arange(B, device=q.device, dtype=torch.int32)
        # state_vk is an independent [v,k]-laid-out tensor (transpose + copy).
        state_vk = state_kv.clone().transpose(-1, -2).contiguous()
        s_scales_val = 1 if has_init else 0

        meta = _make_seg_meta(B, S, q_offsets, s_scales_val, s_offsets, mask=mask)
        ci_kv = cache_indices.clone() if cache_indices is not None else None
        ci_vk = cache_indices.clone() if cache_indices is not None else None

        # kv — kernel writes final state back into state_kv in place
        o_kv = seg_la_fwd(
            q,
            k,
            v,
            state_kv,
            decay,
            meta,
            caches=caches_kv,
            cache_indices=ci_kv,
            softmax_scale=softmax_scale,
            state_layout="kv",
        )
        # vk — kernel writes final state back into state_vk in place
        o_vk = seg_la_fwd(
            q,
            k,
            v,
            state_vk,
            decay,
            meta,
            caches=caches_vk,
            cache_indices=ci_vk,
            softmax_scale=softmax_scale,
            state_layout="vk",
        )
        return o_kv, o_vk, state_kv, state_vk, caches_kv, caches_vk

    def _assert_output_match(self, o_kv, o_vk):
        self.assertTrue(
            torch.allclose(o_vk, o_kv, atol=self.atol, rtol=self.rtol),
            f"vk != kv output  max_diff={(o_vk - o_kv).abs().max().item():.6f}",
        )

    def _assert_state_transpose_match(self, s_kv_final, s_vk_final):
        """vk final state must equal kv final state transposed (kernel-written)."""
        s_kv_t = s_kv_final.transpose(-1, -2)
        self.assertTrue(
            torch.allclose(s_vk_final, s_kv_t, atol=self.atol, rtol=self.rtol),
            f"state: vk != kv.transpose  max_diff={(s_vk_final - s_kv_t).abs().max().item():.6f}",
        )

    # ---- PREFILL ----
    def test_prefill_zero_init_vsplit32(self):
        """bs=2 → V_SPLIT_DIM=32; s_scales=0 (zero state init)."""
        B, S, H, D = 2, 64, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)

        o_kv, o_vk, skv, svk, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            has_init=False,
        )
        self._assert_output_match(o_kv, o_vk)
        self._assert_state_transpose_match(skv, svk)

    def test_prefill_zero_init_vsplit64(self):
        """bs=4 → V_SPLIT_DIM=64; s_scales=0."""
        B, S, H, D = 4, 64, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)

        o_kv, o_vk, skv, svk, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            has_init=False,
        )
        self._assert_output_match(o_kv, o_vk)
        self._assert_state_transpose_match(skv, svk)

    def test_prefill_nonzero_init(self):
        """s_scales=1 with random state — exercises tl.trans on prefill load."""
        B, S, H, D = 2, 64, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = _new_state(B, H, D, q.device)

        o_kv, o_vk, skv, svk, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            has_init=True,
        )
        self._assert_output_match(o_kv, o_vk)
        self._assert_state_transpose_match(skv, svk)

    # ---- DECODE ----
    def test_decode_vsplit32(self):
        """bs ≤ 128 → V_SPLIT_DIM=32."""
        B, S, H, D = 64, 1, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = _new_state(B, H, D, q.device)

        o_kv, o_vk, skv, svk, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            has_init=True,
        )
        self._assert_output_match(o_kv, o_vk)
        self._assert_state_transpose_match(skv, svk)

    def test_decode_vsplit64(self):
        """bs > 128 → V_SPLIT_DIM=64."""
        B, S, H, D = 256, 1, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = _new_state(B, H, D, q.device)

        o_kv, o_vk, skv, svk, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            has_init=True,
        )
        self._assert_output_match(o_kv, o_vk)
        self._assert_state_transpose_match(skv, svk)

    # ---- SPEC ----
    def test_spec_chain(self):
        """Speculative decode: chain mask, spec writes no state (read-only).

        s_scales=1 so the VK tl.trans-on-load path is exercised.
        """
        B, S, H, D = 2, 16, self.H, self.D
        q, k, v, q_offsets = _make_packed_inputs(B, S, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = _new_state(B, H, D, q.device)
        mask = torch.tril(torch.ones(B, S, S, device=q.device, dtype=torch.int32))

        o_kv, o_vk, _, _, _, _ = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            mask=mask,
            has_init=True,
        )
        # Spec does NOT write back state; compare outputs only.
        self._assert_output_match(o_kv, o_vk)

    # ---- MTP ----
    def test_mtp_chain(self):
        """MTP path: caches/intermediate_ssm with step=4 tokens per request.

        Ensures both s_ptrs and c_ptrs are swapped consistently — the MTP
        scatter (layout-agnostic whole-slot copy) stays correct.
        """
        step = 4
        B = step
        H, D = self.H, self.D
        S = step
        q, k, v, q_offsets = _make_packed_inputs(B, step, H, D)
        decay = _make_decays(H, D, q.device)
        scale = D**-0.5
        state_kv = _new_state(B, H, D, q.device)

        # Sentinel init so unwritten slots stay distinguishable; kernel writes
        # real values. caches_vk is an independent [v,k] tensor.
        caches_kv = torch.full(
            (B, step, H, D, D), -1.0, device=q.device, dtype=torch.float32
        )
        caches_vk = caches_kv.clone().transpose(-1, -2).contiguous()
        cache_indices = torch.arange(B, device=q.device, dtype=torch.int32)

        o_kv, o_vk, _, _, caches_kv, caches_vk = self._run_both_layouts(
            B,
            S,
            state_kv,
            decay,
            scale,
            q,
            k,
            v,
            q_offsets,
            caches_kv=caches_kv,
            caches_vk=caches_vk,
            cache_indices=cache_indices,
            has_init=True,
        )
        self._assert_output_match(o_kv, o_vk)
        # MTP writes to caches, not state; caches_vk == transpose(caches_kv).
        self._assert_state_transpose_match(caches_kv, caches_vk)


if __name__ == "__main__":
    unittest.main()
