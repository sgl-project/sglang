"""L1 operator correctness: cuLA verify/commit vs pure PyTorch reference.

No model load required — tests the cuLA adapter conventions (layout, decay sign,
scale, off-by-one) by running the same random (q, k, v, h0) through the cuLA
CUDA kernels and a pure-PyTorch recurrent reference, then asserting outputs
and committed state match within numerical tolerance.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from cula.lightning import (
        linear_attention_state_update_kvbuffer,
        linear_attention_verify_kvbuffer,
    )

    CULA_AVAILABLE = True
except ImportError:
    CULA_AVAILABLE = False

register_cuda_ci(est_time=120, suite="base-b-test-4-gpu-b200")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")


# ---------------------------------------------------------------------------
# Pure PyTorch reference for multi-token Lightning Attention (MTP verify)
# ---------------------------------------------------------------------------
def _torch_la_mtp_ref(q, k, v, state_kmajor, decay, scale, T):
    """Pure PyTorch recurrent Lightning Attention.

    Args:
        q, k: [B*T, H, D] bf16 packed token-major
        v:    [B*T, HV, D] bf16
        state_kmajor: [B, HV, D, D] fp32 K-major initial state
        decay: [H] fp32 positive slopes
        scale: float
        T: int draft_token_num
    Returns:
        out:       [B*T, HV, D] bf16 all-T verify output
        state_new: [B, HV, D, D] fp32 state after T recurrent steps
        inter:     [B, T, HV, D, D] fp32 per-step intermediate states
    """
    B = q.shape[0] // T
    H = q.shape[1]
    HV = v.shape[1]
    D = q.shape[2]

    q_f = q.view(B, T, H, D).float() * scale
    k_f = k.view(B, T, H, D).float()
    v_f = v.view(B, T, HV, D).float()

    # Repeat decay per q-head to match HV (grouped linear attention)
    decay_per_q = torch.exp(-decay.float())
    decay_per_hv = decay_per_q.repeat_interleave(HV // H).view(1, HV, 1, 1)

    state = state_kmajor.float().clone()
    out = torch.zeros(B, T, HV, D, dtype=torch.bfloat16, device=q.device)
    inter = torch.zeros(B, T, HV, D, D, dtype=torch.float32, device=q.device)

    for t in range(T):
        q_t = q_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        k_t = k_f[:, t].repeat_interleave(HV // H, dim=1)  # [B, HV, D]
        v_t = v_f[:, t]  # [B, HV, D]
        state = state * decay_per_hv + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_t, state).bfloat16()
        inter[:, t] = state.clone()

    return out.view(B * T, HV, D), state, inter


@unittest.skipUnless(
    CULA_AVAILABLE, "cuLA not installed (pip install cuda-linear-attention)"
)
class TestCulaKernelCorrectness(CustomTestCase):
    """Validate cuLA verify/commit kernels against pure PyTorch reference."""

    # ------------------------------------------------------------------
    # Verify kernel: output matches PyTorch recurrent reference
    # ------------------------------------------------------------------
    def test_cula_verify_matches_pytorch(self):
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        torch.manual_seed(0)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        # fp32 q/k/v: the cuLA verify kernel now loads q/k into fp32 registers
        # natively (no bf16 cast), matching the runtime path (Ling fp32 rotary).
        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        state_kmajor = (
            torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01
        )

        # Reference
        o_ref, _, _ = _torch_la_mtp_ref(q, k, v, state_kmajor, decay, scale, T)

        # cuLA
        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        out_cula = torch.zeros(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        linear_attention_verify_kvbuffer(
            q.view(B, T, H, D),
            k.view(B, T, H, D),
            v.view(B, T, H, D),
            s_cula,
            out_cula,
            decay,
            pool_idx,
            scale,
            T,
        )

        rel = (out_cula.float() - o_ref.view(B, T, H, D).float()).pow(
            2
        ).mean().sqrt() / (o_ref.float().abs().max() + 1e-8)
        self.assertLess(
            rel.item(), 1e-2, f"verify output mismatch vs pytorch ref: {rel:.5f}"
        )

    # ------------------------------------------------------------------
    # Commit kernel: full accept state matches reference after T steps
    # ------------------------------------------------------------------
    def test_cula_commit_full_accept(self):
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        torch.manual_seed(42)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        state_kmajor = (
            torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01
        )

        # Reference state after T steps
        _, ref_state_kmajor, _ = _torch_la_mtp_ref(
            q, k, v, state_kmajor, decay, scale, T
        )

        # cuLA commit (full accept)
        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        accepted_len = torch.full((B,), T, device="cuda", dtype=torch.int32)
        linear_attention_state_update_kvbuffer(
            k.view(B, T, H, D),
            v.view(B, T, H, D),
            s_cula,
            decay,
            pool_idx,
            accepted_len,
            T,
        )
        cula_state_kmajor = s_cula.transpose(-1, -2).contiguous()

        rel = (cula_state_kmajor - ref_state_kmajor).pow(2).mean().sqrt() / (
            ref_state_kmajor.abs().max() + 1e-8
        )
        self.assertLess(
            rel.item(), 1e-3, f"commit state mismatch vs pytorch ref: {rel:.5f}"
        )

    # ------------------------------------------------------------------
    # Commit kernel: partial accept state matches reference at step L
    # ------------------------------------------------------------------
    def test_cula_commit_partial_accept(self):
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        L = 2
        torch.manual_seed(123)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        state_kmajor = (
            torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01
        )

        # Reference intermediate state at step L (partial accept)
        _, _, inter = _torch_la_mtp_ref(q, k, v, state_kmajor, decay, scale, T)
        ref_state_at_L = inter[:, L - 1]  # [B, HV, D, D]

        # cuLA commit (partial accept)
        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        accepted_len = torch.full((B,), L, device="cuda", dtype=torch.int32)
        linear_attention_state_update_kvbuffer(
            k.view(B, T, H, D),
            v.view(B, T, H, D),
            s_cula,
            decay,
            pool_idx,
            accepted_len,
            T,
        )
        cula_state_kmajor = s_cula.transpose(-1, -2).contiguous()

        rel = (cula_state_kmajor - ref_state_at_L).pow(2).mean().sqrt() / (
            ref_state_at_L.abs().max() + 1e-8
        )
        self.assertLess(
            rel.item(),
            1e-3,
            f"partial commit (L={L}) state mismatch vs pytorch ref: {rel:.5f}",
        )

    # ------------------------------------------------------------------
    # Commit kernel: zero accept (L=0) — state unchanged
    # ------------------------------------------------------------------
    def test_cula_commit_zero_accept(self):
        _skip_if_no_gpu()
        B, T, H, D = 2, 4, 4, 128
        torch.manual_seed(99)
        scale = D**-0.5
        decay = 0.3 * torch.arange(H, device="cuda", dtype=torch.float32) / H

        q = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B * T, H, D, device="cuda", dtype=torch.float32)
        state_kmajor = (
            torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01
        )

        pool_idx = torch.arange(B, device="cuda", dtype=torch.int32)
        s_cula = state_kmajor.transpose(-1, -2).contiguous()
        s_snapshot = s_cula.clone()
        accepted_len = torch.zeros(B, device="cuda", dtype=torch.int32)

        linear_attention_state_update_kvbuffer(
            k.view(B, T, H, D),
            v.view(B, T, H, D),
            s_cula,
            decay,
            pool_idx,
            accepted_len,
            T,
        )
        self.assertTrue(
            torch.equal(s_cula, s_snapshot), "L=0 must leave state unchanged"
        )


if __name__ == "__main__":
    unittest.main()
