"""Regression tests: decode kernels must keep sigmoid(beta) in fp32.

The GDN packed decode kernel and the GDN/KDA linear-replay decode kernel
used to round ``sigmoid(beta)`` through the activation dtype:

    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

The rounded beta feeds the delta-rule update of the persistent SSM state, so
the error accumulates over decode steps: after 1000 steps the state rel-L2
vs a beta-in-fp32 recurrence is ~1.6e-3, vs ~1e-7 without the round-trip.
Same fix as vllm-project/vllm#53877.

The oracles keep beta in fp32 on purpose -- a reference that computes
``torch.sigmoid(b.float()).to(dtype)`` carries the same round-trip and is
blind to this bug.
"""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-large-amd")


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def _ref_gdn_step(
    state, mixed_qkv, a, b, A_log, dt_bias, H, HV, K, V, scale=1.0, qk_l2norm=True
):
    """One GDN decode step in fp32; beta stays in fp32 (see module docstring).

    ``state`` is [HV, V, K]; ``mixed_qkv`` is the packed (q | k | v) row.
    ``qk_l2norm`` matches the kernel's ``use_qk_l2norm_in_kernel=True`` path:
    x / sqrt(x.x + 1e-6), applied before ``scale``.
    """
    qkv = mixed_qkv.float()
    q = qkv[: H * K].view(H, K)
    k = qkv[H * K : 2 * H * K].view(H, K)
    v = qkv[2 * H * K :].view(HV, V)
    if qk_l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale
    rep = HV // H
    k_hv = k.repeat_interleave(rep, dim=0)  # [HV, K]

    g = -torch.exp(A_log.float()) * _softplus(a.float() + dt_bias.float())  # [HV]
    beta = torch.sigmoid(b.float())  # [HV] -- no activation-dtype round-trip

    h = state * torch.exp(g)[:, None, None]
    v_corr = v - torch.einsum("hvk,hk->hv", h, k_hv)
    v_corr = v_corr * beta[:, None]
    return h + torch.einsum("hv,hk->hvk", v_corr, k_hv)


def _assert_keeps_fp32_beta(self, got: float, b_val: float = 0.5) -> None:
    """Assert ``got`` is the fp32 sigmoid, not the activation-dtype rounding."""
    fp32_ref = 1.0 / (1.0 + math.exp(-b_val))  # sigmoid(0.5) = 0.6224593...
    bf16_ref = float(torch.tensor(fp32_ref).to(torch.bfloat16).float())
    self.assertLess(abs(got - fp32_ref), 1e-6, "beta lost fp32 precision")
    # guard against silently reintroducing the round-trip: the bf16-rounded
    # value differs from the fp32 one by ~1.4e-3 here
    self.assertGreater(abs(got - bf16_ref), 1e-4, "beta was rounded through bf16")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDecodeKeepsBetaInFp32(CustomTestCase):
    """Degenerate config (zero state, unit inputs): after one step the stored
    state equals sigmoid(b) itself, so the state pool exposes the round-trip
    directly."""

    DTYPES = (torch.bfloat16, torch.float16)

    def test_packed_decode_keeps_beta_in_fp32(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                device = "cuda"
                mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
                a = torch.zeros((1, 1), device=device, dtype=dtype)
                b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
                params = torch.zeros((1,), device=device, dtype=dtype)
                idx = torch.ones((1,), device=device, dtype=torch.int32)
                state = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
                out = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=params,
                    dt_bias=params,
                    scale=1.0,
                    initial_state=state,
                    out=out,
                    ssm_state_indices=idx,
                )
                _assert_keeps_fp32_beta(self, state[1, 0, 0, 0].item())

    def test_linear_replayssm_decode_keeps_beta_in_fp32(self):
        from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
            fused_recurrent_linear_replayssm_decode,
        )

        # K=V=16 (not 1): the replayssm kernel needs BKT >= 16 for tl.dot.
        # With unit inputs and a zero state the flushed state is still
        # sigmoid(b) elementwise after one step.
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                device = "cuda"
                num_slots, L, K, V = 2, 1, 16, 16
                mixed_qkv = torch.ones((1, 2 * K + V), device=device, dtype=dtype)
                a = torch.zeros((1, 1), device=device, dtype=dtype)
                b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
                params = torch.zeros((1,), device=device, dtype=dtype)
                idx = torch.ones((1,), device=device, dtype=torch.int32)
                state = torch.zeros(
                    (num_slots, 1, V, K), device=device, dtype=torch.float32
                )
                d_cache = torch.zeros(
                    (num_slots, 1, L, V), device=device, dtype=torch.float32
                )
                k_cache = torch.zeros((num_slots, 1, L, K), device=device, dtype=dtype)
                g_cache = torch.zeros(
                    (num_slots, 1, L), device=device, dtype=torch.float32
                )
                out = torch.empty((1, 1, 1, V), device=device, dtype=dtype)
                write_pos = torch.zeros((1,), device=device, dtype=torch.int32)
                force_flush = torch.ones((1,), device=device, dtype=torch.int32)

                fused_recurrent_linear_replayssm_decode(
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    A_log=params,
                    dt_bias=params,
                    scale=1.0,
                    initial_state=state,
                    d_cache=d_cache,
                    k_cache=k_cache,
                    g_cache=g_cache,
                    out=out,
                    ssm_state_indices=idx,
                    write_pos=write_pos,
                    force_flush=force_flush,
                    nk=1,  # K=16 -> BK=16; nk=1 keeps BKT=16 for tl.dot
                )
                _assert_keeps_fp32_beta(self, state[1, 0, 0, 0].item())

    def test_packed_decode_beta_drift_over_1000_steps(self):
        """1000 decode steps vs a beta-in-fp32 torch recurrence."""
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        device, dtype = "cuda", torch.bfloat16
        B, H, HV, K, V, num_steps = 1, 16, 48, 128, 128, 1000
        g = torch.Generator(device=device).manual_seed(42)
        A_log = (torch.randn(HV, generator=g, device=device) * 0.5).to(dtype)
        dt_bias = (torch.randn(HV, generator=g, device=device) * 0.5).to(dtype)
        state = torch.zeros((2, HV, V, K), device=device, dtype=torch.float32)
        idx = torch.ones((B,), device=device, dtype=torch.int32)
        ref = torch.zeros((HV, V, K), device=device, dtype=torch.float32)

        for _ in range(num_steps):
            mixed_qkv = torch.randn(
                B, 2 * H * K + HV * V, generator=g, device=device, dtype=dtype
            )
            a = (
                torch.randn(B, HV, generator=g, device=device, dtype=dtype) * 0.5
            ).contiguous()
            b = torch.randn(B, HV, generator=g, device=device, dtype=dtype).contiguous()
            out = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=1.0,
                initial_state=state,
                out=out,
                ssm_state_indices=idx,
                # qk l2norm keeps the delta rule stable over 1000 steps
                # and is the path the real model uses
                use_qk_l2norm_in_kernel=True,
            )
            ref = _ref_gdn_step(
                ref, mixed_qkv[0], a[0], b[0], A_log, dt_bias, H, HV, K, V
            )

        got = state[idx[0]].float()
        rel_l2 = (got - ref).norm() / ref.norm()
        # pre-fix this measures ~1.6e-3; post-fix ~1e-7
        self.assertLess(
            rel_l2.item(), 1e-4, f"beta round-trip drift: rel-L2 {rel_l2.item():.3e}"
        )


if __name__ == "__main__":
    unittest.main()
