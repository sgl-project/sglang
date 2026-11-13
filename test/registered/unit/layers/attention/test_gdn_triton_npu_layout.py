"""Unit tests for TritonGDNKernel.extend NPU SSM state layout adaptation.

On NPU, ``MambaPool`` and ``fused_sigmoid_gating_delta_rule_update_decode_npu``
store recurrent SSM states in ``[num_slots, HV, V, K]`` layout, whereas
``sgl_kernel_npu``'s ``chunk_gated_delta_rule_npu`` consumes ``initial_state``
and returns ``(final_state, h)`` in ``[N, HV, K, V]`` layout. This test pins
the ``[V, K] <-> [K, V]`` transpose boundary in ``TritonGDNKernel.extend`` on
CPU without requiring NPU hardware or model weights.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.attention.linear.kernels.gdn_triton as gdn_triton_mod
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_chunk_gated_delta_rule_npu(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    return_none_states: bool = False,
):
    """Reference step-by-step delta rule matching chunk_gated_delta_rule_npu's
    [N, HV, K, V] initial_state / final_state / h contract.
    """
    assert not head_first
    assert q.shape[0] == 1
    _, total_tokens, num_q_heads, head_k_dim = q.shape
    _, _, num_v_heads, head_v_dim = v.shape
    num_seqs = cu_seqlens.numel() - 1
    assert initial_state.shape == (num_seqs, num_v_heads, head_k_dim, head_v_dim), (
        f"Expected [N, HV, K, V]={(num_seqs, num_v_heads, head_k_dim, head_v_dim)}, "
        f"got {tuple(initial_state.shape)}"
    )
    assert initial_state.is_contiguous()

    scale = head_k_dim**-0.5
    outs = torch.empty_like(v, dtype=torch.float32)
    final_states = []
    chunk_states = []

    for seq_idx in range(num_seqs):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        # Internal state in [HV, K, V], matching chunk_gated_delta_rule_npu
        st_kv = initial_state[seq_idx].float().clone()
        for tok_offset, t in enumerate(range(bos, eos)):
            if tok_offset % 64 == 0:
                chunk_states.append(st_kv.clone())
            qt = q[0, t].float()
            kt = k[0, t].float()
            if use_qk_l2norm_in_kernel:
                qt = F.normalize(qt, p=2, dim=-1)
                kt = F.normalize(kt, p=2, dim=-1)
            qt = qt.repeat_interleave(num_v_heads // num_q_heads, dim=0) * scale
            kt = kt.repeat_interleave(num_v_heads // num_q_heads, dim=0)
            vt = v[0, t].float()
            gt = g[0, t].float().exp().view(num_v_heads, 1, 1)
            bt = beta[0, t].float().view(num_v_heads, 1)

            st_kv = st_kv * gt
            pred_v = torch.einsum("hkv,hk->hv", st_kv, kt)
            delta_v = (vt - pred_v) * bt
            st_kv = st_kv + torch.einsum("hk,hv->hkv", kt, delta_v)
            outs[0, t] = torch.einsum("hkv,hk->hv", st_kv, qt)
        final_states.append(st_kv)

    out_tensor = outs.to(v.dtype)
    if return_none_states:
        return out_tensor, None, None
    final_tensor = torch.stack(final_states, dim=0)
    h_tensor = torch.stack(chunk_states, dim=0).unsqueeze(0).to(v.dtype)
    return out_tensor, final_tensor, h_tensor


def _reference_vk_step(q, k, v, g, beta, state_vk):
    """Direct [HV, V, K] reference for a single sequence to verify orientation."""
    total_tokens, num_q_heads, head_k_dim = q.shape
    _, num_v_heads, head_v_dim = v.shape
    scale = head_k_dim**-0.5
    st_vk = state_vk.float().clone()
    outs = []
    chunks_vk = []
    for t in range(total_tokens):
        if t % 64 == 0:
            chunks_vk.append(st_vk.clone())
        qt = (
            F.normalize(q[t].float(), p=2, dim=-1).repeat_interleave(
                num_v_heads // num_q_heads, dim=0
            )
            * scale
        )
        kt = F.normalize(k[t].float(), p=2, dim=-1).repeat_interleave(
            num_v_heads // num_q_heads, dim=0
        )
        vt = v[t].float()
        gt = g[t].float().exp().view(num_v_heads, 1, 1)
        bt = beta[t].float().view(num_v_heads, 1)
        st_vk = st_vk * gt
        pred_v = torch.einsum("hvk,hk->hv", st_vk, kt)
        delta_v = (vt - pred_v) * bt
        st_vk = st_vk + torch.einsum("hv,hk->hvk", delta_v, kt)
        outs.append(torch.einsum("hvk,hk->hv", st_vk, qt))
    return torch.stack(outs, dim=0), st_vk, torch.stack(chunks_vk, dim=0)


class TestGdnTritonNpuLayout(CustomTestCase):
    def _make_batch(self, seq_lens, num_q_heads=2, num_v_heads=4, k_dim=16, v_dim=16):
        torch.manual_seed(42)
        total_tokens = sum(seq_lens)
        q = torch.randn(1, total_tokens, num_q_heads, k_dim, dtype=torch.float32)
        k = torch.randn(1, total_tokens, num_q_heads, k_dim, dtype=torch.float32)
        v = torch.randn(1, total_tokens, num_v_heads, v_dim, dtype=torch.float32)
        g = -torch.rand(1, total_tokens, num_v_heads, dtype=torch.float32) * 0.2
        beta = torch.sigmoid(
            torch.randn(1, total_tokens, num_v_heads, dtype=torch.float32)
        )
        cu = [0]
        for seq_len in seq_lens:
            cu.append(cu[-1] + seq_len)
        cu_seqlens = torch.tensor(cu, dtype=torch.int64)
        return q, k, v, g, beta, cu_seqlens

    def test_square_vk_pool_transposes_in_and_out(self):
        # K == V == 16: catches silent transpose corruption when K and V match.
        q, k, v, g, beta, cu_seqlens = self._make_batch([65, 20], k_dim=16, v_dim=16)
        ssm_states = torch.randn(5, 4, 16, 16, dtype=torch.float32)
        cache_indices = torch.tensor([3, 1], dtype=torch.int64)

        ref_out0, ref_st0, ref_h0 = _reference_vk_step(
            q[0, :65], k[0, :65], v[0, :65], g[0, :65], beta[0, :65], ssm_states[3]
        )
        ref_out1, ref_st1, ref_h1 = _reference_vk_step(
            q[0, 65:], k[0, 65:], v[0, 65:], g[0, 65:], beta[0, 65:], ssm_states[1]
        )

        kernel = TritonGDNKernel()
        with (
            patch.object(gdn_triton_mod, "is_cpu", return_value=False),
            patch.object(gdn_triton_mod, "is_npu", return_value=True),
            patch.object(
                gdn_triton_mod,
                "chunk_gated_delta_rule",
                side_effect=_fake_chunk_gated_delta_rule_npu,
            ),
        ):
            out, final_state, h = kernel.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=cu_seqlens,
            )

        self.assertTrue(final_state.is_contiguous())
        self.assertEqual(tuple(final_state.shape), (2, 4, 16, 16))
        self.assertEqual(tuple(h.shape), (1, 3, 4, 16, 16))
        torch.testing.assert_close(out[0, :65], ref_out0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[0, 65:], ref_out1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state[0], ref_st0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state[1], ref_st1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h[0, :2], ref_h0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h[0, 2:], ref_h1, atol=1e-5, rtol=1e-5)

    def test_rectangular_vk_pool_and_strided_envelope(self):
        # K != V (K=16, V=32) + non-contiguous page-major / envelope slot stride.
        q, k, v, g, beta, cu_seqlens = self._make_batch([66], k_dim=16, v_dim=32)
        backing = torch.randn(4, 2, 4, 32, 16, dtype=torch.float32)
        ssm_states = backing[:, 0]  # non-contiguous [num_slots, HV, V, K]
        self.assertFalse(ssm_states.is_contiguous())
        cache_indices = torch.tensor([2], dtype=torch.int64)

        ref_out, ref_st, ref_h = _reference_vk_step(
            q[0], k[0], v[0], g[0], beta[0], ssm_states[2]
        )

        kernel = TritonGDNKernel()
        with (
            patch.object(gdn_triton_mod, "is_cpu", return_value=False),
            patch.object(gdn_triton_mod, "is_npu", return_value=True),
            patch.object(
                gdn_triton_mod,
                "chunk_gated_delta_rule",
                side_effect=_fake_chunk_gated_delta_rule_npu,
            ),
        ):
            out, final_state, h = kernel.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=cu_seqlens,
            )

        self.assertEqual(tuple(final_state.shape), (1, 4, 32, 16))
        self.assertEqual(tuple(h.shape), (1, 2, 4, 32, 16))
        torch.testing.assert_close(out[0], ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state[0], ref_st, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h[0], ref_h, atol=1e-5, rtol=1e-5)

    def test_pretransposed_kv_view_is_preserved(self):
        q, k, v, g, beta, cu_seqlens = self._make_batch([12], k_dim=16, v_dim=32)
        phys_vk = torch.randn(3, 4, 32, 16, dtype=torch.float32)
        ssm_states_kv_view = phys_vk.transpose(-1, -2)  # stride(-2) == 1 < stride(-1)
        cache_indices = torch.tensor([1], dtype=torch.int64)

        ref_out, ref_st_vk, _ = _reference_vk_step(
            q[0], k[0], v[0], g[0], beta[0], phys_vk[1]
        )

        kernel = TritonGDNKernel()
        with (
            patch.object(gdn_triton_mod, "is_cpu", return_value=False),
            patch.object(gdn_triton_mod, "is_npu", return_value=True),
            patch.object(
                gdn_triton_mod,
                "chunk_gated_delta_rule",
                side_effect=_fake_chunk_gated_delta_rule_npu,
            ),
        ):
            out, final_state, h = kernel.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states_kv_view,
                cache_indices=cache_indices,
                query_start_loc=cu_seqlens,
            )

        self.assertEqual(tuple(final_state.shape), (1, 4, 16, 32))
        self.assertEqual(tuple(h.shape), (1, 1, 4, 16, 32))
        torch.testing.assert_close(out[0], ref_out, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            final_state[0].transpose(-1, -2), ref_st_vk, atol=1e-5, rtol=1e-5
        )

    def test_optional_none_outputs_and_mis_guard(self):
        q, k, v, g, beta, cu_seqlens = self._make_batch([8], k_dim=16, v_dim=16)
        ssm_states = torch.randn(2, 4, 16, 16, dtype=torch.float32)
        cache_indices = torch.tensor([0], dtype=torch.int64)
        kernel = TritonGDNKernel()

        with (
            patch.object(gdn_triton_mod, "is_cpu", return_value=False),
            patch.object(gdn_triton_mod, "is_npu", return_value=True),
            patch.object(
                gdn_triton_mod,
                "chunk_gated_delta_rule",
                side_effect=lambda **kw: _fake_chunk_gated_delta_rule_npu(
                    return_none_states=True, **kw
                ),
            ),
        ):
            out, final_state, h = kernel.extend(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=cu_seqlens,
            )
            self.assertEqual(tuple(out.shape), (1, 8, 4, 16))
            self.assertIsNone(final_state)
            self.assertIsNone(h)

            with self.assertRaises(NotImplementedError):
                kernel.extend(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=cu_seqlens,
                    inplace_update=False,
                )


if __name__ == "__main__":
    unittest.main()
