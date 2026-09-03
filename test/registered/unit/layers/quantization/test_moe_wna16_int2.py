"""Unit tests for the 2-bit (int2) ``moe_wna16`` path on synthetic data: no
checkpoint, no server (small GPU tests).

  1. the exact 2-bit unpack expressions of ``fused_moe_kernel_gptq_awq`` (byte
     index ``offs_k // 4``, shift ``(offs_k % 4) * 2``, mask ``0x3``, zero point
     2) dequantize a synthetic 2-bit matrix bit-identically to a torch
     reference, including the real expert dims 2560 x 640 g128;
  2. ``to_word_ncontig`` re-arranges the loader's ``[E, N, K/4]`` uint8 layout
     into ``[E, K/16, N]`` int32 words with value k at bits ``2*(k % 16)`` of
     word ``k // 16`` (bit-exact, and invertible); CPU only;
  3. ``fused_moe(use_int2_w2a16=True)`` on the byte layout, symmetric (zero
     point 2) and with per-group ``qzeros``, against a torch MoE reference on
     the dequantized weights;
  4. the same weights in the N-contiguous word layout (int32 ``[E, K/16, N]``,
     scales ``[E, K/g, N]``) through ``fused_moe`` agree with the byte layout;
  5. ``moe_gemv_int2_tab`` (batch-1 GEMV through int64 address tables) against
     an fp32 reference: one shared activation with TOP_K > 1 (the w13 form) and
     per-row activations with TOP_K = 1 and routed weights (the w2 form), fp16
     and bf16 scales, N not a multiple of the block; ``make_tables`` addresses.

Tests that need real expert tensors (pinned-host tables, bandwidth) are not
ported here.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.expert_gemv import (
    make_tables,
    moe_gemv_int2_tab,
    to_word_ncontig,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

# bf16 outputs from fp32 accumulation in a different order: ~4e-3 relative
TOL = dict(atol=1e-3, rtol=2e-2)


# ---------------------------------------------------------------------- synthetic 2-bit data
def pack_2bit_along_last(q):
    """int [.., K] with values in 0..3 -> uint8 [.., K // 4]; value k at bits 2 * (k % 4)
    of byte k // 4 (the moe_wna16 byte layout)."""
    return (
        q[..., 0::4] | (q[..., 1::4] << 2) | (q[..., 2::4] << 4) | (q[..., 3::4] << 6)
    ).to(torch.uint8)


def pack_words_along_k(q):
    """int [.., K] in 0..3 -> int32 [.., K // 16] words; value k at bits 2 * (k % 16)."""
    w = torch.zeros(*q.shape[:-1], q.shape[-1] // 16, dtype=torch.int32)
    for j in range(16):
        w |= (q[..., j::16].to(torch.int32) & 0x3) << (2 * j)
    return w


def make_int2_experts(e, n, k, group, has_zp, dtype, gen):
    """Synthetic experts in the moe_wna16 loader layout.

    Returns w_ref [e, n, k] (dequantized), qweight uint8 [e, n, k // 4],
    scales [e, n, k // group], qzeros uint8 [e, n // 4, k // group] or None.
    """
    q = torch.randint(0, 4, (e, n, k), generator=gen)
    scales = (torch.rand(e, n, k // group, generator=gen) * 0.1 + 0.01).to(dtype)
    if has_zp:
        zp = torch.randint(0, 4, (e, n, k // group), generator=gen)
        qzeros = pack_2bit_along_last(zp.transpose(1, 2)).transpose(1, 2).contiguous()
        zp_full = zp.repeat_interleave(group, dim=2)
    else:
        qzeros = None
        zp_full = torch.full_like(q, 2)
    w_ref = ((q - zp_full).float() * scales.float().repeat_interleave(group, dim=2)).to(
        dtype
    )
    return w_ref, pack_2bit_along_last(q), scales, qzeros


def torch_moe(a, w1, w2, score, topk):
    b, d = a.shape
    a = a.view(b, -1, d).repeat(1, topk, 1).reshape(-1, d)
    out = torch.zeros(b * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            h = a[mask] @ w1[i].transpose(0, 1)
            n = h.shape[1] // 2
            out[mask] = (F.silu(h[:, :n]) * h[:, n:]) @ w2[i].transpose(0, 1)
    return (
        out.view(b, -1, w2.shape[1]) * topk_weight.view(b, -1, 1).to(out.dtype)
    ).sum(dim=1)


def gemv_reference(w32, s, x, group=128):
    """w32 [T, K/16, N] int32 words, s [T, K/g, N], x [K] or [T, K] -> [T, N] fp32."""
    t, kw, n = w32.shape
    k = kw * 16
    q = torch.empty((t, k, n), dtype=torch.int32)
    for j in range(16):
        q[:, j::16, :] = (w32 >> (2 * j)) & 0x3
    w = (q.float() - 2.0) * s.float().repeat_interleave(group, dim=1)
    if x.dim() == 1:
        return torch.einsum("tkn,k->tn", w, x.float())
    return torch.einsum("tkn,tk->tn", w, x.float())


class TestWordLayoutCPU(CustomTestCase):
    def test_to_word_ncontig_bit_exact_and_invertible(self):
        gen = torch.Generator().manual_seed(7)
        for e, n, k in ((3, 64, 256), (2, 640, 2560), (1, 16, 64)):
            with self.subTest(shape=(e, n, k)):
                q = torch.randint(0, 4, (e, n, k), generator=gen)
                byte_layout = pack_2bit_along_last(q)  # [E, N, K/4]
                self.assertEqual(byte_layout.shape, (e, n, k // 4))
                word = to_word_ncontig(byte_layout)  # [E, K/16, N]
                self.assertEqual(word.dtype, torch.int32)
                self.assertEqual(word.shape, (e, k // 16, n))
                self.assertTrue(word.is_contiguous())
                expected = pack_words_along_k(q).transpose(1, 2).contiguous()
                self.assertTrue(torch.equal(word, expected))
                # every value lands at bits 2 * (k % 16) of word k // 16, column n
                for kk in (0, 1, 15, 16, 17, k - 1):
                    got = (word[:, kk // 16, :] >> (2 * (kk % 16))) & 0x3
                    self.assertTrue(torch.equal(got, q[:, :, kk].to(torch.int32)))
                # invertible: the little-endian byte view restores the byte layout
                back = word.view(torch.uint8).view(e, k // 16, n, 4)
                back = back.permute(0, 2, 1, 3).reshape(e, n, k // 4)
                self.assertTrue(torch.equal(back, byte_layout))


@unittest.skipUnless(torch.cuda.is_available(), "Triton MoE kernels require CUDA")
class TestInt2Unpack(CustomTestCase):
    def test_unpack_expressions_bit_exact(self):
        import triton
        import triton.language as tl

        @triton.jit
        def deq_w2(
            b_ptr,
            s_ptr,
            out_ptr,
            K,
            N,
            group_size,
            stride_bk,
            stride_bn,
            stride_sk,
            stride_sn,
            BLOCK_K: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            # the exact expressions of fused_moe_kernel_gptq_awq's int2 branch
            pid_k = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            b_ptrs = (
                b_ptr + (offs_k[:, None] // 4) * stride_bk + offs_n[None, :] * stride_bn
            )
            b_shifter = (offs_k[:, None] % 4) * 2
            m = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=m, other=0)
            b = (b >> b_shifter) & 0x3
            s_ptrs = (
                s_ptr
                + (offs_k[:, None] // group_size) * stride_sk
                + offs_n[None, :] * stride_sn
            )
            s = tl.load(s_ptrs, mask=m, other=0.0).to(tl.float32)
            val = (b.to(tl.float32) - 2) * s  # b_zp_num = 2 (symmetric)
            tl.store(out_ptr + offs_k[:, None] * N + offs_n[None, :], val, mask=m)

        gen = torch.Generator().manual_seed(7)
        for k, n, gs in ((2560, 640, 128), (640, 2560, 128), (256, 64, 64)):
            with self.subTest(K=k, N=n, group=gs):
                vals = torch.randint(0, 4, (k, n), dtype=torch.int32, generator=gen)
                sc = (torch.rand(k // gs, n, generator=gen) * 0.02 + 0.001).float()
                ref = (vals.float() - 2) * sc.repeat_interleave(gs, dim=0)
                # as MoeWNA16 loads it: [N, K/4] uint8, K packed along the last dim
                std = pack_2bit_along_last(vals.T.contiguous()).cuda()
                scg = sc.cuda()
                out = torch.empty((k, n), dtype=torch.float32, device="cuda")
                bk = bn = 64
                deq_w2[(triton.cdiv(k, bk), triton.cdiv(n, bn))](
                    std,
                    scg,
                    out,
                    k,
                    n,
                    gs,
                    std.stride(1),
                    std.stride(0),
                    scg.stride(0),
                    scg.stride(1),
                    BLOCK_K=bk,
                    BLOCK_N=bn,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out.cpu(), ref))


@unittest.skipUnless(torch.cuda.is_available(), "Triton MoE kernels require CUDA")
class TestFusedMoeInt2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _case(self, m, n, k, e, topk, group, has_zp, dtype, ncontig):
        gen = torch.Generator().manual_seed(m * 1000 + n + k + group + int(has_zp))
        dev = "cuda"
        a = (torch.randn((m, k), generator=gen) / 10).to(dtype).to(dev)
        score = torch.randn((m, e), generator=gen).to(dtype).to(dev)
        w1_ref, w1_q, w1_s, w1_zp = make_int2_experts(
            e, 2 * n, k, group, has_zp, dtype, gen
        )
        w2_ref, w2_q, w2_s, w2_zp = make_int2_experts(
            e, k, n, group, has_zp, dtype, gen
        )
        if ncontig:  # what process_weights_after_loading does for 2-bit experts
            w1_q, w2_q = to_word_ncontig(w1_q), to_word_ncontig(w2_q)
            w1_s = w1_s.transpose(1, 2).contiguous()
            w2_s = w2_s.transpose(1, 2).contiguous()
        to = lambda t: None if t is None else t.to(dev)
        # softmax weights without top-k renormalization, as torch_moe applies them
        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            topk_config=TopKConfig(top_k=topk, renormalize=False),
        )
        ref = torch_moe(a, w1_ref.to(dev), w2_ref.to(dev), score, topk)
        out = fused_moe(
            a.clone(),  # fused_experts writes its output over the input rows
            to(w1_q),
            to(w2_q),
            topk_output,
            use_int2_w2a16=True,
            w1_scale=to(w1_s),
            w2_scale=to(w2_s),
            w1_zp=to(w1_zp),
            w2_zp=to(w2_zp),
            block_shape=[0, group],
        )
        return out, ref

    def test_byte_layout_vs_torch_reference(self):
        for m in (1, 32):
            for k in (128, 256):
                for group in (64, 128):
                    for has_zp in (False, True):
                        with self.subTest(m=m, k=k, group=group, has_zp=has_zp):
                            out, ref = self._case(
                                m, 128, k, 8, 2, group, has_zp, torch.bfloat16, False
                            )
                            torch.testing.assert_close(out, ref, **TOL)

    def test_word_layout_matches_byte_layout(self):
        for m in (1, 32):
            for k in (128, 256):
                for group in (64, 128):
                    with self.subTest(m=m, k=k, group=group):
                        out_w, ref = self._case(
                            m, 128, k, 8, 2, group, False, torch.bfloat16, True
                        )
                        out_b, _ = self._case(
                            m, 128, k, 8, 2, group, False, torch.bfloat16, False
                        )
                        torch.testing.assert_close(out_w, ref, **TOL)
                        torch.testing.assert_close(out_w, out_b, **TOL)

    def test_real_expert_dims(self):
        # e=8 of the model's 512 experts, n=640, k=2560, g128, symmetric, both layouts
        for ncontig in (False, True):
            with self.subTest(ncontig=ncontig):
                out, ref = self._case(
                    4, 640, 2560, 8, 2, 128, False, torch.bfloat16, ncontig
                )
                torch.testing.assert_close(out, ref, **TOL)


@unittest.skipUnless(torch.cuda.is_available(), "Triton GEMV requires CUDA")
class TestInt2Gemv(CustomTestCase):
    def test_gemv_tab_vs_reference(self):
        gen = torch.Generator().manual_seed(0)
        e, k13, n13, k2, n2, topk = 8, 256, 200, 128, 256, 3
        for sdtype in (torch.float16, torch.bfloat16):
            with self.subTest(scale_dtype=sdtype):
                q13 = torch.randint(0, 4, (e, k13, n13), generator=gen)
                q2 = torch.randint(0, 4, (e, k2, n2), generator=gen)
                w13 = (
                    pack_words_along_k(q13.transpose(1, 2)).transpose(1, 2).contiguous()
                )
                w2 = pack_words_along_k(q2.transpose(1, 2)).transpose(1, 2).contiguous()
                s13 = (torch.rand(e, k13 // 128, n13, generator=gen) * 0.1 + 0.01).to(
                    sdtype
                )
                s2 = (torch.rand(e, k2 // 128, n2, generator=gen) * 0.1 + 0.01).to(
                    sdtype
                )
                self.assertEqual(w13.shape, (e, k13 // 16, n13))
                ids = torch.randperm(e, generator=gen)[:topk].to(torch.int32)
                x13 = torch.randn(k13, generator=gen).to(torch.bfloat16)
                x2 = torch.randn(topk, k2, generator=gen).to(torch.bfloat16)
                rw = torch.rand(topk, generator=gen) + 0.5
                ref13 = gemv_reference(w13[ids.long()], s13[ids.long()], x13)
                ref2 = gemv_reference(w2[ids.long()], s2[ids.long()], x2) * rw[:, None]
                w13d, s13d, w2d, s2d = (t.cuda() for t in (w13, s13, w2, s2))
                wt13, st13, sw13, ss13 = make_tables(w13d, s13d)
                wt2, st2, sw2, ss2 = make_tables(w2d, s2d)
                self.assertEqual((sw13, ss13), (w13d.stride(1), s13d.stride(1)))
                self.assertTrue(wt13.is_cuda and wt13.dtype == torch.int64)
                self.assertEqual(int(wt13[3]), w13d.data_ptr() + 3 * w13d.stride(0) * 4)
                self.assertEqual(int(st13[5]), s13d.data_ptr() + 5 * s13d.stride(0) * 2)
                ids_d, rw_d = ids.cuda(), rw.float().cuda()
                # w13 form: one shared [1, K] activation, TOP_K = topk maps row r -> a[0]
                o13 = moe_gemv_int2_tab(
                    x13[None].cuda(),
                    wt13,
                    st13,
                    ids_d,
                    rw_d,
                    n13,
                    k13,
                    sw13,
                    ss13,
                    top_k=topk,
                    mul_routed_weight=False,
                    scale_bf16=sdtype == torch.bfloat16,
                )
                # w2 form: per-row activations, TOP_K = 1, routed weights applied
                o2 = moe_gemv_int2_tab(
                    x2.cuda(),
                    wt2,
                    st2,
                    ids_d,
                    rw_d,
                    n2,
                    k2,
                    sw2,
                    ss2,
                    top_k=1,
                    mul_routed_weight=True,
                    scale_bf16=sdtype == torch.bfloat16,
                )
                torch.cuda.synchronize()
                self.assertEqual(o13.shape, (topk, n13))
                self.assertEqual(o13.dtype, torch.bfloat16)
                self.assertEqual(o2.shape, (topk, n2))
                for got, ref in ((o13.float().cpu(), ref13), (o2.float().cpu(), ref2)):
                    rel = ((got - ref).norm() / ref.norm()).item()
                    self.assertLess(rel, 1e-2, rel)  # bf16 output rounding ~4e-3

    def test_gemv_rejects_bad_block(self):
        with self.assertRaises(AssertionError):
            moe_gemv_int2_tab(
                torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda"),
                None,
                None,
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                None,
                8,
                64,
                8,
                8,
                top_k=1,
                mul_routed_weight=False,
            )


if __name__ == "__main__":
    unittest.main()
