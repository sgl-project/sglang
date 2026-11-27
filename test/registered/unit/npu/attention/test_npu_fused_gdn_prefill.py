import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=5, suite="extra-a-test-npu")

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.fused_gdn_prefill_950pr import (
    build_gdn_950pr_chunk_meta,
    can_use_fused_gdn_prefill_950pr,
    fused_gdn_prefill_950pr,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.utils import is_npu


def _is_arch35_hardware() -> bool:
    if not is_npu():
        return False
    try:
        return is_npu_arch35()
    except ImportError:
        return False


def _make_dummy_layer(nk: int = 16, nv: int = 48, d: int = 128, device: str = "cpu"):
    return SimpleNamespace(
        num_q_heads=nk,
        num_k_heads=nk,
        num_v_heads=nv,
        head_q_dim=d,
        head_k_dim=d,
        head_v_dim=d,
        q_dim=nk * d,
        k_dim=nk * d,
        v_dim=nv * d,
        A_log=torch.randn((nv,), device=device, dtype=torch.float32) * 0.2 - 1.0,
        dt_bias=torch.randn((nv,), device=device, dtype=torch.bfloat16) * 0.1,
    )


class TestNPUFusedGDNPrefill950PR(CustomTestCase):
    def test_build_chunk_meta_varlen(self):
        cidx, coff, nt_total, total_valid = build_gdn_950pr_chunk_meta(
            [733, 64, 15, 0], device=torch.device("cpu"), chunk_size=64
        )
        # 733 -> 12 chunks, 64 -> 1 chunk, 15 -> 1 chunk, 0 -> 0 chunks
        self.assertEqual(nt_total, 14)
        self.assertEqual(total_valid, 812)
        self.assertEqual(cidx.shape, (14, 2))
        self.assertEqual(coff.tolist(), [0, 12, 13, 14, 14])
        self.assertEqual(cidx[0].tolist(), [0, 0])
        self.assertEqual(cidx[11].tolist(), [0, 11])
        self.assertEqual(cidx[12].tolist(), [1, 0])
        self.assertEqual(cidx[13].tolist(), [2, 0])

    def test_can_use_fused_gdn_prefill_guards(self):
        layer = _make_dummy_layer(nk=16, nv=48, d=128)
        mixed_qkv = torch.empty(
            (64, layer.q_dim + layer.k_dim + layer.v_dim), dtype=torch.bfloat16
        )
        a = torch.empty((64, 48), dtype=torch.bfloat16)
        b = torch.empty((64, 48), dtype=torch.bfloat16)
        ssm = torch.empty((4, 48, 128, 128), dtype=torch.bfloat16)
        chunk_meta = build_gdn_950pr_chunk_meta([64], device=torch.device("cpu"))

        with patch(
            "sglang.srt.hardware_backend.npu.attention.fused_gdn_prefill_950pr.is_npu_arch35",
            return_value=False,
        ):
            self.assertFalse(
                can_use_fused_gdn_prefill_950pr(layer, mixed_qkv, a, b, ssm, chunk_meta)
            )

        with patch(
            "sglang.srt.hardware_backend.npu.attention.fused_gdn_prefill_950pr.is_npu_arch35",
            return_value=True,
        ):
            with patch.object(
                envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL, "get", return_value=False
            ):
                self.assertFalse(
                    can_use_fused_gdn_prefill_950pr(
                        layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
            with patch.object(
                envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL, "get", return_value=True
            ):
                self.assertTrue(
                    can_use_fused_gdn_prefill_950pr(
                        layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
                # Reject non-128 head_dim
                bad_layer = _make_dummy_layer(nk=16, nv=48, d=64)
                self.assertFalse(
                    can_use_fused_gdn_prefill_950pr(
                        bad_layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
                # Reject None chunk_meta
                self.assertFalse(
                    can_use_fused_gdn_prefill_950pr(layer, mixed_qkv, a, b, ssm, None)
                )

    @unittest.skipUnless(
        _is_arch35_hardware(), "Requires Ascend 950 (arch35) NPU hardware"
    )
    def test_fused_gdn_prefill_varlen_padding_and_checkpoint_matches_cann(self):
        import torch_npu

        torch.manual_seed(123)
        for nk, nv in [(16, 48), (16, 32), (16, 16), (8, 32)]:
            d = 128
            seq_lens = [733, 129, 64, 15]
            b_size = len(seq_lens)
            total_valid = sum(seq_lens)
            pad_extra = 37
            t_padded = total_valid + pad_extra
            layer = _make_dummy_layer(nk=nk, nv=nv, d=d, device="npu")
            qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim

            mixed_qkv = (
                torch.randn((t_padded, qkv_dim), device="npu", dtype=torch.bfloat16)
                * 0.5
            )
            a = torch.randn((t_padded, nv), device="npu", dtype=torch.bfloat16) * 0.3
            b_in = torch.randn((t_padded, nv), device="npu", dtype=torch.bfloat16) * 0.5

            ssm_ref = (
                torch.randn((16, nv, d, d), device="npu", dtype=torch.bfloat16) * 0.1
            )
            ssm_opt = ssm_ref.clone()
            cache_idx = torch.tensor([1, 4, 7, 10], device="npu", dtype=torch.int32)
            ckpt_dst = torch.tensor([2, 5, -1, 11], device="npu", dtype=torch.int32)
            # Save chunk 10 (tok 640) for seq 0, chunk 2 (tok 128) for seq 1, skip seq 2, chunk 0 (init) for seq 3
            ckpt_idx = torch.tensor([10, 2, -1, 0], device="npu", dtype=torch.int32)

            cu_list = [0]
            for l_val in seq_lens:
                cu_list.append(cu_list[-1] + l_val)
            cu = torch.tensor(cu_list, device="npu", dtype=torch.int32)
            chunk_meta = build_gdn_950pr_chunk_meta(
                seq_lens, device=torch.device("npu")
            )

            o_opt = fused_gdn_prefill_950pr(
                mixed_qkv,
                a,
                b_in,
                layer,
                ssm_opt,
                cache_idx,
                cu,
                chunk_meta,
                ckpt_chunk_idx=ckpt_idx,
                ckpt_dst_slots=ckpt_dst,
            )
            torch.npu.synchronize()

            # Padded tail must be exact zero
            self.assertEqual(o_opt[total_valid:].abs().max().item(), 0.0)

            # Compare per sequence against CANN 9.2 npu_chunk_gated_delta_rule
            q_raw, k_raw, v_all = torch.split(
                mixed_qkv[:total_valid], [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            q_all = F.normalize(
                q_raw.view(total_valid, nk, d).float(), p=2, dim=-1, eps=1e-6
            ).to(torch.bfloat16)
            k_all = F.normalize(
                k_raw.view(total_valid, nk, d).float(), p=2, dim=-1, eps=1e-6
            ).to(torch.bfloat16)
            v_all = v_all.view(total_valid, nv, d)
            g_all = (
                -torch.exp(layer.A_log)
                * F.softplus(a[:total_valid].float() + layer.dt_bias.float())
            ).to(torch.float32)
            beta_all = torch.sigmoid(b_in[:total_valid].float()).to(torch.bfloat16)

            for i, l_val in enumerate(seq_lens):
                s, e = cu_list[i], cu_list[i + 1]
                h0 = ssm_ref[cache_idx[i] : cache_idx[i] + 1].contiguous()
                o_ref, ht_ref = torch_npu.npu_chunk_gated_delta_rule(
                    q_all[s:e].contiguous(),
                    k_all[s:e].contiguous(),
                    v_all[s:e].contiguous(),
                    beta=beta_all[s:e].contiguous(),
                    initial_state=h0,
                    actual_seq_lengths=torch.tensor(
                        [l_val], device="npu", dtype=torch.int32
                    ),
                    scale=d**-0.5,
                    g=g_all[s:e].contiguous(),
                )
                cos_o = F.cosine_similarity(
                    o_opt[s:e].float().flatten(), o_ref.float().flatten(), dim=0
                ).item()
                cos_h = F.cosine_similarity(
                    ssm_opt[cache_idx[i]].float().flatten(),
                    ht_ref[0].float().flatten(),
                    dim=0,
                ).item()
                self.assertGreater(
                    cos_o, 0.9998, f"nk={nk},nv={nv},seq={i}: o cos={cos_o}"
                )
                self.assertGreater(
                    cos_h, 0.9998, f"nk={nk},nv={nv},seq={i}: ht cos={cos_h}"
                )

                # Verify checkpoint state matches prefix[:ckpt_chunk * 64]
                ci = int(ckpt_idx[i].item())
                dst = int(ckpt_dst[i].item())
                if ci >= 0 and dst >= 0:
                    cut = ci * 64
                    if cut == 0:
                        h_ckpt_ref = h0[0]
                    else:
                        _, h_ckpt_ref_b = torch_npu.npu_chunk_gated_delta_rule(
                            q_all[s : s + cut].contiguous(),
                            k_all[s : s + cut].contiguous(),
                            v_all[s : s + cut].contiguous(),
                            beta=beta_all[s : s + cut].contiguous(),
                            initial_state=h0,
                            actual_seq_lengths=torch.tensor(
                                [cut], device="npu", dtype=torch.int32
                            ),
                            scale=d**-0.5,
                            g=g_all[s : s + cut].contiguous(),
                        )
                        h_ckpt_ref = h_ckpt_ref_b[0]
                    cos_c = F.cosine_similarity(
                        ssm_opt[dst].float().flatten(),
                        h_ckpt_ref.float().flatten(),
                        dim=0,
                    ).item()
                    self.assertGreater(
                        cos_c, 0.9998, f"nk={nk},nv={nv},seq={i}: ckpt cos={cos_c}"
                    )

    @unittest.skipUnless(
        _is_arch35_hardware(), "Requires Ascend 950 (arch35) NPU hardware"
    )
    def test_fused_gdn_prefill_adversarial_identical_keys(self):
        import torch_npu

        torch.manual_seed(7)
        nk, nv, d, seq_len = 16, 48, 128, 192
        layer = _make_dummy_layer(nk=nk, nv=nv, d=d, device="npu")
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim

        # Identical keys across all 192 tokens + high beta + near-zero decay (worst case for solve_tril)
        one_row = torch.randn((1, qkv_dim), device="npu", dtype=torch.bfloat16) * 0.5
        mixed_qkv = one_row.expand(seq_len, qkv_dim).contiguous()
        a = torch.full((seq_len, nv), -10.0, device="npu", dtype=torch.bfloat16)
        b_in = torch.full(
            (seq_len, nv), 3.0, device="npu", dtype=torch.bfloat16
        )  # sigmoid(3) ~ 0.95

        ssm_ref = torch.zeros((2, nv, d, d), device="npu", dtype=torch.bfloat16)
        ssm_opt = ssm_ref.clone()
        cache_idx = torch.tensor([0], device="npu", dtype=torch.int32)
        cu = torch.tensor([0, seq_len], device="npu", dtype=torch.int32)
        chunk_meta = build_gdn_950pr_chunk_meta([seq_len], device=torch.device("npu"))

        o_opt = fused_gdn_prefill_950pr(
            mixed_qkv, a, b_in, layer, ssm_opt, cache_idx, cu, chunk_meta
        )
        torch.npu.synchronize()
        self.assertFalse(torch.isnan(o_opt).any().item())
        self.assertFalse(torch.isinf(o_opt).any().item())
        self.assertFalse(torch.isnan(ssm_opt[0]).any().item())

        q_raw, k_raw, v_all = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q_all = F.normalize(
            q_raw.view(seq_len, nk, d).float(), p=2, dim=-1, eps=1e-6
        ).to(torch.bfloat16)
        k_all = F.normalize(
            k_raw.view(seq_len, nk, d).float(), p=2, dim=-1, eps=1e-6
        ).to(torch.bfloat16)
        v_all = v_all.view(seq_len, nv, d)
        g_all = (
            -torch.exp(layer.A_log) * F.softplus(a.float() + layer.dt_bias.float())
        ).to(torch.float32)
        beta_all = torch.sigmoid(b_in.float()).to(torch.bfloat16)
        o_ref, ht_ref = torch_npu.npu_chunk_gated_delta_rule(
            q_all.contiguous(),
            k_all.contiguous(),
            v_all.contiguous(),
            beta=beta_all.contiguous(),
            initial_state=ssm_ref[:1],
            actual_seq_lengths=torch.tensor([seq_len], device="npu", dtype=torch.int32),
            scale=d**-0.5,
            g=g_all.contiguous(),
        )
        cos_o = F.cosine_similarity(
            o_opt.float().flatten(), o_ref.float().flatten(), dim=0
        ).item()
        cos_h = F.cosine_similarity(
            ssm_opt[0].float().flatten(), ht_ref[0].float().flatten(), dim=0
        ).item()
        self.assertGreater(cos_o, 0.999, f"Adversarial identical-key o cos={cos_o}")
        self.assertGreater(cos_h, 0.999, f"Adversarial identical-key ht cos={cos_h}")

    @unittest.skipUnless(
        _is_arch35_hardware(), "Requires Ascend 950 (arch35) NPU hardware"
    )
    def test_fused_gdn_prefill_strongly_correlated_keys_accuracy(self):
        import torch_npu

        torch.manual_seed(0)
        nk, nv, d, seq_len = 16, 48, 128, 256
        layer = _make_dummy_layer(nk=nk, nv=nv, d=d, device="npu")
        layer.A_log = torch.full((nv,), -3.0, device="npu", dtype=torch.float32)
        layer.dt_bias = torch.full((nv,), -2.0, device="npu", dtype=torch.bfloat16)

        u1 = torch.randn((1, nk * d), device="npu", dtype=torch.bfloat16)
        q_raw = u1 + 0.2 * torch.randn(
            (seq_len, nk * d), device="npu", dtype=torch.bfloat16
        )
        k_raw = u1 + 0.2 * torch.randn(
            (seq_len, nk * d), device="npu", dtype=torch.bfloat16
        )
        v_raw = torch.randn((seq_len, nv * d), device="npu", dtype=torch.bfloat16) * 3.0
        mixed_qkv = torch.cat([q_raw, k_raw, v_raw], dim=-1).contiguous()
        a = torch.full((seq_len, nv), -4.0, device="npu", dtype=torch.bfloat16)
        b_in = torch.full((seq_len, nv), 2.5, device="npu", dtype=torch.bfloat16)

        ssm_ref = torch.zeros((1, nv, d, d), device="npu", dtype=torch.bfloat16)
        ssm_opt = ssm_ref.clone()
        cu = torch.tensor([0, seq_len], device="npu", dtype=torch.int32)
        chunk_meta = build_gdn_950pr_chunk_meta([seq_len], device=torch.device("npu"))

        o_opt = fused_gdn_prefill_950pr(
            mixed_qkv,
            a,
            b_in,
            layer,
            ssm_opt,
            torch.tensor([0], device="npu", dtype=torch.int32),
            cu,
            chunk_meta,
        )
        q_all = F.normalize(
            q_raw.view(seq_len, nk, d).float(), p=2, dim=-1, eps=1e-6
        ).to(torch.bfloat16)
        k_all = F.normalize(
            k_raw.view(seq_len, nk, d).float(), p=2, dim=-1, eps=1e-6
        ).to(torch.bfloat16)
        g_all = (
            -torch.exp(layer.A_log) * F.softplus(a.float() + layer.dt_bias.float())
        ).to(torch.float32)
        beta_all = torch.sigmoid(b_in.float()).to(torch.bfloat16)
        o_ref, ht_ref = torch_npu.npu_chunk_gated_delta_rule(
            q_all,
            k_all,
            v_raw.view(seq_len, nv, d),
            beta=beta_all,
            initial_state=ssm_ref,
            actual_seq_lengths=torch.tensor([seq_len], device="npu", dtype=torch.int32),
            scale=d**-0.5,
            g=g_all,
        )
        cos_o = F.cosine_similarity(
            o_opt.float().flatten(), o_ref.float().flatten(), dim=0
        ).item()
        cos_h = F.cosine_similarity(
            ssm_opt[0].float().flatten(), ht_ref[0].float().flatten(), dim=0
        ).item()
        self.assertGreater(cos_o, 0.999, f"Correlated-key o cos={cos_o}")
        self.assertGreater(cos_h, 0.999, f"Correlated-key ht cos={cos_h}")

        # Verify strict FP64 recurrent reference accuracy bar (<= CANN 9.2 error)
        q64 = q_all.cpu().double()
        k64 = k_all.cpu().double()
        v64 = v_raw.view(seq_len, nv, d).cpu().double()
        g64 = g_all.cpu().double()
        b64 = beta_all.cpu().double()
        h64 = torch.zeros((nv, d, d), dtype=torch.float64)
        o64 = torch.zeros((seq_len, nv, d), dtype=torch.float64)
        gqa = nv // nk
        scale = d**-0.5
        for t in range(seq_len):
            for vh in range(nv):
                kh = vh // gqa
                h64[vh] = h64[vh] * torch.exp(g64[t, vh])
                delta = (v64[t, vh] - h64[vh] @ k64[t, kh]) * b64[t, vh]
                h64[vh] = h64[vh] + torch.outer(delta, k64[t, kh])
                o64[t, vh] = (h64[vh] @ q64[t, kh]) * scale
        o_rel_fp64 = (o_opt.cpu().double() - o64).norm().item() / o64.norm().item()
        ht_rel_fp64 = (
            ssm_opt[0].cpu().double() - h64
        ).norm().item() / h64.norm().item()
        self.assertLessEqual(
            o_rel_fp64, 0.025, f"Correlated-key o_rel_fp64={o_rel_fp64}"
        )
        self.assertLessEqual(
            ht_rel_fp64, 0.015, f"Correlated-key ht_rel_fp64={ht_rel_fp64}"
        )

    @unittest.skipUnless(
        _is_arch35_hardware(), "Requires Ascend 950 (arch35) NPU hardware"
    )
    def test_decode_conv1d_gating_and_rmsnorm_strided_z_parity(self):
        import triton

        from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
            _fused_conv1d_decode_packed_gating_kernel,
            fused_rmsnorm_gated_strided_z,
        )

        torch.manual_seed(42)
        d = 128
        for nk, nv in [(16, 48), (16, 32)]:
            qkv_dim = (2 * nk + nv) * d
            qkvz_dim = qkv_dim + nv * d
            nv_pad = triton.next_power_of_2(nv)
            w_t = torch.randn((4, qkv_dim), device="npu", dtype=torch.bfloat16) * 0.1
            a_log = torch.randn((nv,), device="npu", dtype=torch.float32) * 0.2 - 1.0
            dt_bias = torch.randn((nv,), device="npu", dtype=torch.bfloat16) * 0.1

            # Verify conv1d+gating parity across bs=1 (bd=1024), bs=4/8 (bd=2048), including pad slot -1
            for bs, block_d in [(1, 1024), (4, 2048), (8, 2048)]:
                qkvz = torch.randn((bs, qkvz_dim), device="npu", dtype=torch.bfloat16)
                ba = torch.randn((bs, 2 * nv), device="npu", dtype=torch.bfloat16)
                state_ref = (
                    torch.randn((16, 3, qkv_dim), device="npu", dtype=torch.bfloat16)
                    * 0.1
                )
                state_opt = state_ref.clone()
                cache_idx = torch.arange(bs, device="npu", dtype=torch.int32)
                if bs >= 4:
                    cache_idx[-1] = -1
                m_out = torch.empty((bs, qkv_dim), device="npu", dtype=torch.bfloat16)
                g_out = torch.empty((bs, nv), device="npu", dtype=torch.float32)
                b_out = torch.empty((bs, nv), device="npu", dtype=torch.bfloat16)

                _fused_conv1d_decode_packed_gating_kernel[(bs * (qkv_dim // block_d),)](
                    qkvz,
                    ba,
                    w_t,
                    state_opt,
                    cache_idx,
                    a_log,
                    dt_bias,
                    m_out,
                    g_out,
                    b_out,
                    bs,
                    qkvz_dim,
                    qkv_dim,
                    nv,
                    nv_pad,
                    block_d,
                )
                torch.npu.synchronize()

                # Reference in PyTorch
                b_ref = ba[:, :nv]
                a_ref = ba[:, nv:].float()
                g_ref = -torch.exp(a_log) * F.softplus(a_ref + dt_bias.float())
                self.assertLess((g_out - g_ref).abs().max().item(), 1e-4)
                self.assertEqual((b_out - b_ref).abs().max().item(), 0.0)
                for bi in range(bs):
                    slot = int(cache_idx[bi].item())
                    x0 = qkvz[bi, :qkv_dim]
                    if slot >= 0:
                        s_old = state_ref[slot]
                        acc = (
                            s_old[0].float() * w_t[0].float()
                            + s_old[1].float() * w_t[1].float()
                            + s_old[2].float() * w_t[2].float()
                            + x0.float() * w_t[3].float()
                        )
                        y_ref = (acc * torch.sigmoid(acc)).to(torch.bfloat16)
                        self.assertLess(
                            (m_out[bi].float() - y_ref.float()).abs().max().item(),
                            1e-2,
                        )
                        self.assertEqual(
                            (state_opt[slot, 0] - s_old[1]).abs().max().item(), 0.0
                        )
                        self.assertEqual(
                            (state_opt[slot, 1] - s_old[2]).abs().max().item(), 0.0
                        )
                        self.assertEqual(
                            (state_opt[slot, 2] - x0).abs().max().item(), 0.0
                        )
                    else:
                        self.assertEqual((m_out[bi] - x0).abs().max().item(), 0.0)

            # Verify fused_rmsnorm_gated_strided_z parity across decode and prefill T
            rms_w = torch.randn((d,), device="npu", dtype=torch.bfloat16)
            for T in [1, 4, 16, 32, 512]:
                qkvz = torch.randn((T, qkvz_dim), device="npu", dtype=torch.bfloat16)
                core = torch.randn((T, nv, d), device="npu", dtype=torch.bfloat16)
                y_opt = fused_rmsnorm_gated_strided_z(
                    core, qkvz, rms_w, qkv_dim, nv, d, 1e-6
                )
                torch.npu.synchronize()
                x_f = core.float()
                rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                z_f = qkvz[:, qkv_dim:].view(T, nv, d).float()
                y_ref = (
                    ((x_f * rstd) * rms_w.float() * F.silu(z_f))
                    .to(torch.bfloat16)
                    .view(T, nv * d)
                )
                self.assertLess(
                    (y_opt.float() - y_ref.float()).abs().max().item(),
                    2e-2,
                    f"rmsnorm_z mismatch for nv={nv}, T={T}",
                )

    def test_decode_ones_cache_persistent_across_batch_sizes(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
            AscendGDNAttnBackend,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode

        backend = AscendGDNAttnBackend.__new__(AscendGDNAttnBackend)
        cached = {}
        for bs in [1, 2, 4, 8]:
            backend.forward_metadata = SimpleNamespace(
                mamba_cache_indices=torch.arange(bs, dtype=torch.int32)
            )
            backend.prepare_gdn_inputs(bs, ForwardMode.DECODE, None)
            t = backend._decode_ones_cache[bs]
            self.assertEqual(t.shape, (bs,))
            self.assertEqual(t.dtype, torch.int32)
            self.assertEqual(t.tolist(), [1] * bs)
            cached[bs] = (t, t.data_ptr())

        # Re-preparing another bs must preserve earlier bs tensor identities/addresses
        # so multi-bs NPU Decode Graph captures never reference freed tensors.
        for bs in [4, 1, 8, 2]:
            backend.forward_metadata = SimpleNamespace(
                mamba_cache_indices=torch.arange(bs, dtype=torch.int32)
            )
            backend.prepare_gdn_inputs(bs, ForwardMode.DECODE, None)
            for k, (orig_t, orig_ptr) in cached.items():
                self.assertIs(backend._decode_ones_cache[k], orig_t)
                self.assertEqual(backend._decode_ones_cache[k].data_ptr(), orig_ptr)


if __name__ == "__main__":
    unittest.main()
