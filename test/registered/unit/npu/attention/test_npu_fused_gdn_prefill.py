import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=5, suite="extra-a-test-npu")


def _has_sgl_kernel_fused_gdn() -> bool:
    if not is_npu():
        return False
    try:
        if not is_npu_arch35():
            return False
        from sgl_kernel_npu.operators.attention.gdn import (
            build_gdn_prefill_chunk_meta,
            fused_conv1d_decode_gating,
            fused_gdn_prefill,
            fused_rmsnorm_gated_strided_z,
        )

        return all(
            op is not None
            for op in (
                build_gdn_prefill_chunk_meta,
                fused_conv1d_decode_gating,
                fused_gdn_prefill,
                fused_rmsnorm_gated_strided_z,
            )
        )
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


class TestNPUFusedGDNPrefillIntegration(CustomTestCase):
    def test_can_use_fused_gdn_prefill_guards(self):
        from sglang.srt.hardware_backend.npu.attention import ascend_gdn_backend as agb

        layer = _make_dummy_layer(nk=16, nv=48, d=128)
        mixed_qkv = torch.empty(
            (64, layer.q_dim + layer.k_dim + layer.v_dim), dtype=torch.bfloat16
        )
        a = torch.empty((64, 48), dtype=torch.bfloat16)
        b = torch.empty((64, 48), dtype=torch.bfloat16)
        ssm = torch.empty((4, 48, 128, 128), dtype=torch.bfloat16)
        chunk_meta = (
            torch.zeros((1, 2), dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            1,
            64,
        )

        # Reject when sgl_kernel_npu operator is absent (older sgl-kernel-npu wheel)
        with patch.object(agb, "fused_gdn_prefill", None):
            self.assertFalse(
                agb.can_use_fused_gdn_prefill_950pr(
                    layer, mixed_qkv, a, b, ssm, chunk_meta
                )
            )

        dummy_op = object()
        with (
            patch.object(agb, "fused_gdn_prefill", dummy_op),
            patch.object(agb, "is_npu_arch35", return_value=False),
        ):
            self.assertFalse(
                agb.can_use_fused_gdn_prefill_950pr(
                    layer, mixed_qkv, a, b, ssm, chunk_meta
                )
            )

        with (
            patch.object(agb, "fused_gdn_prefill", dummy_op),
            patch.object(agb, "is_npu_arch35", return_value=True),
        ):
            with patch.object(
                envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL, "get", return_value=False
            ):
                self.assertFalse(
                    agb.can_use_fused_gdn_prefill_950pr(
                        layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
            with patch.object(
                envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL, "get", return_value=True
            ):
                self.assertTrue(
                    agb.can_use_fused_gdn_prefill_950pr(
                        layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
                bad_layer = _make_dummy_layer(nk=16, nv=48, d=64)
                self.assertFalse(
                    agb.can_use_fused_gdn_prefill_950pr(
                        bad_layer, mixed_qkv, a, b, ssm, chunk_meta
                    )
                )
                self.assertFalse(
                    agb.can_use_fused_gdn_prefill_950pr(
                        layer, mixed_qkv, a, b, ssm, None
                    )
                )

    def test_decode_ones_cache_persistent_across_batch_sizes(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
            AscendGDNAttnBackend,
        )

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

    @unittest.skipUnless(
        _has_sgl_kernel_fused_gdn(),
        "Requires Ascend 950 (arch35) hardware with sgl_kernel_npu fused GDN operators",
    )
    def test_sgl_kernel_fused_gdn_prefill_and_decode_integration(self):
        import torch_npu
        from sgl_kernel_npu.operators.attention.gdn import (
            build_gdn_prefill_chunk_meta,
            fused_conv1d_decode_gating,
            fused_gdn_prefill,
            fused_rmsnorm_gated_strided_z,
        )

        torch.manual_seed(123)
        nk, nv, d = 16, 48, 128
        seq_lens = [129, 64, 15]
        total_valid = sum(seq_lens)
        t_padded = total_valid + 19
        layer = _make_dummy_layer(nk=nk, nv=nv, d=d, device="npu")
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim

        mixed_qkv = (
            torch.randn((t_padded, qkv_dim), device="npu", dtype=torch.bfloat16) * 0.5
        )
        a = torch.randn((t_padded, nv), device="npu", dtype=torch.bfloat16) * 0.3
        b_in = torch.randn((t_padded, nv), device="npu", dtype=torch.bfloat16) * 0.5
        ssm_ref = torch.randn((8, nv, d, d), device="npu", dtype=torch.bfloat16) * 0.1
        ssm_opt = ssm_ref.clone()
        cache_idx = torch.tensor([1, 2, 3], device="npu", dtype=torch.int32)
        cu = torch.tensor([0, 129, 193, 208], device="npu", dtype=torch.int32)
        cidx, coff, _, tv = build_gdn_prefill_chunk_meta(
            seq_lens, device=torch.device("npu")
        )

        o_opt = fused_gdn_prefill(
            mixed_qkv,
            a,
            b_in,
            layer.A_log,
            layer.dt_bias,
            ssm_opt,
            cache_idx,
            cu,
            cidx,
            coff,
            nk,
            nv,
            total_valid_tokens=tv,
        )
        self.assertEqual(o_opt[total_valid:].abs().max().item(), 0.0)

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

        o_ref, ht_ref = torch_npu.npu_chunk_gated_delta_rule(
            q_all[:129].contiguous(),
            k_all[:129].contiguous(),
            v_all[:129].contiguous(),
            beta=beta_all[:129].contiguous(),
            initial_state=ssm_ref[1:2].contiguous(),
            actual_seq_lengths=torch.tensor([129], device="npu", dtype=torch.int32),
            scale=d**-0.5,
            g=g_all[:129].contiguous(),
        )
        cos_o = F.cosine_similarity(
            o_opt[:129].float().flatten(), o_ref.float().flatten(), dim=0
        ).item()
        cos_h = F.cosine_similarity(
            ssm_opt[1].float().flatten(), ht_ref[0].float().flatten(), dim=0
        ).item()
        self.assertGreater(cos_o, 0.9998)
        self.assertGreater(cos_h, 0.9998)

        # Verify decode conv1d+gating and strided-Z RMSNorm integration
        qkvz_dim = qkv_dim + nv * d
        qkvz = torch.randn((4, qkvz_dim), device="npu", dtype=torch.bfloat16)
        ba = torch.randn((4, 2 * nv), device="npu", dtype=torch.bfloat16)
        w_t = torch.randn((4, qkv_dim), device="npu", dtype=torch.bfloat16) * 0.1
        c_state = torch.randn((8, 3, qkv_dim), device="npu", dtype=torch.bfloat16) * 0.1
        c_idx = torch.tensor([0, 1, 2, -1], device="npu", dtype=torch.int32)
        m_out, g_out, b_out = fused_conv1d_decode_gating(
            qkvz, ba, w_t, c_state, c_idx, layer.A_log, layer.dt_bias, nv
        )
        self.assertEqual(m_out.shape, (4, qkv_dim))
        self.assertEqual(g_out.shape, (4, nv))
        self.assertEqual(b_out.shape, (4, nv))

        rms_w = torch.randn((d,), device="npu", dtype=torch.bfloat16)
        core = torch.randn((4, nv, d), device="npu", dtype=torch.bfloat16)
        y = fused_rmsnorm_gated_strided_z(core, qkvz, rms_w, qkv_dim, 1e-6)
        self.assertEqual(y.shape, (4, nv * d))

        # Regression: 4D outputs from chunk_gated_delta_rule ([1, T, NV, D]) and
        # recurrent_gated_delta_rule ([B, 1, NV, D]) reshape cleanly to [T, NV, D].
        for core_4d in (core.view(1, 4, nv, d), core.view(4, 1, nv, d)):
            y_4d = fused_rmsnorm_gated_strided_z(
                core_4d.reshape(-1, nv, d).contiguous(), qkvz, rms_w, qkv_dim, 1e-6
            )
            torch.testing.assert_close(y_4d, y, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
