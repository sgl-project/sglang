"""Integration tests for Ascend 950 fused GDN speculative verify delegation & fallback."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention import (
    ascend_gdn_backend,
    ascend_hybrid_linear_attn_backend,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=20, suite="extra-a-test-npu")

_HAS_SGL_KERNEL_VERIFY = (
    ascend_gdn_backend._sgl_fused_gdn_verify is not None
    and ascend_gdn_backend._sgl_fused_conv1d_gdn_verify is not None
    and ascend_hybrid_linear_attn_backend._sgl_fused_gdn_commit is not None
)


class TestSpecUtilsMambaTrackSkip(unittest.TestCase):
    """Exhaustive boundary verification for the NPU CPU mamba-track skip invariant."""

    def test_cpu_page_boundary_skip_never_misses_real_crossing(self):
        from sglang.srt.platforms.npu import NPUSRTPlatform
        from sglang.srt.speculative.spec_utils import _can_skip_mamba_track_on_cpu

        self.assertTrue(NPUSRTPlatform().skip_untracked_mamba_commit_on_cpu())
        for track_interval in (16, 64, 128, 256):
            for draft_token_num in (4, 8):
                max_span = 2 * draft_token_num
                for sl_cpu in range(0, track_interval * 3 + 16):
                    should_skip = (
                        track_interval > max_span
                        and sl_cpu >= 0
                        and sl_cpu // track_interval
                        == (sl_cpu + max_span) // track_interval
                    )
                    if not should_skip:
                        continue
                    # Under 1-step overlap, true pre_verify is in [sl_cpu, sl_cpu + K]
                    # and accept_len is in [1, K].
                    for lag in range(0, draft_token_num + 1):
                        pre = sl_cpu + lag
                        for accept_len in range(1, draft_token_num + 1):
                            post = pre + accept_len
                            self.assertEqual(
                                pre // track_interval,
                                post // track_interval,
                                f"False skip! interval={track_interval} K={draft_token_num} "
                                f"sl_cpu={sl_cpu} pre={pre} post={post}",
                            )

        # Verify helper accepts safe batch and rejects boundary-crossing batch.
        safe_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
            tree_cache=SimpleNamespace(page_size=128),
        )
        crossing_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([10, 120], dtype=torch.int32),
            tree_cache=SimpleNamespace(page_size=128),
        )
        with patch(
            "sglang.srt.speculative.spec_utils.mamba_track_grid", return_value=128
        ):
            self.assertTrue(_can_skip_mamba_track_on_cpu(safe_batch, 2, 8))
            self.assertFalse(_can_skip_mamba_track_on_cpu(crossing_batch, 2, 8))


@unittest.skipUnless(
    is_npu() and torch.npu.is_available() and is_npu_arch35(),
    "Requires Ascend 950 (arch35) NPU",
)
class TestNpuFusedGdnVerifyDelegation(unittest.TestCase):
    def test_copy_tracked_and_rollback_wrappers_with_fallback(self):
        L, S, W, D = 2, 10, 6, 256
        B, draft_k = 4, 4
        conv = torch.randn(L, S, W, D, dtype=torch.bfloat16, device="npu")
        conv_ref = conv.clone()

        src = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="npu")
        dst = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="npu")
        steps = torch.tensor([2, -1, 0, -1], dtype=torch.int64, device="npu")

        # Active path (sgl_kernel_npu if installed, else PyTorch fallback)
        ascend_hybrid_linear_attn_backend._copy_tracked_conv_states(
            conv, src, dst, steps
        )
        # Explicit fallback path when sgl_kernel_npu lacks copy_tracked_conv_states
        with patch.object(
            ascend_hybrid_linear_attn_backend, "_sgl_copy_tracked_conv_states", None
        ):
            ascend_hybrid_linear_attn_backend._copy_tracked_conv_states(
                conv_ref, src, dst, steps
            )
        torch.testing.assert_close(conv, conv_ref, atol=0, rtol=0)

        # Rollback wrapper: active tail vs legacy fallback must agree on last 3 active rows
        rb_slots = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="npu")
        rb_steps = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
        conv_fb = conv.clone()
        ascend_hybrid_linear_attn_backend._rollback_conv_states(
            conv, rb_slots, rb_steps, draft_k
        )
        with patch.object(
            ascend_hybrid_linear_attn_backend,
            "_CONV_ROLLBACK_SUPPORTS_ACTIVE_TAIL",
            False,
        ):
            ascend_hybrid_linear_attn_backend._rollback_conv_states(
                conv_fb, rb_slots, rb_steps, draft_k
            )
        torch.testing.assert_close(
            conv[:, rb_slots, -3:, :], conv_fb[:, rb_slots, -3:, :], atol=0, rtol=0
        )

    def test_verify_gating_and_missing_kernel_fallback(self):
        nk, nv, D = 16, 48, 128
        qkv_dim = (2 * nk + nv) * D
        layer = SimpleNamespace(
            num_q_heads=nk,
            num_k_heads=nk,
            num_v_heads=nv,
            head_q_dim=D,
            head_k_dim=D,
            head_v_dim=D,
            A_log=torch.randn(nv, dtype=torch.float32, device="npu"),
            dt_bias=torch.randn(nv, dtype=torch.bfloat16, device="npu"),
            bias=None,
            conv_weights=torch.randn(qkv_dim, 1, 4, dtype=torch.bfloat16, device="npu"),
        )
        bs, k = 4, 4
        mixed_qkv = torch.randn(bs * k, qkv_dim, dtype=torch.bfloat16, device="npu")
        a = torch.randn(bs * k, nv, dtype=torch.bfloat16, device="npu")
        b = torch.randn(bs * k, nv, dtype=torch.bfloat16, device="npu")
        ssm = torch.randn(8, nv, D, D, dtype=torch.bfloat16, device="npu")

        # Missing sgl_kernel_npu operator must disable fused path cleanly.
        with patch.object(ascend_gdn_backend, "_sgl_fused_gdn_verify", None):
            self.assertFalse(
                ascend_gdn_backend._can_use_fused_gdn_verify(
                    layer, mixed_qkv, a, b, ssm, bs, k, topk=1
                )
            )
        # Env kill-switch must disable fused path cleanly.
        with patch.dict(os.environ, {"SGLANG_NPU_ENABLE_FUSED_GDN_VERIFY": "0"}):
            self.assertFalse(
                ascend_gdn_backend._can_use_fused_gdn_verify(
                    layer, mixed_qkv, a, b, ssm, bs, k, topk=1
                )
            )
        # topk > 1 or K > 16 must be rejected.
        self.assertFalse(
            ascend_gdn_backend._can_use_fused_gdn_verify(
                layer, mixed_qkv, a, b, ssm, bs, k, topk=2
            )
        )
        self.assertFalse(
            ascend_gdn_backend._can_use_fused_gdn_verify(
                layer, mixed_qkv, a, b, ssm, bs, draft_token_num=17, topk=1
            )
        )

    @unittest.skipUnless(
        _HAS_SGL_KERNEL_VERIFY,
        "Requires sgl-kernel-npu with fused_gdn_verify / fused_conv1d_gdn_verify / fused_gdn_commit",
    )
    def test_sgl_kernel_npu_end_to_end_verify_commit_matches_rgdr(self):
        from sgl_kernel_npu.operators.attention.gdn import (
            create_gdn_verify_workspace,
            fused_conv1d_gdn_verify,
            fused_gdn_commit,
            fused_gdn_gating_without_sigmoid,
            fused_gdn_verify,
            recurrent_gated_delta_rule,
        )
        from sgl_kernel_npu.operators.mamba import causal_conv1d

        torch.manual_seed(2026)
        L, bs, K, nk, nv, D = 2, 12, 4, 16, 48, 128
        qkv_dim = (2 * nk + nv) * D
        qkvz_dim = qkv_dim + nv * D
        num_slots = bs * 2 + 4

        ssm_all = (
            torch.randn(L, num_slots, nv, D, D, dtype=torch.bfloat16, device="npu")
            * 0.15
        )
        ssm_ref = ssm_all.clone()
        conv_all = (
            torch.randn(
                L, num_slots, K + 2, qkv_dim, dtype=torch.bfloat16, device="npu"
            )
            * 0.2
        )
        conv_ref = conv_all.clone()
        inter_all = torch.empty(L, bs * K, nv, D, D, dtype=torch.bfloat16, device="npu")

        ws = create_gdn_verify_workspace()
        self.assertTrue(
            ws.ensure_allocated(
                num_layers=L,
                min_bs=bs,
                k_ver=K,
                nk=nk,
                nv=nv,
                head_dim=D,
                device=ssm_all.device,
            )
        )

        cidx = torch.arange(1, bs + 1, dtype=torch.int32, device="npu")
        qsl = torch.arange(0, (bs + 1) * K, K, dtype=torch.int32, device="npu")
        sidx = torch.arange(bs * K, dtype=torch.int32, device="npu").view(bs, K)
        asl = torch.full((bs,), K, dtype=torch.int32, device="npu")
        nat = torch.ones((bs,), dtype=torch.int32, device="npu")
        nat_conv = torch.full((bs,), K, dtype=torch.int32, device="npu")

        for li in range(L):
            qkvz = (
                torch.randn(bs * K, qkvz_dim, dtype=torch.bfloat16, device="npu") * 0.4
            )
            ba = torch.randn(bs * K, 2 * nv, dtype=torch.bfloat16, device="npu") * 0.4
            w_t = torch.randn(4, qkv_dim, dtype=torch.bfloat16, device="npu") * 0.1
            A_log = torch.randn(nv, dtype=torch.float32, device="npu") * 0.2 - 1.0
            dt_bias = torch.randn(nv, dtype=torch.bfloat16, device="npu") * 0.1

            mixed_ref = causal_conv1d(
                qkvz[:, :qkv_dim].contiguous(),
                w_t,
                conv_states=conv_ref[li],
                bias=None,
                query_start_loc=qsl,
                cache_indices=cidx,
                num_accepted_tokens=nat_conv,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
            g_ref, beta_ref = fused_gdn_gating_without_sigmoid(
                A_log,
                ba[:, nv:].contiguous(),
                ba[:, :nv].contiguous(),
                dt_bias,
            )
            o_ref = recurrent_gated_delta_rule(
                mixed_ref.view(bs, K, qkv_dim),
                ssm_ref[li],
                beta=beta_ref.view(bs, K, nv).to(torch.bfloat16),
                scale=D**-0.5,
                actual_seq_lengths=asl,
                ssm_state_indices=sidx,
                nk=nk,
                nv=nv,
                intermediate_state=inter_all[li],
                cache_indices=cidx,
                num_accepted_tokens=nat,
                g=g_ref.view(bs, K, nv).to(torch.float32),
            )

            # Layer 0 tests fused_conv1d_gdn_verify; Layer 1 tests fused_gdn_verify.
            if li == 0:
                o_new = fused_conv1d_gdn_verify(
                    qkvz,
                    ba,
                    w_t,
                    conv_all[li],
                    A_log,
                    dt_bias,
                    ssm_all[li],
                    cidx,
                    batch_size=bs,
                    k_ver=K,
                    num_k_heads=nk,
                    num_v_heads=nv,
                    head_dim=D,
                    query_start_loc=qsl,
                    workspace=ws,
                    linear_layer_idx=li,
                )[0]
                torch.testing.assert_close(conv_all[li], conv_ref[li], atol=0, rtol=0)
            else:
                o_new = fused_gdn_verify(
                    mixed_ref.view(bs, K, qkv_dim),
                    ba[:, nv:].contiguous().view(bs, K, nv),
                    ba[:, :nv].contiguous().view(bs, K, nv),
                    A_log,
                    dt_bias,
                    ssm_all[li],
                    cidx,
                    num_k_heads=nk,
                    num_v_heads=nv,
                    head_dim=D,
                    workspace=ws,
                    linear_layer_idx=li,
                )[0]
            self.assertLess((o_new.float() - o_ref.float()).abs().max().item(), 1.5e-2)

        # Single-launch all-layer commit + track
        last_steps = torch.arange(bs, dtype=torch.int64, device="npu") % K
        dst_slots = cidx.to(torch.int64)
        track_slots = torch.arange(bs + 2, 2 * bs + 2, dtype=torch.int64, device="npu")
        track_steps = torch.clamp(last_steps - 1, min=0)
        track_steps[1::2] = -1

        fused_gdn_commit(
            ssm_all,
            ws.rec_k_all,
            ws.rec_v_all,
            ws.rec_g_all,
            dst_slots,
            last_steps,
            track_slots=track_slots,
            track_steps=track_steps,
        )
        for li in range(L):
            for bi in range(bs):
                ls = int(last_steps[bi].item())
                ref_s = inter_all[li, bi * K + ls]
                self.assertLess(
                    (ssm_all[li, int(dst_slots[bi])].float() - ref_s.float())
                    .abs()
                    .max()
                    .item(),
                    2e-2,
                )
                ts = int(track_steps[bi].item())
                if ts >= 0:
                    ref_t = inter_all[li, bi * K + ts]
                    self.assertLess(
                        (ssm_all[li, int(track_slots[bi])].float() - ref_t.float())
                        .abs()
                        .max()
                        .item(),
                        2e-2,
                    )


if __name__ == "__main__":
    unittest.main()
