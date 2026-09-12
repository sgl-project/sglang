import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.extend_attention_split_dim import (
    can_use_split_dim_absorbed_extend,
)
from sglang.srt.environ import envs
from sglang.srt.utils import get_device, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "Kimi-K3 Triton prefill requires gfx950"
)
class TestKimiK3TritonPrefill(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def test_split_dim_ragged_bf16(self):
        device = get_device()
        h_q, h_kv, d_qk, d_v = 12, 1, 576, 512
        extend_lens = (97, 128)
        prefix_lens = (53, 91)
        scale = 192**-0.5
        total_extend, total_prefix = sum(extend_lens), sum(prefix_lens)

        q = torch.randn(total_extend, h_q, d_qk, dtype=torch.bfloat16, device=device)
        k = torch.randn(total_extend, h_kv, d_qk, dtype=torch.bfloat16, device=device)
        v = torch.randn(total_extend, h_kv, d_v, dtype=torch.bfloat16, device=device)
        k_buffer = torch.randn(
            total_prefix, h_kv, d_qk, dtype=torch.bfloat16, device=device
        )
        v_buffer = torch.randn(
            total_prefix, h_kv, d_v, dtype=torch.bfloat16, device=device
        )
        qo_indptr = torch.tensor(
            [0, extend_lens[0], total_extend], dtype=torch.int32, device=device
        )
        kv_indptr = torch.tensor(
            [0, prefix_lens[0], total_prefix], dtype=torch.int32, device=device
        )
        kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)
        output = torch.empty(
            total_extend, h_q, d_v, dtype=torch.bfloat16, device=device
        )

        extend_attention_fwd(
            q,
            k,
            v,
            output,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max(extend_lens),
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=scale,
        )

        reference = torch.empty_like(output, dtype=torch.float32)
        for batch, (extend_len, prefix_len) in enumerate(zip(extend_lens, prefix_lens)):
            q_start = int(qo_indptr[batch])
            prefix_start = int(kv_indptr[batch])
            q_batch = q[q_start : q_start + extend_len].float()
            k_prefix = k_buffer[prefix_start : prefix_start + prefix_len, 0].float()
            v_prefix = v_buffer[prefix_start : prefix_start + prefix_len, 0].float()
            k_current = k[q_start : q_start + extend_len, 0].float()
            v_current = v[q_start : q_start + extend_len, 0].float()
            causal = torch.triu(
                torch.ones(
                    extend_len,
                    extend_len,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            for head in range(h_q):
                prefix_scores = q_batch[:, head] @ k_prefix.T * scale
                current_scores = q_batch[:, head] @ k_current.T * scale
                current_scores.masked_fill_(causal, float("-inf"))
                scores = torch.cat([prefix_scores, current_scores], dim=1)
                values = torch.cat([v_prefix, v_current], dim=0)
                reference[q_start : q_start + extend_len, head] = (
                    torch.softmax(scores, dim=-1) @ values
                )

        torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)

    def test_split_dim_dispatch_gates(self):
        device = get_device()
        q = torch.empty(1, 12, 576, dtype=torch.bfloat16, device=device)
        k = torch.empty(1, 1, 576, dtype=torch.bfloat16, device=device)
        v = torch.empty(1, 1, 512, dtype=torch.bfloat16, device=device)
        o = torch.empty(1, 12, 512, dtype=torch.bfloat16, device=device)
        k_buffer = torch.empty(1, 1, 576, dtype=torch.bfloat16, device=device)
        v_buffer = torch.empty(1, 1, 512, dtype=torch.bfloat16, device=device)
        kwargs = dict(
            lse=None,
            sinks=None,
            k_scale=1.0,
            v_scale=1.0,
            custom_mask=None,
            is_causal=True,
            sliding_window_size=-1,
            logit_cap=0.0,
            xai_temperature_len=-1,
            skip_prefix=False,
            skip_extend=False,
            page_size=1,
            score_mod=None,
            aux_tensors=None,
        )
        self.assertTrue(
            can_use_split_dim_absorbed_extend(q, k, v, o, k_buffer, v_buffer, **kwargs)
        )

        fp8_k_buffer = k_buffer.to(torch.float8_e4m3fn)
        fp8_v_buffer = v_buffer.to(torch.float8_e4m3fn)
        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(False):
            self.assertFalse(
                can_use_split_dim_absorbed_extend(
                    q, k, v, o, fp8_k_buffer, fp8_v_buffer, **kwargs
                )
            )
        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(True):
            self.assertTrue(
                can_use_split_dim_absorbed_extend(
                    q, k, v, o, fp8_k_buffer, fp8_v_buffer, **kwargs
                )
            )
            self.assertTrue(
                can_use_split_dim_absorbed_extend(
                    q,
                    k,
                    v,
                    o,
                    fp8_k_buffer,
                    fp8_v_buffer,
                    **{**kwargs, "k_scale": 0.5, "v_scale": 0.25},
                )
            )

        for override in (
            {"page_size": 2},
            {"logit_cap": 1.0},
            {"sliding_window_size": 128},
            {"skip_prefix": True},
            {"is_causal": False},
            {"lse": torch.empty(1, 12, dtype=torch.float32, device=device)},
            {"sinks": torch.empty(12, dtype=torch.float32, device=device)},
            {"k_scale": 0.5},
        ):
            self.assertFalse(
                can_use_split_dim_absorbed_extend(
                    q,
                    k,
                    v,
                    o,
                    k_buffer,
                    v_buffer,
                    **{**kwargs, **override},
                )
            )

    def test_zero_prefix_fp8_flag(self):
        device = get_device()
        tokens, heads, d_qk, d_v = 128, 12, 192, 128
        q = torch.randn(tokens, heads, d_qk, dtype=torch.bfloat16, device=device) * 0.25
        k = torch.randn(tokens, heads, d_qk, dtype=torch.bfloat16, device=device) * 0.25
        v = torch.randn(tokens, heads, d_v, dtype=torch.bfloat16, device=device) * 0.25
        k_buffer = torch.empty(1, heads, d_qk, dtype=torch.float8_e4m3fn, device=device)
        v_buffer = torch.empty(1, heads, d_v, dtype=torch.float8_e4m3fn, device=device)
        qo_indptr = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, 0], dtype=torch.int32, device=device)
        kv_indices = torch.empty(0, dtype=torch.int64, device=device)
        bf16_output = torch.empty(
            tokens, heads, d_v, dtype=torch.bfloat16, device=device
        )
        fp8_output = torch.empty_like(bf16_output)
        scale = d_qk**-0.5

        def run(output):
            extend_attention_fwd(
                q,
                k,
                v,
                output,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask=None,
                is_causal=True,
                mask_indptr=None,
                max_len_extend=tokens,
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=scale,
            )

        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(False):
            run(bf16_output)
        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(True):
            run(fp8_output)

        causal = (
            torch.arange(tokens, device=device)[None, :]
            <= torch.arange(tokens, device=device)[:, None]
        )

        def reference(q_ref, k_ref, v_ref):
            scores = torch.einsum("qhd,khd->qhk", q_ref, k_ref) * scale
            scores.masked_fill_(~causal[:, None, :], float("-inf"))
            return torch.einsum("qhk,khd->qhd", torch.softmax(scores, dim=-1), v_ref)

        bf16_reference = reference(q.float(), k.float(), v.float())
        fp8_reference = reference(
            q.to(torch.float8_e4m3fn).float(),
            k.to(torch.float8_e4m3fn).float(),
            v.to(torch.float8_e4m3fn).float(),
        )
        torch.testing.assert_close(
            bf16_output.float(), bf16_reference, rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            fp8_output.float(), fp8_reference, rtol=2e-2, atol=2e-2
        )

    def test_absorbed_fp8_prefix(self):
        device = get_device()
        tokens, prefix, heads, d_qk, d_v = 64, 73, 12, 576, 512
        q = torch.randn(tokens, heads, d_qk, dtype=torch.bfloat16, device=device) * 0.25
        k = torch.randn(tokens, 1, d_qk, dtype=torch.bfloat16, device=device) * 0.25
        v = torch.randn(tokens, 1, d_v, dtype=torch.bfloat16, device=device) * 0.25
        k_buffer = (
            torch.randn(prefix, 1, d_qk, dtype=torch.bfloat16, device=device) * 0.25
        ).to(torch.float8_e4m3fn)
        v_buffer = (
            torch.randn(prefix, 1, d_v, dtype=torch.bfloat16, device=device) * 0.25
        ).to(torch.float8_e4m3fn)
        qo_indptr = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, prefix], dtype=torch.int32, device=device)
        kv_indices = torch.arange(prefix, dtype=torch.int64, device=device)
        output = torch.empty(tokens, heads, d_v, dtype=torch.bfloat16, device=device)
        generic_output = torch.empty_like(output)
        scale, k_scale, v_scale = 192**-0.5, 0.5, 0.25

        def run(candidate):
            extend_attention_fwd(
                q,
                k,
                v,
                candidate,
                k_buffer,
                v_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                custom_mask=None,
                is_causal=True,
                mask_indptr=None,
                max_len_extend=tokens,
                k_scale=k_scale,
                v_scale=v_scale,
                sm_scale=scale,
            )

        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(False):
            run(generic_output)
        with envs.SGLANG_TRITON_FP8_PREFILL_ATTN.override(True):
            run(output)

        q_fp8 = q.to(torch.float8_e4m3fn).float()
        prefix_scores = (
            torch.einsum("qhd,kd->qhk", q_fp8, k_buffer[:, 0].float()) * scale * k_scale
        )
        current_scores = torch.einsum("qhd,kd->qhk", q.float(), k[:, 0].float())
        current_scores *= scale
        causal = (
            torch.arange(tokens, device=device)[None, :]
            <= torch.arange(tokens, device=device)[:, None]
        )
        current_scores.masked_fill_(~causal[:, None, :], float("-inf"))
        scores = torch.cat([prefix_scores, current_scores], dim=-1)
        values = torch.cat([v_buffer[:, 0].float() * v_scale, v[:, 0].float()], dim=0)
        reference = torch.einsum("qhk,kv->qhv", torch.softmax(scores, dim=-1), values)
        torch.testing.assert_close(output.float(), reference, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            generic_output.float(), reference, rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            output.float(), generic_output.float(), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
