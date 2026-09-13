"""Coverage for the AITER FP8-Q unified-attention decode path."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from aiter.ops.triton.attention.unified_attention import unified_attention

    import sglang.srt.layers.attention.aiter_backend as aiter_backend
    from sglang.kernels.ops.quantization.fp8_kernel import (
        fp8_dtype,
        scaled_fp8_quant,
    )
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def get_kv_buffer(self, _layer_id):
        return self.k_cache, self.v_cache

    def get_key_buffer(self, _layer_id):
        return self.k_cache


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER unified attention")
class TestAiterFP8QUnifiedAttention(CustomTestCase):
    def _make_backend_case(self, branch, kv_cache_dtype=None):
        if kv_cache_dtype is None:
            kv_cache_dtype = fp8_dtype
        batch, num_q_heads, num_kv_heads, head_dim = 2, 2, 1, 8
        device = "cuda"
        scale = torch.tensor([0.02], dtype=torch.float32, device=device)
        k_cache = torch.zeros(
            batch,
            1,
            num_kv_heads,
            head_dim,
            dtype=kv_cache_dtype,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)

        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = branch == "mla"
        backend.kv_cache_is_vectorized_5d = branch == "vectorized"
        backend.use_triton_unified_attention = branch != "legacy"
        backend.kv_cache_dtype = kv_cache_dtype
        backend.input_dtype = torch.bfloat16
        backend.page_size = 1
        backend.scale = head_dim**-0.5
        backend.logits_soft_cap = 0.0
        backend.k_scale = scale
        backend.v_scale = scale
        backend.workspace_buffer = torch.empty(1, device=device)
        backend.max_num_partitions = 1
        backend.kv_last_page_len = torch.ones(batch, dtype=torch.int32, device=device)
        backend.token_to_kv_pool = _FakeKVPool(k_cache, v_cache)
        backend.forward_metadata = SimpleNamespace(
            kv_indices=torch.arange(batch, dtype=torch.int32, device=device).view(
                batch, 1
            ),
            swa_page_table=None,
            qo_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device),
            kv_last_page_len=backend.kv_last_page_len,
            max_q_len=1,
            work_metadata=None,
            work_indptr=None,
            work_info_set=None,
            reduce_indptr=None,
            reduce_final_map=None,
            reduce_partial_map=None,
            num_kv_splits=None,
        )
        backend._mla_decode_fwd_with_head_pad = mock.Mock(
            return_value=torch.empty(
                batch,
                num_q_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        backend._get_aiter_paged_ragged_kv_cache_dtype = mock.Mock(
            return_value="fp8_e4m3"
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=num_q_heads,
            tp_k_head_num=num_kv_heads,
            tp_v_head_num=num_kv_heads,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            k_scale=scale,
            v_scale=scale,
            sliding_window_size=-1,
            scaling=head_dim**-0.5,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            batch_size=batch,
            seq_lens=torch.ones(batch, dtype=torch.int32, device=device),
        )
        q = torch.randn(
            batch,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        return backend, layer, forward_batch, q

    def test_q_quantization_is_isolated_to_unified_attention(self):
        for branch in ("mla", "vectorized", "unified", "legacy"):
            with self.subTest(branch=branch):
                backend, layer, forward_batch, q = self._make_backend_case(branch)
                original_q = q.reshape(q.shape[0], -1).clone()
                sentinel_q = torch.full(
                    original_q.shape, 7, dtype=fp8_dtype, device=original_q.device
                )

                with (
                    mock.patch.object(
                        aiter_backend,
                        "scaled_fp8_quant",
                        return_value=(sentinel_q, layer.k_scale),
                    ) as quant,
                    mock.patch.object(aiter_backend, "unified_attention") as unified,
                    mock.patch.object(
                        aiter_backend, "forward_decode_vectorized_5d"
                    ) as vectorized,
                    mock.patch.object(
                        aiter_backend, "paged_attention_ragged"
                    ) as legacy,
                ):
                    output = backend.forward_decode(
                        q, None, None, layer, forward_batch, save_kv_cache=False
                    )

                self.assertEqual(output.numel(), q.numel())
                if branch == "unified":
                    quant.assert_called_once()
                    self.assertIs(quant.call_args.args[1], layer.k_scale)
                    self.assertIs(unified.call_args.kwargs["q_descale"], layer.k_scale)
                    torch.testing.assert_close(
                        unified.call_args.kwargs["q"].reshape(q.shape[0], -1),
                        sentinel_q,
                    )
                else:
                    quant.assert_not_called()
                    if branch == "mla":
                        observed_q = (
                            backend._mla_decode_fwd_with_head_pad.call_args.args[0]
                        )
                    elif branch == "vectorized":
                        observed_q = vectorized.call_args.args[1]
                    else:
                        observed_q = legacy.call_args.args[2]
                    torch.testing.assert_close(
                        observed_q.reshape(q.shape[0], -1), original_q
                    )

    def test_bf16_kv_keeps_bf16_q(self):
        backend, layer, forward_batch, q = self._make_backend_case(
            "unified", torch.bfloat16
        )
        original_q = q.reshape(q.shape[0], -1).clone()

        with (
            mock.patch.object(aiter_backend, "scaled_fp8_quant") as quant,
            mock.patch.object(aiter_backend, "unified_attention") as unified,
        ):
            output = backend.forward_decode(
                q, None, None, layer, forward_batch, save_kv_cache=False
            )

        self.assertEqual(output.numel(), q.numel())
        quant.assert_not_called()
        self.assertIsNone(unified.call_args.kwargs["q_descale"])
        observed_q = unified.call_args.kwargs["q"]
        self.assertEqual(observed_q.dtype, torch.bfloat16)
        torch.testing.assert_close(
            observed_q.reshape(q.shape[0], -1),
            original_q,
        )

    def test_fp8_q_kv_matches_bf16_reference_at_decode_shape(self):
        # This is the per-TP-rank production shape used by the full trace.
        batch, num_q_heads, num_kv_heads = 4, 16, 1
        seq_len, head_dim, page_size = 8192, 256, 16
        device = "cuda"
        torch.manual_seed(0)

        base = torch.randn(batch, head_dim, device=device, dtype=torch.float32)
        q = (
            base[:, None, :].expand(-1, num_q_heads, -1)
            + 0.01
            * torch.randn(
                batch, num_q_heads, head_dim, device=device, dtype=torch.float32
            )
        ).to(torch.bfloat16)
        k = 0.1 * torch.randn(
            batch,
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        k[:, 0, 0, :] = 2 * base
        k = k.to(torch.bfloat16)
        v = (
            0.75
            + 0.25
            * torch.randn(
                batch,
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)

        fp8_max = torch.finfo(fp8_dtype).max
        k_scale = (k.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        v_scale = (v.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
        q_fp8, _ = scaled_fp8_quant(q.reshape(batch, -1), k_scale)
        k_fp8, _ = scaled_fp8_quant(k.reshape(-1, head_dim), k_scale)
        v_fp8, _ = scaled_fp8_quant(v.reshape(-1, head_dim), v_scale)

        q_fp8 = q_fp8.view(batch, num_q_heads, head_dim)
        k_fp8 = k_fp8.view(-1, page_size, num_kv_heads, head_dim)
        v_fp8 = v_fp8.view(-1, page_size, num_kv_heads, head_dim)
        pages_per_seq = seq_len // page_size
        block_table = torch.arange(
            batch * pages_per_seq, dtype=torch.int32, device=device
        ).view(batch, pages_per_seq)
        output = torch.empty_like(q, dtype=torch.bfloat16)

        unified_attention(
            q=q_fp8,
            k=k_fp8,
            v=v_fp8,
            out=output,
            cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=device),
            seqused_k=torch.full((batch,), seq_len, dtype=torch.int32, device=device),
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=k_scale,
            k_descale=k_scale,
            v_descale=v_scale,
            sinks=None,
        )

        scores = torch.einsum(
            "bhd,btd->bht", q.float(), k[:, :, 0, :].float()
        ) / math.sqrt(head_dim)
        expected = torch.einsum(
            "bht,btd->bhd",
            torch.softmax(scores, dim=-1),
            v[:, :, 0, :].float(),
        )
        actual = output.float()

        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreater(expected.abs().mean().item(), 0.25)
        mismatch = (actual - expected).abs() > 0.15 + 0.15 * expected.abs()
        mismatch_fraction = mismatch.float().mean().item()
        self.assertLess(
            mismatch_fraction,
            0.005,
            f"FP8 mismatch fraction {mismatch_fraction:.4%} exceeds 0.5%; "
            f"max abs diff={(actual - expected).abs().max().item():.6f}",
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        self.assertGreater(cosine, 0.99)


if __name__ == "__main__":
    unittest.main()
