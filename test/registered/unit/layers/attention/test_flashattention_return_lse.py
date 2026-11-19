import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.iquest_q1 import _apply_learned_sink
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _make_backend(tokens, heads, head_dim):
    pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )
    req_pool = SimpleNamespace(
        req_to_token=torch.arange(1, 65, device="cuda", dtype=torch.int32).view(1, -1)
    )
    runner = SimpleNamespace(
        sliding_window_size=7,
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            context_len=64,
            attention_arch=AttentionArch.MHA,
            is_local_attention_model=False,
            head_dim=head_dim,
            hf_text_config=SimpleNamespace(num_attention_heads=heads),
            get_num_kv_heads=lambda _: 1,
        ),
        device="cuda",
        req_to_token_pool=req_pool,
        token_to_kv_pool=pool,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_dtype_str="auto",
        page_size=1,
        is_draft_worker=False,
        runner_ids=RunnerIds(
            req_to_token=req_pool.req_to_token,
            token_to_kv_pool_allocator=SimpleNamespace(hands_out_virtual_ids=False),
            token_to_kv_pool=pool,
            page_size=1,
            device="cuda",
        ),
    )
    backend = FlashAttentionBackend(runner)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(tokens, device="cuda", dtype=torch.int64),
        req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
        seq_lens=torch.tensor([tokens], device="cuda"),
        seq_lens_cpu=torch.tensor([tokens]),
        seq_lens_sum=tokens,
        out_cache_loc=torch.arange(1, tokens + 1, device="cuda"),
        extend_num_tokens=tokens,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[tokens],
        extend_seq_lens=torch.tensor([tokens], device="cuda"),
        extend_prefix_lens=torch.zeros(1, device="cuda", dtype=torch.int64),
        extend_start_loc=torch.zeros(1, device="cuda", dtype=torch.int64),
    )
    return backend, batch


class TestFlashAttentionReturnLse(CustomTestCase):
    def test_fa3_lse_and_sink_match_joint_softmax(self):
        torch.manual_seed(29)
        tokens, heads, head_dim = 37, 6, 128
        with get_context().override_server_args():
            backend, batch = _make_backend(tokens, heads, head_dim)
            q = torch.randn(
                tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(tokens, 1, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            sink = torch.randn(1, head_dim, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(tokens, device="cuda")
            for window in (-1, 7):
                layer = RadixAttention(
                    heads,
                    head_dim,
                    head_dim**-0.5,
                    num_kv_heads=1,
                    layer_id=0,
                    sliding_window_size=window,
                )
                backend.init_forward_metadata(batch)
                output, lse = backend.forward_extend(
                    q, k, v, layer, batch, return_lse=True
                )
                logits = (
                    torch.einsum(
                        "thd,shd->hts", q.float(), k.float().expand(-1, heads, -1)
                    )
                    * layer.scaling
                )
                mask = positions[None, :] <= positions[:, None]
                if window >= 0:
                    mask &= positions[None, :] >= positions[:, None] - window
                logits = logits.masked_fill(~mask, -float("inf"))
                torch.testing.assert_close(
                    lse, logits.logsumexp(-1).T, rtol=1e-5, atol=1e-5
                )
                sink_logits = (
                    torch.einsum("thd,kd->ht", q.float(), sink.float()) * layer.scaling
                )
                joint = torch.cat([logits, sink_logits[..., None]], -1).softmax(-1)
                expected = torch.einsum(
                    "hts,shd->thd", joint[..., :-1], v.float().expand(-1, heads, -1)
                )
                actual = _apply_learned_sink(
                    q, sink, output.view_as(q), lse, layer.scaling
                )
                torch.testing.assert_close(
                    actual.float(), expected, rtol=0.02, atol=0.01
                )
                decode = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=1,
                    input_ids=batch.input_ids[-1:],
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=batch.seq_lens_cpu,
                    seq_lens_sum=tokens,
                    out_cache_loc=batch.out_cache_loc[-1:],
                )
                backend.init_forward_metadata(decode)
                decoded, decode_lse = backend.forward_decode(
                    q[-1:], None, None, layer, decode, return_lse=True
                )
                torch.testing.assert_close(decode_lse, lse[-1:], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(decoded, output[-1:], rtol=0.02, atol=0.01)

    def test_layer_override_guards_and_compact_decode(self):
        torch.manual_seed(31)
        tokens, heads, head_dim = 37, 6, 128
        with get_context().override_server_args():
            backend, batch = _make_backend(tokens, heads, head_dim)
            q = torch.randn(
                tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(tokens, 1, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)

            def make_layer(**overrides):
                return RadixAttention(
                    heads,
                    head_dim,
                    head_dim**-0.5,
                    num_kv_heads=1,
                    layer_id=0,
                    **overrides,
                )

            backend.init_forward_metadata(batch)
            backend.forward_extend(q, k, v, make_layer(), batch)

            fp32_layer = make_layer(out_dtype=torch.float32)
            with self.assertRaises(ValueError):
                backend.forward_extend(q, None, None, fp32_layer, batch)
            batch._attn_output = torch.empty(
                tokens, heads * head_dim, device="cuda", dtype=torch.float32
            )
            try:
                with self.assertRaises(ValueError):
                    backend.forward_extend(q, None, None, fp32_layer, batch)
            finally:
                batch._attn_output = None

            decode = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=batch.input_ids[-1:],
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                seq_lens_sum=tokens,
                out_cache_loc=batch.out_cache_loc[-1:],
            )
            backend.init_forward_metadata(decode)
            with self.assertRaises(NotImplementedError):
                backend.forward_decode(
                    q[-1:],
                    None,
                    None,
                    make_layer(sliding_window_size=7, window_anchor_at_start=True),
                    decode,
                )

            rel_bias = torch.randn(1, heads, 128, device="cuda", dtype=torch.bfloat16)

            def combine(o, m, l):
                w = torch.exp(m - m.amax(0))
                return (o * w[..., None]).sum(0) / (l * w).sum(0)[..., None]

            decoded = backend.forward_decode(
                q[-1:],
                None,
                None,
                fp32_layer,
                decode,
                rel_bias=rel_bias,
                rel_bias_compact_decode=True,
                rel_bias_decode_epilogue=combine,
            )
            dist = torch.arange(tokens - 1, -1, -1, device="cuda")
            logits = (
                torch.einsum(
                    "hd,shd->hs", q[-1].float(), k.float().expand(-1, heads, -1)
                )
                * fp32_layer.scaling
                + rel_bias[0].float()[:, dist]
            )
            expected = torch.einsum(
                "hs,shd->hd", logits.softmax(-1), v.float().expand(-1, heads, -1)
            )
            torch.testing.assert_close(
                decoded.view(heads, head_dim), expected, rtol=0.02, atol=0.01
            )


if __name__ == "__main__":
    unittest.main()
