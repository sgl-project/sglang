import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.iquest_q1 import IQuestQ1Attention, _apply_learned_sink
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def torch_reference(query, sink_key, attn_output, lse, scale):
    group = query.shape[1] // sink_key.shape[0]
    sink = sink_key.to(query.dtype).repeat_interleave(group, dim=0)
    sink_logit = (query.float() * sink.float()).sum(-1) * scale
    factor = torch.sigmoid(lse.float() - sink_logit)
    return (attn_output.float() * factor.unsqueeze(-1)).to(attn_output.dtype)


class TestIQuestQ1LearnedSink(CustomTestCase):
    def test_fa3_lse_and_sink_match_joint_softmax(self):
        torch.manual_seed(29)
        tokens, heads, head_dim = 37, 6, 128
        with get_context().override_server_args():
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
                req_to_token=torch.arange(1, 65, device="cuda", dtype=torch.int32).view(
                    1, -1
                )
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
                attn_cp_size=1,
                tp_size=1,
                is_draft_worker=False,
                kv_index_translator=KVIndexTranslator(
                    req_to_token=req_pool.req_to_token,
                    token_to_kv_pool_allocator=object(),
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

    def test_head_dim_128_matches_torch_reduction(self):
        torch.manual_seed(47)
        for tokens, dtype in itertools.product(
            (1, 8, 16384), (torch.float32, torch.float16, torch.bfloat16)
        ):
            with self.subTest(tokens=tokens, dtype=dtype):
                q = torch.randn(tokens, 6, 128, device="cuda", dtype=dtype)
                sink = torch.randn(1, 128, device="cuda", dtype=dtype) * 4
                output = torch.randn(q.shape, device="cuda", dtype=torch.float32)
                lse = torch.randn(tokens, 6, device="cuda") * 4
                expected = torch_reference(q, sink, output, lse, 128**-0.5)
                actual = _apply_learned_sink(q, sink, output, lse, 128**-0.5)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=5e-7)

    def test_matches_torch_across_layouts_and_dtypes(self):
        torch.manual_seed(17)
        shapes = (
            (0, 6, 1, 128),
            (1, 6, 1, 128),
            (7, 6, 1, 128),
            (8, 12, 2, 128),
            (9, 48, 8, 128),
            (37, 16, 16, 80),
            (1024, 6, 1, 128),
            (17, 8, 1, 256),
        )
        for shape, dtype, strided in itertools.product(
            shapes, (torch.float32, torch.float16, torch.bfloat16), (False, True)
        ):
            with self.subTest(shape=shape, dtype=dtype, strided=strided):
                tokens, heads, kv_heads, head_dim = shape
                step = 2 if strided else 1
                q = torch.randn(
                    tokens, heads, head_dim * step, device="cuda", dtype=dtype
                )[..., ::step]
                output = torch.randn_like(q)
                if strided:
                    output = torch.randn(
                        heads, tokens, head_dim * step, device="cuda", dtype=dtype
                    ).transpose(0, 1)[..., ::step]
                sink = torch.randn(kv_heads, head_dim * step, device="cuda")[:, ::step]
                lse = torch.randn(heads, tokens, device="cuda").T
                if not strided:
                    lse = lse.contiguous()
                original = output.clone()
                scale = head_dim**-0.5
                expected = torch_reference(q, sink, output, lse, scale)
                actual = _apply_learned_sink(q, sink, output, lse, scale)
                rtol = 3e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=5e-7)
                torch.testing.assert_close(output, original, rtol=0, atol=0)
                self.assertEqual(actual.shape, q.shape)
                self.assertEqual(actual.dtype, output.dtype)
                self.assertTrue(actual.is_contiguous())

    def test_model_projection_layout(self):
        torch.manual_seed(19)
        attn = IQuestQ1Attention.__new__(IQuestQ1Attention)
        torch.nn.Module.__init__(attn)
        attn.num_heads = 6
        attn.num_kv_heads = 1
        attn.head_dim = 128
        attn.q_size = 768
        attn.scaling = 128**-0.5
        attn.sink_k = torch.nn.Parameter(
            torch.randn(1, 128, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        q = torch.randn(8, 768, device="cuda", dtype=torch.bfloat16)
        output = torch.randn_like(q)
        lse = torch.randn(8, 6, device="cuda")
        expected = torch_reference(
            q.view(8, 6, 128), attn.sink_k, output.view(8, 6, 128), lse, attn.scaling
        ).view(8, 768)
        for value in (output, output.view(8, 6, 128)):
            with self.subTest(shape=value.shape):
                actual = attn._apply_zero_value_sink(q, value, lse)
                torch.testing.assert_close(
                    actual, expected, rtol=torch.finfo(q.dtype).eps, atol=5e-7
                )

    def test_extreme_lse(self):
        q = torch.ones(5, 6, 128, device="cuda")
        sink = torch.zeros(1, 128, device="cuda")
        output = torch.randn_like(q)
        lse = (
            torch.tensor([-float("inf"), -1000, 0, 1000, float("inf")], device="cuda")
            .unsqueeze(-1)
            .expand(5, 6)
        )
        actual = _apply_learned_sink(q, sink, output, lse, 128**-0.5)
        factors = torch.tensor([0, 0, 0.5, 1, 1], device="cuda")
        torch.testing.assert_close(
            actual, output * factors[:, None, None], rtol=0, atol=0
        )

    def test_torch_compile_fullgraph_preserves_outputs(self):
        torch.manual_seed(41)
        compiled = torch.compile(_apply_learned_sink, fullgraph=True)
        for tokens in (1, 8):
            with self.subTest(tokens=tokens):
                q = torch.randn(tokens, 6, 128, device="cuda", dtype=torch.bfloat16)
                sink = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
                output = torch.randn_like(q)
                lse = torch.randn(tokens, 6, device="cuda")
                for _ in range(2):
                    q.normal_()
                    sink.normal_()
                    output.normal_()
                    lse.normal_()
                    expected = _apply_learned_sink(q, sink, output, lse, 128**-0.5)
                    actual = compiled(q, sink, output, lse, 128**-0.5)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cuda_graph_replay_uses_updated_inputs(self):
        torch.manual_seed(31)
        for tokens in (1, 8):
            with self.subTest(tokens=tokens):
                q = torch.randn(tokens, 6, 128, device="cuda", dtype=torch.bfloat16)
                sink = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
                output = torch.randn_like(q)
                lse = torch.randn(tokens, 6, device="cuda")
                scale = 128**-0.5
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _apply_learned_sink(q, sink, output, lse, scale)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = _apply_learned_sink(q, sink, output, lse, scale)
                for _ in range(3):
                    q.normal_()
                    sink.normal_()
                    output.normal_()
                    lse.normal_()
                    expected = torch_reference(q, sink, output, lse, scale)
                    graph.replay()
                    torch.testing.assert_close(
                        actual, expected, rtol=torch.finfo(q.dtype).eps, atol=5e-7
                    )


if __name__ == "__main__":
    unittest.main()
