import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.flash_attention_v3 import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionKernels,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


class TestFlashAttentionCustomKernels(CustomTestCase):
    def test_custom_kernels_get_every_call(self):
        """Kernels passed to the backend run its prefill and decode calls."""
        torch.manual_seed(0)
        tokens, heads, head_dim = 17, 4, 128
        calls = []

        def recorded(name, kernel):
            def run(**kwargs):
                calls.append(name)
                return kernel(**kwargs)

            return run

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
            req_to_token = torch.arange(1, 65, device="cuda", dtype=torch.int32)
            req_pool = SimpleNamespace(req_to_token=req_to_token.view(1, -1))
            runner = SimpleNamespace(
                sliding_window_size=None,
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
                    token_to_kv_pool_allocator=SimpleNamespace(
                        hands_out_virtual_ids=False
                    ),
                    token_to_kv_pool=pool,
                    page_size=1,
                    device="cuda",
                ),
            )
            backend = FlashAttentionBackend(
                runner,
                kernels=FlashAttentionKernels(
                    varlen_func=recorded("varlen", flash_attn_varlen_func),
                    with_kvcache_func=recorded("kvcache", flash_attn_with_kvcache),
                ),
            )
            reference = FlashAttentionBackend(runner)
            layer = RadixAttention(
                heads, head_dim, head_dim**-0.5, num_kv_heads=1, layer_id=0
            )
            extend = ForwardBatch(
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
            decode = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=extend.input_ids[-1:],
                req_pool_indices=extend.req_pool_indices,
                seq_lens=extend.seq_lens,
                seq_lens_cpu=extend.seq_lens_cpu,
                seq_lens_sum=tokens,
                out_cache_loc=extend.out_cache_loc[-1:],
            )
            q = torch.randn(
                tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(tokens, 1, head_dim, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)

            outputs = []
            for b in (backend, reference):
                b.init_forward_metadata(extend)
                prefill = b.forward_extend(q, k, v, layer, extend)
                b.init_forward_metadata(decode)
                outputs.append(
                    (prefill, b.forward_decode(q[-1:], None, None, layer, decode))
                )

        self.assertEqual(calls, ["kvcache", "kvcache"])
        for actual, expected in zip(*outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
