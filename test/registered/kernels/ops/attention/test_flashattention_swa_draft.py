import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestFlashAttentionSWADraft(CustomTestCase):
    def test_multistep_swa_graph_updates_real_cache_and_attention(self):
        torch.manual_seed(53)
        prefix, steps, heads, dim = 8, 3, 6, 128
        with get_context().override_server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
            speculative_eagle_topk=1,
        ):
            pool = SWAKVPool(
                size=128,
                size_swa=256,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=dim,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[],
                device="cuda",
            )
            mapping = torch.arange(129, device="cuda", dtype=torch.int64) + 64
            mapping[0] = 0
            pool.register_mapping(mapping)
            req_pool = SimpleNamespace(
                req_to_token=torch.arange(1, 65, device="cuda", dtype=torch.int32).view(
                    1, -1
                )
            )
            runner = SimpleNamespace(
                sliding_window_size=4,
                model_config=SimpleNamespace(
                    is_encoder_decoder=False,
                    context_len=64,
                    attention_arch=AttentionArch.MHA,
                    is_local_attention_model=False,
                    head_dim=dim,
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
                is_draft_worker=True,
                kv_index_translator=KVIndexTranslator(
                    req_to_token=req_pool.req_to_token,
                    token_to_kv_pool_allocator=object(),
                    token_to_kv_pool=pool,
                    page_size=1,
                    device="cuda",
                ),
            )
            backends = [
                FlashAttentionBackend(
                    runner, speculative_step_id=i, speculative_num_steps=steps, topk=1
                )
                for i in range(steps - 1)
            ]
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=torch.zeros(1, device="cuda", dtype=torch.int64),
                req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
                seq_lens=torch.tensor([prefix], device="cuda"),
                seq_lens_cpu=torch.tensor([prefix]),
                seq_lens_sum=prefix,
                out_cache_loc=torch.arange(
                    prefix + 1, prefix + steps + 1, device="cuda"
                ),
                positions=torch.tensor([prefix], device="cuda"),
                spec_info=EagleDraftInput(),
            )
            layer = RadixAttention(
                heads, dim, dim**-0.5, num_kv_heads=1, layer_id=0, sliding_window_size=3
            )
            q = torch.randn(
                steps - 1, 1, heads, dim, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn(
                prefix + steps - 1, 1, dim, device="cuda", dtype=torch.bfloat16
            )
            v = torch.randn_like(k)
            for backend in backends:
                backend.init_cuda_graph_state(1, 1)
                backend.init_forward_metadata_out_graph(batch, in_capture=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for i, backend in enumerate(backends):
                    backend.forward_decode(
                        q[i],
                        k[prefix + i : prefix + i + 1],
                        v[prefix + i : prefix + i + 1],
                        layer,
                        batch,
                    )
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [
                    backend.forward_decode(
                        q[i],
                        k[prefix + i : prefix + i + 1],
                        v[prefix + i : prefix + i + 1],
                        layer,
                        batch,
                    )
                    for i, backend in enumerate(backends)
                ]
            for offset in (0, 32):
                req_pool.req_to_token.copy_(
                    torch.arange(
                        offset + 1, offset + 65, device="cuda", dtype=torch.int32
                    ).view(1, -1)
                )
                batch.out_cache_loc.copy_(
                    torch.arange(
                        offset + prefix + 1, offset + prefix + steps + 1, device="cuda"
                    )
                )
                pool.get_key_buffer(0).zero_()
                pool.get_value_buffer(0).zero_()
                locations = torch.arange(offset + 1, offset + prefix + 1, device="cuda")
                pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(locations, mapping[locations]),
                    k[:prefix],
                    v[:prefix],
                )
                for backend in backends:
                    backend.init_forward_metadata_out_graph(batch, in_capture=False)
                graph.replay()
                for i, output in enumerate(outputs):
                    end = prefix + i + 1
                    keys = k[end - 4 : end].float().expand(-1, heads, -1)
                    values = v[end - 4 : end].float().expand(-1, heads, -1)
                    logits = (
                        torch.einsum("thd,shd->hts", q[i].float(), keys) * layer.scaling
                    )
                    expected = torch.einsum("hts,shd->thd", logits.softmax(-1), values)
                    torch.testing.assert_close(
                        output.float().view_as(expected), expected, rtol=0.02, atol=0.01
                    )
                    slot = offset + prefix + i + 1 + 64
                    torch.testing.assert_close(
                        pool.get_key_buffer(0)[slot], k[prefix + i], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(0)[slot], v[prefix + i], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
