import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as mla
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_backend(dtype, page_size):
    backend = object.__new__(mla.TRTLLMMLABackend)
    backend.backend = "trtllm-gen"
    backend.device = "cuda"
    backend.data_type = dtype
    backend.q_data_type = torch.bfloat16
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.qk_nope_head_dim = 128
    backend.kv_cache_dim = 576
    backend.num_q_heads = 16
    backend.page_size = page_size
    backend.max_context_len = 256
    backend.disable_chunked_prefix_cache = True
    backend._kv_shard_pool = None
    return backend


def make_batch(suffix):
    prefix = (31, 64, 0)
    prefix_tensor = torch.tensor(prefix, dtype=torch.int32, device="cuda")
    suffix_tensor = torch.tensor(suffix, dtype=torch.int32, device="cuda")
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(prefix),
        req_pool_indices=torch.arange(len(prefix), device="cuda"),
        seq_lens=prefix_tensor + suffix_tensor,
        extend_prefix_lens=prefix_tensor,
        extend_prefix_lens_cpu=list(prefix),
        extend_seq_lens=suffix_tensor,
        extend_seq_lens_cpu=list(suffix),
        out_cache_loc=None,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "TRTLLM-GEN MLA requires SM10x",
)
class TestPagedPrefillKernel(CustomTestCase):
    def test_ragged_suffix_matches_causal_reference(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for page_size in (32, 64):
                for suffix in ((3, 1, 7), (129, 1, 7)):
                    with self.subTest(dtype=dtype, page_size=page_size, suffix=suffix):
                        self._check(dtype, page_size, suffix)

    def _check(self, dtype, page_size, suffix):
        torch.manual_seed(42)
        backend = make_backend(dtype=dtype, page_size=page_size)
        batch = make_batch(suffix)
        num_tokens = sum(suffix)
        num_pages = backend.max_context_len // page_size
        pages = (torch.randperm(3 * num_pages, device="cuda") + 1).view(3, num_pages)
        positions = torch.arange(backend.max_context_len, device="cuda")
        backend.req_to_token = (
            pages[:, positions // page_size] * page_size + positions % page_size
        ).int()
        backend.kv_index_translator = SimpleNamespace(is_translating=False)
        backend.workspace_buffer = torch.zeros(
            150 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        backend._multi_ctas_kv_counter_buffer = (
            mla.make_persistent_multi_ctas_kv_counter_buffer(
                torch.device("cuda"), 16, 3
            )
        )
        parallel = SimpleNamespace(dcp_enabled=False, attn_cp_size=1)
        with patch.object(mla, "get_parallel", return_value=parallel):
            backend.use_native_paged_prefill = backend._supports_native_paged_prefill()
            backend.init_forward_metadata(batch)
        cache = torch.randn(4096, 1, 576, dtype=torch.bfloat16, device="cuda").to(dtype)
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: cache)
        q = torch.randn(num_tokens, 16, 576, dtype=torch.bfloat16, device="cuda")
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=16, scaling=192**-0.5)

        def run():
            return backend.forward_extend(
                q, None, None, layer, batch, save_kv_cache=False
            )

        expected = []
        offset = 0
        for req, (prefix, suffix) in enumerate(
            zip(batch.extend_prefix_lens_cpu, batch.extend_seq_lens_cpu)
        ):
            indices = backend.req_to_token[req, : prefix + suffix].long()
            kv = (
                cache.view(torch.uint8)[indices].view(dtype)[:, 0].float()
                if dtype == torch.float8_e4m3fn
                else cache[indices, 0].float()
            )
            query = q[offset : offset + suffix].to(dtype).float()
            scores = torch.einsum("qhd,kd->hqk", query, kv) * layer.scaling
            mask = (
                torch.arange(prefix + suffix, device="cuda")[None, :]
                > (prefix + torch.arange(suffix, device="cuda"))[:, None]
            )
            scores.masked_fill_(mask[None], float("-inf"))
            expected.append(
                torch.einsum("hqk,kd->qhd", scores.softmax(-1), kv[:, :512])
            )
            offset += suffix
        expected = torch.cat(expected).reshape(num_tokens, -1)
        out = run()
        torch.testing.assert_close(out.float(), expected, atol=0.025, rtol=0.025)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_out = run()
        graph.replay()
        torch.testing.assert_close(captured_out, out)
        q.mul_(0.5)
        out = run()
        graph.replay()
        torch.testing.assert_close(captured_out, out)


if __name__ == "__main__":
    unittest.main()
