import math
import unittest

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()


class TestAdaFlashNpuGraph(unittest.TestCase):
    def test_ragged_query_lengths_rebind_without_changing_token_shape(self):
        torch.manual_seed(23)
        q = torch.randn(32, 2, 128, device="npu", dtype=torch.bfloat16)
        k = torch.randn(9, 128, 128, device="npu", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        blocks = torch.arange(9, device="npu", dtype=torch.int32).reshape(3, 3)
        mask = torch.ones((2048, 2048), device="npu", dtype=torch.bool).triu(1)

        def attention(query_ends, kv_lengths):
            return torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                block_table=blocks,
                block_size=128,
                num_heads=2,
                num_key_value_heads=1,
                input_layout="TND",
                actual_seq_lengths=query_ends,
                actual_seq_lengths_kv=kv_lengths,
                atten_mask=mask,
                sparse_mode=3,
                scale=1 / math.sqrt(128),
            )[0]

        for _ in range(3):
            attention([8, 16, 32], [128, 137, 270])
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            output = attention([8, 16, 32], [128, 137, 270])
        for query_ends, kv_lengths in [
            ([1, 17, 32], [121, 145, 269]),
            ([16, 32, 32], [136, 145, 1]),
            ([7, 16, 32], [127, 138, 270]),
        ]:
            graph.update(
                cpu_update_input=[
                    {
                        "actual_seq_lengths": query_ends,
                        "actual_seq_lengths_kv": kv_lengths,
                    }
                ]
            )
            graph.replay()
            torch.npu.synchronize()
            actual = output.clone()
            expected = attention(query_ends, kv_lengths)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
