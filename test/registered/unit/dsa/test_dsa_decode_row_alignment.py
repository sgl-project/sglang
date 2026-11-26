"""Exercise partial Attention-DP row alignment through the GPU mapper."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA DSA mapper")
class TestDraftDSARowAlignment(unittest.TestCase):
    def test_partial_attention_dp_padding(self):
        real_rows, query_rows = 2, 4
        backend = object.__new__(DeepseekSparseAttnBackend)
        page_table = (
            torch.arange(real_rows * 8, device="cuda", dtype=torch.int32).reshape(
                real_rows, 8
            )
            + 100
        )
        topk = torch.tensor([[0, 3, -1, 7]], device="cuda", dtype=torch.int32).repeat(
            real_rows, 1
        )
        backend.forward_metadata = SimpleNamespace(page_table_1=page_table)
        backend.dsa_decode_impl = "triton"
        backend.hisparse_coordinator = None
        backend.use_fused_topk = False
        backend.attn_dcp_size = 1
        backend.dsa_shards_index_k = False
        backend._resolve_kpool_tail_backend = lambda indices, impl: impl
        backend._check_kpool_tail_backend = lambda *args: None
        backend.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                read=lambda _: SimpleNamespace(
                    key=lambda: torch.empty(0, device="cuda")
                )
            )
        )
        backend._forward_triton_decode = lambda **kwargs: kwargs["page_table_1"]
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=1,
            v_head_dim=2,
            head_dim=4,
            scaling=1.0,
        )
        q = torch.zeros(query_rows, 1, 4, device="cuda")
        actual = backend.forward_decode(
            q,
            None,
            None,
            layer,
            SimpleNamespace(),
            save_kv_cache=False,
            topk_indices=topk,
        )
        expected = torch.full((query_rows, 4), -1, device="cuda", dtype=torch.int32)
        expected[:real_rows, 0] = page_table[:, 0]
        expected[:real_rows, 1] = page_table[:, 3]
        expected[:real_rows, 3] = page_table[:, 7]
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
