"""SM120 packed-cache DCP numerical and CUDA graph regression tests.

Run with pytest on an SM120 GPU. No model weights or distributed launch needed.
"""

import math

import pytest
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_dcp,
)
from sglang.kernels.ops.attention.flash_mla_sm120 import flashinfer_sparse_mla_forward

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires an SM120 GPU",
)


@pytest.mark.parametrize("dcp", [2, 4, 8])
@pytest.mark.parametrize("rows,heads", [(4, 32), (256, 64)])
def test_packed_shards_lse_empty_owner_and_graph(dcp, rows, heads):
    torch.manual_seed(41)
    capacity, topk = 4096, 2048
    device = "cuda"
    latent = (torch.randn(capacity, 512, device=device) * 0.25).to(torch.float8_e4m3fn)
    scales = torch.rand(capacity, 4, device=device) + 0.5
    rope = (torch.randn(capacity, 64, device=device) * 0.25).to(torch.bfloat16)
    packed = torch.empty(capacity, 656, dtype=torch.uint8, device=device)
    packed[:, :512] = latent.view(torch.uint8)
    packed[:, 512:528] = scales.view(torch.uint8)
    packed[:, 528:] = rope.view(torch.uint8)
    dequant = torch.cat(
        [
            (latent.float().view(capacity, 4, 128) * scales[..., None]).flatten(1),
            rope.float(),
        ],
        dim=-1,
    )
    query = (torch.randn(rows, heads, 576, device=device) * 0.25).to(torch.bfloat16)
    table = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    table[0, 0] = capacity - 1  # Last valid slot; every other owner is empty.
    table[1, :127] = torch.arange(127, device=device) * 8
    table[2] = torch.randperm(capacity, device=device)[:topk]
    # Rows 3 onward model graph padding, empty on every rank.
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)

    def attention(kv, indices, counts):
        return flashinfer_sparse_mla_forward(
            q=query,
            kv_cache=kv,
            indices=indices,
            seq_lens=counts,
            workspace_buffer=workspace,
            page_size=64,
            kv_cache_dim=656,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            sm_scale=192**-0.5,
            skip_softmax_threshold_scale_factor=None,
            return_lse=True,
        )

    resident = [packed[rank::dcp].contiguous() for rank in range(dcp)]

    def sharded():
        outputs, lses = [], []
        for rank in range(dcp):
            indices, counts = transform_index_page_table_dcp(table, dcp, rank)
            out, lse = attention(resident[rank], indices, counts)
            outputs.append(out)
            lses.append(lse)
        merged, merged_lse = dcp_lse_combine_triton(
            torch.stack(outputs),
            torch.stack(lses),
            is_lse_base_on_e=False,
            return_lse=True,
        )
        return merged, merged_lse, outputs, lses

    def check(result):
        merged, merged_lse, outputs, lses = result
        expected, expected_lse = attention(packed, table, (table >= 0).sum(-1).int())
        torch.testing.assert_close(merged[:3], expected[:3], atol=0.004, rtol=0.02)
        torch.testing.assert_close(
            merged_lse[:3], expected_lse[:3], atol=0.004, rtol=0.002
        )
        assert torch.all(merged[3:] == 0)
        assert torch.isneginf(merged_lse[3:]).all()
        for row in range(3):
            ids = table[row][table[row] >= 0].long()
            scores = query[row].float() @ dequant[ids].T * (192**-0.5)
            torch.testing.assert_close(
                expected_lse[row],
                torch.logsumexp(scores, -1) / math.log(2),
                atol=0.01,
                rtol=0.002,
            )
            reference = scores.softmax(-1) @ dequant[ids, :512]
            torch.testing.assert_close(
                expected[row].float(), reference, atol=0.004, rtol=0.02
            )
        for rank in range(dcp):
            empty = ~((table >= 0) & (table % dcp == rank)).any(-1)
            assert torch.all(outputs[rank][empty] == 0)
            assert torch.isneginf(lses[rank][empty]).all()

    check(sharded())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sharded()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = sharded()
    for shift in (1, 17):
        table.copy_(torch.where(table >= 0, (table + shift) % capacity, table))
        query.mul_(0.9)
        graph.replay()
        check(captured)
    torch.cuda.synchronize()
