"""Q4 LIM-to-CopySFA metadata correctness and graph replay on Ascend."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
    fused_lim_copy_sfa_metadata,
)


@pytest.mark.parametrize("width", [1, 4])
def test_fused_copy_sfa_metadata_width_and_graph(width):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device = torch.device("npu", torch.npu.current_device())
    batch, capacity, context, request_rows = 3, 12288, 32768, 4
    source = torch.arange(batch * width * 2048, dtype=torch.int32, device=device).view(
        -1, 2048
    )
    original = source.clone()
    topk = torch.full((batch * width,), 7, dtype=torch.int32, device=device)
    counts = torch.full((batch,), 9, dtype=torch.int32, device=device)
    valid = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    sparse = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    slots = torch.tensor([1, 2, request_rows], dtype=torch.int32, device=device)
    prefix = torch.tensor([16000, 2000, 0], dtype=torch.int32, device=device)
    last = torch.full((request_rows + 1,), -1, dtype=torch.int32, device=device)
    kv = torch.empty(batch, dtype=torch.int32, device=device)
    cache = torch.empty_like(kv)
    hbm = torch.empty((batch, capacity // 128), dtype=torch.int32, device=device)
    dram = torch.empty((batch, context // 128), dtype=torch.int32, device=device)

    def run():
        fused_lim_copy_sfa_metadata(
            source_ids=source,
            topk_misses=topk,
            miss_counts=counts,
            valid_rows=valid,
            sparse_rows=sparse,
            request_slots=slots,
            next_prefix=prefix,
            last_prefix=last,
            actual_seq_lengths_kv=kv,
            num_cache_tokens=cache,
            hbm_block_table=hbm,
            dram_block_table=dram,
            capacity=capacity,
            host_context_len=context,
            request_rows=request_rows,
            query_width=width,
        )

    def check(valid_cpu, sparse_cpu, slots_cpu):
        torch.npu.synchronize()
        active = [v and s for v, s in zip(valid_cpu, sparse_cpu)]
        assert kv.cpu().tolist() == [capacity + width - 1 if a else 0 for a in active]
        assert cache.cpu().tolist() == [capacity if a else 0 for a in active]
        assert topk.cpu().tolist() == [
            7 if a else 0 for a in active for _ in range(width)
        ]
        assert counts.cpu().tolist() == [9 if a else 0 for a in active]
        for row in range(batch):
            indices = slice(row * width, (row + 1) * width)
            if valid_cpu[row]:
                torch.testing.assert_close(
                    source[indices], original[indices], rtol=0, atol=0
                )
            else:
                assert torch.all(source[indices] == -1)
            hbm_start = slots_cpu[row] * (capacity // 128) if active[row] else 0
            dram_start = slots_cpu[row] * (context // 128) if active[row] else 0
            assert hbm[row].cpu().tolist() == list(
                range(hbm_start, hbm_start + capacity // 128)
            )
            assert dram[row].cpu().tolist() == list(
                range(dram_start, dram_start + context // 128)
            )

    run()
    check([True, True, False], [True, False, False], [1, 2, request_rows])
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # Exercise active -> padding, short -> resident, and padding -> resident.
    valid.copy_(torch.tensor([False, True, True], device=device))
    sparse.copy_(torch.tensor([False, True, True], device=device))
    slots.copy_(torch.tensor([request_rows, 2, 3], dtype=torch.int32, device=device))
    prefix.copy_(torch.tensor([0, 18000, 19000], dtype=torch.int32, device=device))
    source.copy_(original)
    topk.fill_(7)
    counts.fill_(9)
    graph.replay()
    check([False, True, True], [False, True, True], [request_rows, 2, 3])
