"""Q4 LIM-to-CopySFA metadata correctness and graph replay on Ascend."""

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.npu_lim_copy_sfa_metadata import (
    prepare_copy_sfa_metadata,
)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 14])
def test_fused_copy_sfa_metadata_width_and_graph(width):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device = torch.device("npu", torch.npu.current_device())
    batch, capacity, request_rows = 3, 12288, 4
    source = torch.arange(batch * width * 2048, dtype=torch.int32, device=device).view(
        -1, 2048
    )
    original = source.clone()
    valid = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    sparse = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    slots = torch.tensor([1, 2, request_rows], dtype=torch.int32, device=device)
    prefix = torch.tensor([16000, 2000, 0], dtype=torch.int32, device=device)
    kv = torch.empty(batch, dtype=torch.int32, device=device)
    hbm = torch.empty((batch, capacity // 128), dtype=torch.int32, device=device)

    def run():
        prepare_copy_sfa_metadata(
            source_ids=source,
            valid_rows=valid,
            sparse_rows=sparse,
            request_slots=slots,
            seq_lens=prefix,
            actual_seq_lengths_kv=kv,
            hbm_block_table=hbm,
            capacity=capacity,
            query_width=width,
        )

    def check(valid_cpu, sparse_cpu, slots_cpu):
        torch.npu.synchronize()
        assert kv.cpu().tolist() == [
            capacity + width - 1 if sp else (int(prefix[row].cpu()) if v else 0)
            for row, (v, sp) in enumerate(zip(valid_cpu, sparse_cpu))
        ]
        for row in range(batch):
            indices = slice(row * width, (row + 1) * width)
            if valid_cpu[row]:
                torch.testing.assert_close(
                    source[indices], original[indices], rtol=0, atol=0
                )
            else:
                assert torch.all(source[indices] == -1)
            hbm_start = slots_cpu[row] * (capacity // 128) if valid_cpu[row] else 0
            assert hbm[row].cpu().tolist() == list(
                range(hbm_start, hbm_start + capacity // 128)
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
    graph.replay()
    check([False, True, True], [False, True, True], [request_rows, 2, 3])
