import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.block_tables import (
    fill_request_block_table,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.copy_sfa_buffers import (
    copy_sfa_host_row_stride,
    view_copy_sfa_output,
)


def test_fill_request_block_table_multiple_rows_and_padding():
    req_ids = torch.tensor([3, 1, 9], dtype=torch.int32)
    valid_req = torch.tensor([True, True, False])
    offsets = torch.arange(3, dtype=torch.int32)
    output = torch.empty((3, 3), dtype=torch.int32)

    fill_request_block_table(req_ids, valid_req, offsets, 5, output)

    assert torch.equal(
        output,
        torch.tensor([[15, 16, 17], [5, 6, 7], [0, 1, 2]], dtype=torch.int32),
    )


def test_fill_request_block_table_single_page_and_zero_request():
    req_ids = torch.tensor([0, 2], dtype=torch.int32)
    valid_req = torch.tensor([False, True])
    offsets = torch.tensor([0], dtype=torch.int32)
    output = torch.empty((2, 1), dtype=torch.int32)

    fill_request_block_table(req_ids, valid_req, offsets, 4, output)

    assert torch.equal(output, torch.tensor([[0], [8]], dtype=torch.int32))


def test_copy_sfa_output_uses_contiguous_prefix_for_unpadded_heads():
    output_buffer = torch.arange(4 * 8 * 512, dtype=torch.float32).view(4, 8, 512)

    output = view_copy_sfa_output(output_buffer, 2, 3, 512)

    assert output.shape == (2, 3, 512)
    assert output.is_contiguous()
    assert output.stride() == (3 * 512, 512, 1)
    assert output.data_ptr() == output_buffer.data_ptr()
    assert torch.equal(output.reshape(-1), output_buffer.reshape(-1)[: 2 * 3 * 512])


def test_copy_sfa_host_stride_aligns_without_changing_legacy_stride():
    assert copy_sfa_host_row_stride(32772, enabled=True) == 32896
    assert copy_sfa_host_row_stride(32768, enabled=True) == 32768
    assert copy_sfa_host_row_stride(32772, enabled=False) == 32772
