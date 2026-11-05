import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.copy_sfa_buffers import (
    copy_sfa_host_row_stride,
    view_copy_sfa_output,
)


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
