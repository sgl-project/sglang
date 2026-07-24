import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    _make_loaded_weight_compact,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _assert_independent_compact_copy(
    source: torch.Tensor, result: torch.Tensor
) -> None:
    assert torch.equal(result, source)
    assert result is not source
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def test_keeps_already_compact_tensor():
    source = torch.arange(16).reshape(4, 4)
    assert _make_loaded_weight_compact(source) is source


def test_materializes_zero_offset_storage_view():
    backing = torch.arange(32).reshape(8, 4)
    source = backing.narrow(0, 0, 2)
    assert source.storage_offset() == 0
    assert source.untyped_storage().nbytes() > source.numel() * source.element_size()
    _assert_independent_compact_copy(source, _make_loaded_weight_compact(source))


def test_materializes_nonzero_offset_storage_view():
    backing = torch.arange(32).reshape(8, 4)
    source = backing.narrow(0, 6, 2)
    assert source.storage_offset() != 0
    _assert_independent_compact_copy(source, _make_loaded_weight_compact(source))


def test_materializes_noncontiguous_tensor():
    source = torch.arange(16).reshape(4, 4).transpose(0, 1)
    assert not source.is_contiguous()
    _assert_independent_compact_copy(source, _make_loaded_weight_compact(source))
