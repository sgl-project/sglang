"""CopySFA adapter ABI compatibility and native-LIM overlap contract."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def adapter():
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/kernels/ops/attention/npu_copy_sfa.py"
    )
    spec = importlib.util.spec_from_file_location("copy_sfa_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_scatter_copy_sparse_flash_attention


def inputs():
    query = torch.zeros(4, 4, 512, dtype=torch.bfloat16)
    rope = torch.zeros(4, 4, 64, dtype=query.dtype)
    metadata = [torch.empty(0, dtype=torch.int32) for _ in range(11)]
    metadata[3] = torch.zeros(4, 2048, dtype=torch.int32)
    metadata[4] = torch.zeros_like(metadata[3])
    return [query, rope, torch.empty(0), *metadata, 123, torch.empty_like(query), 0.1]


@pytest.mark.parametrize("full_visible", [False, True])
def test_operator_receives_the_native_resident_refresh_contract(adapter, full_visible):
    calls = []

    def native_operator(*args, full_visible=False, resident_refresh_tail=False):
        assert args[6].shape == args[7].shape == (4, 2048)
        calls.append((len(args), full_visible, resident_refresh_tail))
        args[-2].fill_(2)

    args = inputs()
    adapter(*args, operator=native_operator, full_visible=full_visible)
    assert calls == [(17, full_visible, False)]
    assert torch.all(args[-2] == 2)


def test_native_lim_suffix_enables_the_updated_operator(adapter):
    calls = []

    def native_operator(*args, full_visible=False, resident_refresh_tail=False):
        calls.append((full_visible, resident_refresh_tail))
        args[-2].fill_(3)

    args = inputs()
    adapter(
        *args, operator=native_operator, full_visible=True, resident_refresh_tail=True
    )
    assert calls == [(True, True)]
    assert torch.all(args[-2] == 3)


def test_native_suffix_contract_requires_full_visible(adapter):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for native validation")

    args = [
        value.npu() if isinstance(value, torch.Tensor) else value for value in inputs()
    ]
    with pytest.raises(
        RuntimeError, match="resident_refresh_tail requires full_visible"
    ):
        adapter(*args, resident_refresh_tail=True)
