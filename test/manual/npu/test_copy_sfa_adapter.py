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
    return [query, rope, torch.empty(0), *metadata, 123, torch.empty_like(query), 0.1]


@pytest.mark.parametrize("full_visible", [False, True])
def test_existing_operator_does_not_receive_a_new_keyword(adapter, full_visible):
    calls = []

    def legacy_operator(*args, full_visible=False):
        calls.append((len(args), full_visible))
        args[-2].fill_(2)

    args = inputs()
    adapter(*args, operator=legacy_operator, full_visible=full_visible)
    assert calls == [(17, full_visible)]
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


def test_suffix_contract_requires_full_visible(adapter):
    with pytest.raises(ValueError, match="requires full_visible=True"):
        adapter(*inputs(), operator=lambda *args: None, resident_refresh_tail=True)
