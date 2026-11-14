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
    return module


def inputs():
    query = torch.zeros(4, 4, 512, dtype=torch.bfloat16)
    rope = torch.zeros(4, 4, 64, dtype=query.dtype)
    metadata = [torch.empty(0, dtype=torch.int32) for _ in range(11)]
    metadata[3] = torch.zeros(4, 2048, dtype=torch.int32)
    metadata[4] = torch.zeros_like(metadata[3])
    return [
        query,
        rope,
        torch.empty(0),
        torch.empty(0),
        *metadata,
        123,
        torch.empty_like(query),
        0.1,
    ]


@pytest.mark.parametrize(
    "full_visible,refresh", [(False, False), (True, False), (True, True)]
)
def test_operator_receives_split_and_resident_contract(adapter, full_visible, refresh):
    args = inputs()
    resident = {
        name: object()
        for name in (
            "current",
            "positions",
            "valid",
            "request_slots",
            "req_to_token",
            "resident_begin",
            "resident_end",
            "host_token_capacity",
            "dram_rope_dev_ptr",
        )
    }
    calls = []

    def native_operator(*native_args, full_visible, resident_refresh_tail, **kwargs):
        assert len(native_args) == 18
        assert all(actual is expected for actual, expected in zip(native_args, args))
        assert native_args[7].shape == native_args[8].shape == (4, 2048)
        assert kwargs == resident
        calls.append((full_visible, resident_refresh_tail))
        native_args[16].fill_(2)

    adapter.fused_scatter_copy_sparse_flash_attention(
        *args,
        operator=native_operator,
        full_visible=full_visible,
        resident_refresh_tail=refresh,
        **resident,
    )
    assert calls == [(full_visible, refresh)]
    assert torch.all(args[16] == 2)


def test_native_suffix_contract_requires_full_visible(adapter):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is required for native validation")

    args = [
        value.npu() if isinstance(value, torch.Tensor) else value for value in inputs()
    ]
    resident = {
        name: None
        for name in (
            "current",
            "positions",
            "valid",
            "request_slots",
            "req_to_token",
            "resident_begin",
            "resident_end",
            "host_token_capacity",
            "dram_rope_dev_ptr",
        )
    }
    resident["host_token_capacity"] = 0
    resident["dram_rope_dev_ptr"] = 0
    with pytest.raises(
        RuntimeError, match="resident_refresh_tail requires full_visible"
    ):
        adapter.fused_scatter_copy_sparse_flash_attention(
            *args,
            operator=adapter.load_copy_sfa_operator(),
            resident_refresh_tail=True,
            **resident,
        )
