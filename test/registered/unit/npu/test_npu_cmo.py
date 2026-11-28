"""Weight prefetch dispatch preserves stream ordering and skips unsupported chips."""

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from sglang.srt.hardware_backend.npu import cmo
from sglang.srt.hardware_backend.npu import utils as npu_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def prefetch(monkeypatch):
    dispatch = Mock()
    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel_npu.operators",
        SimpleNamespace(data_move=SimpleNamespace(prefetch=dispatch)),
    )
    return dispatch


def test_unsupported_prefetch_does_not_create_stream(prefetch):
    with (
        patch.object(npu_utils, "is_npu_op_supported", return_value=False),
        patch.object(cmo, "torch") as torch,
    ):
        cmo.prepare_weight_cache(object(), object())
    torch.npu.Stream.assert_not_called()
    prefetch.assert_not_called()


def test_prefetch_list_preserves_stream_order_and_dependencies(prefetch):
    stream = Mock()
    handle, first, second = object(), object(), object()
    with (
        patch.object(npu_utils, "is_npu_op_supported", return_value=True),
        patch.object(cmo, "get_cmo_stream", return_value=stream),
        patch.object(cmo, "torch") as torch,
    ):
        torch.npu.stream.return_value = contextlib.nullcontext()
        cmo.prepare_weight_cache(handle, [first, second], 123)
        stream.wait_stream.assert_called_once_with(torch.npu.current_stream())
    assert prefetch.call_args_list == [
        call(first, handle, 123),
        call(second, handle, 123),
    ]


def test_missing_cache_does_not_create_stream_or_probe_generation(prefetch):
    with (
        patch.object(npu_utils, "is_npu_op_supported") as supported,
        patch.object(cmo, "torch") as torch,
    ):
        cmo.prepare_weight_cache(object(), None)
    torch.npu.Stream.assert_not_called()
    supported.assert_not_called()
    prefetch.assert_not_called()


def test_prefetch_single_weight_creates_stream(prefetch):
    stream = Mock()
    handle, weight = object(), object()
    with (
        patch.object(npu_utils, "is_npu_op_supported", return_value=True),
        patch.object(cmo, "get_cmo_stream", return_value=None),
        patch.object(cmo, "set_cmo_stream") as set_stream,
        patch.object(cmo, "torch") as torch,
    ):
        torch.npu.Stream.return_value = stream
        torch.npu.stream.return_value = contextlib.nullcontext()
        cmo.prepare_weight_cache(handle, weight, 123)
        set_stream.assert_called_once_with(stream)
        stream.wait_stream.assert_called_once_with(torch.npu.current_stream())
    prefetch.assert_called_once_with(weight, handle, 123)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
