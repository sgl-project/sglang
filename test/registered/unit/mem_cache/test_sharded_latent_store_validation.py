# SPDX-License-Identifier: Apache-2.0
"""Metadata gates run before the optional raw-FP8 dual-store launch."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.kvcache import set_mla_kv_buffer as store
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _inputs(n=4):
    return [
        torch.empty(128, 576, dtype=torch.uint8),
        torch.arange(n),
        torch.empty(64, 576, dtype=torch.uint8),
        torch.arange(n),
        torch.empty(n, 512, dtype=torch.uint8),
        torch.empty(n, 64, dtype=torch.uint8),
        64,
        4,
        0,
    ]


@pytest.mark.parametrize("widths", [(528, 128), (1024, 128), (512, 0), (0, 64)])
def test_other_formats_never_compile(widths):
    with patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module:
        assert not store.can_use_set_sharded_mla_kv_buffer.__wrapped__(*widths)
    compile_module.assert_not_called()


@pytest.mark.parametrize(
    "cuda,hip,available,major",
    [(None, None, False, 0), ("13.0", "7.0", True, 9), ("13.0", None, True, 8)],
)
def test_unsupported_device_never_compiles(cuda, hip, available, major):
    with (
        patch.object(torch.version, "cuda", cuda),
        patch.object(torch.version, "hip", hip),
        patch.object(torch.cuda, "is_available", return_value=available),
        patch.object(torch.cuda, "get_device_capability", return_value=(major, 0)),
        patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module,
    ):
        assert not store.can_use_set_sharded_mla_kv_buffer.__wrapped__(512, 64)
    compile_module.assert_not_called()


@pytest.mark.parametrize("compile_error", [False, True])
def test_compile_capability_gate(compile_error):
    with (
        patch.object(torch.version, "cuda", "13.0"),
        patch.object(torch.version, "hip", None),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.object(store, "is_arch_support_pdl", return_value=True),
        patch.object(
            store,
            "set_sharded_mla_kv_buffer_module",
            side_effect=RuntimeError("compiler unavailable") if compile_error else None,
        ) as compile_module,
    ):
        assert store.can_use_set_sharded_mla_kv_buffer.__wrapped__(512, 64) is (
            not compile_error
        )
    compile_module.assert_called_once_with(512, 64, True)


@pytest.mark.parametrize(
    "index,replacement,reason",
    [
        (1, torch.zeros(4, 1, dtype=torch.int64), "one-dimensional"),
        (1, torch.zeros(4, dtype=torch.float32), "int32 or int64"),
        (1, torch.arange(8)[::2], "contiguous"),
        (1, torch.arange(3), "matching lengths"),
        (1, torch.arange(4, dtype=torch.int32), "matching lengths and dtypes"),
        (4, torch.empty(3, 512, dtype=torch.uint8), "matching loc length"),
        (4, torch.empty(4, 512), "preconverted uint8"),
        (4, torch.empty(4, 2, 256, dtype=torch.uint8), "sources must have shape"),
        (4, torch.empty(4, 513, dtype=torch.uint8)[:, :512], "16-byte aligned"),
        (4, torch.empty(1, 512, dtype=torch.uint8).expand(4, 512), "non-overlapping"),
        (5, torch.empty(4, 128, dtype=torch.uint8), "preconverted uint8"),
        (
            0,
            torch.empty(128, 2, 576, dtype=torch.uint8),
            "destinations must have shape",
        ),
        (0, torch.empty(128, 575, dtype=torch.uint8), "at least 576"),
        (0, torch.empty(0, 576, dtype=torch.uint8), "nonempty destinations"),
        (0, torch.empty(128, 576), "destinations must be uint8"),
        (2, torch.empty(64, 577, dtype=torch.uint8)[:, :576], "16-byte aligned"),
        (2, torch.empty(1, 576, dtype=torch.uint8).expand(64, 576), "non-overlapping"),
        (6, 0, "invalid page sharding"),
        (7, 1, "invalid page sharding"),
        (8, 4, "invalid page sharding"),
        (8, -1, "invalid page sharding"),
        (6, 64.0, "must be integers"),
    ],
)
def test_invalid_metadata_rejected_before_launch(index, replacement, reason):
    args = _inputs()
    args[index] = replacement
    with patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module:
        assert not store.sharded_mla_kv_buffer_inputs_supported(*args)
        with pytest.raises(ValueError, match=reason):
            store.set_sharded_mla_kv_buffer(*args)
    compile_module.assert_not_called()


def test_cpu_tensors_rejected_before_launch():
    with patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module:
        assert not store.sharded_mla_kv_buffer_inputs_supported(*_inputs())
        with pytest.raises(ValueError, match="same CUDA device"):
            store.set_sharded_mla_kv_buffer(*_inputs())
    compile_module.assert_not_called()


@pytest.mark.parametrize(
    "reserved", ["scratch_reserved_skip_index", "local_reserved_skip_index"]
)
def test_invalid_reserved_indices_rejected(reserved):
    with pytest.raises(ValueError, match="reserved indices"):
        store.set_sharded_mla_kv_buffer(*_inputs(), **{reserved: -2})


def test_launch_failure_is_not_caught_or_retried():
    # Bypass device metadata only to exercise the dispatch contract on CPU.
    with (
        patch.object(store, "_sharded_mla_kv_buffer_input_error", return_value=None),
        patch.object(store, "is_arch_support_pdl", return_value=True),
        patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module,
    ):
        compile_module.return_value.store.side_effect = RuntimeError("launch failed")
        with pytest.raises(RuntimeError, match="launch failed"):
            store.set_sharded_mla_kv_buffer(*_inputs())
    assert compile_module.return_value.store.call_count == 1


def test_dispatch_keeps_independent_reserved_indices():
    args = _inputs()
    with (
        patch.object(store, "_sharded_mla_kv_buffer_input_error", return_value=None),
        patch.object(store, "is_arch_support_pdl", return_value=True),
        patch.object(store, "set_sharded_mla_kv_buffer_module") as compile_module,
    ):
        store.set_sharded_mla_kv_buffer(
            *args, scratch_reserved_skip_index=-1, local_reserved_skip_index=19
        )
    sent = compile_module.return_value.store.call_args.args
    assert sent[-2:] == (-1, 19)
    assert sent[2].data_ptr() == args[4].data_ptr()
    assert sent[3].data_ptr() == args[5].data_ptr()


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *args]))
