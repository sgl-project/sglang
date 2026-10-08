# SPDX-License-Identifier: Apache-2.0
"""Optional sharded store capability must fall back on unsupported inputs/JIT."""

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention import fused_store_index_cache as store
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "key_dtype,indices_dtype,page_size",
    [
        (torch.float16, torch.int64, 64),
        (torch.float32, torch.int64, 64),
        (torch.bfloat16, torch.int32, 64),
        (torch.bfloat16, torch.int64, 32),
        (torch.bfloat16, torch.int64, 128),
    ],
)
def test_unsupported_format_does_not_compile(key_dtype, indices_dtype, page_size):
    with patch.object(store, "_jit_dsa_sharded_store_module") as compile_module:
        assert not store.can_use_dsa_sharded_store.__wrapped__(
            key_dtype, indices_dtype, page_size
        )
    compile_module.assert_not_called()


@pytest.mark.parametrize("cuda,hip", [(False, None), (True, "7.0")])
def test_non_nvidia_runtime_does_not_compile(cuda, hip):
    with (
        patch.object(torch.cuda, "is_available", return_value=cuda),
        patch.object(torch.version, "hip", hip),
        patch.object(store, "_jit_dsa_sharded_store_module") as compile_module,
    ):
        assert not store.can_use_dsa_sharded_store.__wrapped__(
            torch.bfloat16, torch.int64, 64
        )
    compile_module.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_jit_failure_falls_back(fails):
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.version, "hip", None),
        patch.object(
            store,
            "_jit_dsa_sharded_store_module",
            side_effect=RuntimeError("compiler unavailable") if fails else None,
        ) as compile_module,
    ):
        assert store.can_use_dsa_sharded_store.__wrapped__(
            torch.bfloat16, torch.int64, 64
        ) is (not fails)
    compile_module.assert_called_once_with(64)


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
