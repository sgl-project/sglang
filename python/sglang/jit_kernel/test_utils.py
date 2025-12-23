from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@lru_cache(maxsize=None)
def _jit_test_module() -> Module:
    return load_jit(
        "test_utils",
        cpp_files=["test_utils.h"],
        cpp_wrappers=[("assert_same_shape", "assert_same_shape")],
    )


def main() -> None:
    module = _jit_test_module()
    a = torch.empty((3, 4), device="cuda:1", dtype=torch.float32)
    b = torch.empty((4, 4), device="cuda:1", dtype=torch.float32)
    # OK, same shape
    module.assert_same_shape(a, a)
    try:
        module.assert_same_shape(a, b)
    except Exception as e:
        print(f"Expected error: {e}")
    else:
        raise RuntimeError("Expected an error due to shape mismatch")
    print("All tests passed.")


if __name__ == "__main__":
    main()
