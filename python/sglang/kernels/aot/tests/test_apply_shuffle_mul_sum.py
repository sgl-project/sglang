import subprocess
import sys

import pytest


def test_apply_shuffle_mul_sum_empty():
    # Run separately: the old host-side 0/0 terminates the process.
    code = """
import torch
from sgl_kernel import apply_shuffle_mul_sum
x = torch.empty((0, 16), device="cuda", dtype=torch.float16)
y = torch.empty_like(x)
p = torch.empty((0,), device="cuda", dtype=torch.int32)
factors = torch.empty((0,), device="cuda", dtype=x.dtype)
apply_shuffle_mul_sum(x, y, p, factors)
torch.cuda.synchronize()
try:
    apply_shuffle_mul_sum(x, y, torch.zeros((1,), device="cuda", dtype=torch.int32), factors)
except RuntimeError:
    pass
else:
    raise AssertionError("nonempty permutation accepted for empty output")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\n{result.stdout}{result.stderr}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
