# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    _get_b_tma_desc_cached,
    clear_b_tma_desc_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_descriptor_does_not_follow_a_rebound_parameter():
    storage = torch.empty(2, 64, 64, dtype=torch.bfloat16, device="cuda")
    first = torch.nn.Parameter(storage)
    second = torch.nn.Parameter(storage)
    clear_b_tma_desc_cache()
    try:
        old = _get_b_tma_desc_cached(first, 32, 32)
        assert _get_b_tma_desc_cached(first, 32, 32) is old
        # offload rebinds the first parameter; another layer can reuse its address
        first.data = torch.empty(0, dtype=storage.dtype, device=storage.device)
        current = _get_b_tma_desc_cached(second, 32, 32)
        assert current.base.data_ptr() == second.data_ptr()
        assert tuple(current.base.shape) == tuple(second.shape)
        assert _get_b_tma_desc_cached(second, 32, 32) is current
    finally:
        clear_b_tma_desc_cache()


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *args]))
