# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import random
from typing import Tuple

import numpy as np
import pytest
import torch

from sglang.srt.compilation.fusion.triton_ops.fused_swiglu import fused_swiglu_fwd


def seed_rng():
    SEED = 42
    np.random.seed(SEED)  # For Numpy
    torch.manual_seed(SEED)  # For CPU tensors
    torch.cuda.manual_seed_all(SEED)  # For CUDA tensors
    random.seed(SEED)  # For Python's own RNG


@pytest.fixture(autouse=True, scope="module")
def module_fixture():
    seed_rng()


def make_input_and_weights(
    batch_size: int, seq_len: int, d_model: int, d_intermediate: int, dtype: torch.dtype
) -> Tuple[torch.HalfTensor, torch.HalfTensor, torch.HalfTensor]:
    x = torch.randn(
        (batch_size * seq_len, d_model), device="cuda", dtype=dtype
    ).requires_grad_(False)

    w_gate = torch.randn((d_model, d_intermediate), device="cuda", dtype=dtype) / (
        math.sqrt(d_model)
    )
    w_up = torch.randn((d_model, d_intermediate), device="cuda", dtype=dtype) / (
        math.sqrt(d_model)
    )

    w_gate.requires_grad_(False)
    w_up.requires_grad_(False)

    w = torch.concat((w_gate, w_up), dim=1)

    return x, w


def swiglu_ref_torch(
    x: torch.HalfTensor, w_gate: torch.HalfTensor, w_up: torch.HalfTensor
) -> torch.HalfTensor:
    """Reference PyTorch implementation."""
    x_gate = torch.matmul(x, w_gate)
    x_up = torch.matmul(x, w_up)

    return torch.nn.functional.silu(x_gate) * x_up


@pytest.mark.parametrize("d_model", [1024, 2048, 4096])
@pytest.mark.parametrize("d_intermediate", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [1, 32, 128])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16], ids=["bfloat16", "float16"]
)
def test_fused_swiglu(d_model, d_intermediate, batch_size, seq_len, dtype):
    x, w = make_input_and_weights(batch_size, seq_len, d_model, d_intermediate, dtype)
    w_gate, w_up = torch.split(w, w.shape[1] // 2, dim=1)
    out_ref = swiglu_ref_torch(x, w_gate, w_up)
    out = fused_swiglu_fwd(x, w)
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__])
