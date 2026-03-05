# Copyright 2025 SGLang Team
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
"""Mamba1 SSM operations.

Prefill uses the sgl-kernel selective_scan_fwd CUDA kernel.
Decode uses the shared selective_state_update kernel (same as Mamba2),
which handles Mamba1's 2D shapes by normalizing to 4D internally.
"""

from typing import Optional

import torch

from sgl_kernel import selective_scan_fwd


def mamba1_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    initial_state: Optional[torch.Tensor] = None,
):
    """Mamba1 selective scan for prefill.

    Uses the sgl-kernel CUDA kernel (selective_scan_fwd).

    Args:
        u: (batch, seqlen, dim) - input
        delta: (batch, seqlen, dim) - time step
        A: (dim, dstate) - A matrix
        B: (batch, seqlen, dstate) - B projection
        C: (batch, seqlen, dstate) - C projection
        D: (dim,) - optional D parameter
        z: (batch, seqlen, dim) - optional gate
        delta_bias: (dim,) - optional delta bias
        delta_softplus: whether to apply softplus to delta
        return_last_state: whether to return the final state
        initial_state: (batch, dim, dstate) - optional initial state for prefix caching

    Returns:
        out: (batch, seqlen, dim) - output
        last_state: (batch, dim, dstate) - final state (if return_last_state)
    """
    # selective_scan_fwd expects:
    # u: (B, D, L)
    # delta: (B, D, L)
    # A: (D, N)
    # B: (B, G, N, L)
    # C: (B, G, N, L)
    # D: (D,)
    # z: (B, D, L) optional
    # delta_bias: (D,) optional
    # initial_state: (B, D, N) optional

    # Transpose inputs from (B, L, D) to kernel format (B, D, L)
    u_t = u.transpose(1, 2).contiguous()  # (B, D, L)
    delta_t = delta.transpose(1, 2).contiguous()  # (B, D, L)
    # B,C: (B, L, N) -> (B, N, L) -> (B, 1, N, L) for variable B/C format
    B_t = B.transpose(1, 2).unsqueeze(1).contiguous()  # (B, 1, N, L)
    C_t = C.transpose(1, 2).unsqueeze(1).contiguous()  # (B, 1, N, L)
    z_t = z.transpose(1, 2).contiguous() if z is not None else None

    results = selective_scan_fwd(
        u_t, delta_t, A, B_t, C_t, D, z_t, delta_bias,
        initial_state, delta_softplus, True,
    )

    # Transpose output back: (B, D, L) -> (B, L, D)
    out = results[0].transpose(1, 2)

    if return_last_state:
        return out, results[1]  # last_state: (B, D, N)
    return out
