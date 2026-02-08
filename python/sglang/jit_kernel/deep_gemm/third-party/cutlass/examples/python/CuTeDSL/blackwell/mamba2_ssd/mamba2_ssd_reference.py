# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F


def ssd_reference_fp32_all(x, a, delta, B, C, Y_out, Fstate_out, D, has_d, d_has_hdim):
    """
    Rearrange tensor dimensions from cuda layout to reference layout, then directly call TriDao's ssd implementation
    Arguments:
        X/x: (D, L, C, H, B):(C*L, 1, L, D*C*L, H*D*C*L)
        A/delta: (L, C, H, B):(1, L, C*L, H*C*L)
        a: (H):(1)
        B/C: (L, N, C, G, B):(1, C*L, L, N*C*L, G*N*C*L)
        D: (1, H):(0, 1) or (D, H):(1, D)
        has_d: bool
        d_has_hdim: bool
    Return:
        Y_out: (L, D, C, H, B):(1, C*L, L, D*C*L, H*D*C*L)
        Fstate_out: (D, N, H, B):(N, 1, D*N, H*D*N)
    """
    assert x.dtype == a.dtype == delta.dtype == B.dtype == C.dtype

    A = delta * a.view(1, 1, -1, 1)
    X = x * delta.unsqueeze(0)

    # Rearrange to match cutlass layout to tridao's layout
    block_len = A.shape[0]
    initial_states = None
    # A: l c h b-> b c l h
    A = A.permute(3, 1, 0, 2)
    # X: p l c h b -> b c l h p
    X = X.permute(4, 2, 1, 3, 0)
    # B: l n c g b -> b c l g n
    B = B.permute(4, 2, 0, 3, 1)
    # C: l n c g b -> b c l g n
    C = C.permute(4, 2, 0, 3, 1)
    # X/A/B/C: b c l ... -> b (c l) ...
    X, A, B, C = [x.reshape(x.shape[0], -1, *x.shape[3:]) for x in (X, A, B, C)]

    # Ngroup (g to h) mapping
    B_val, CL_val, G_val, N_val = B.shape
    H_val = X.shape[2]
    ngroup_ratio = H_val // G_val
    # B/C: (B, CL, H, N)
    h_to_g_mapping = torch.arange(H_val, device=B.device) // ngroup_ratio
    B = B.gather(2, h_to_g_mapping.view(1, 1, -1, 1).expand(B_val, CL_val, -1, N_val))
    C = C.gather(2, h_to_g_mapping.view(1, 1, -1, 1).expand(B_val, CL_val, -1, N_val))

    ###################################################################
    # Call reference implementation from Tri Dao ssd_minimal_discrete
    Y, final_state = ssd_minimal_discrete_fp32_all(
        X, A, B, C, block_len, initial_states
    )
    ###################################################################

    if has_d:
        D_val = Y.shape[3]
        if not d_has_hdim:
            D = D.expand(D_val, -1)
        Y = Y + torch.einsum("bchp,ph->bchp", X, D)

    # Rearrange to match tridao's layout to cutlass layout
    # Y: b (c l) h p -> b c l h p
    Y = Y.reshape(Y.shape[0], -1, block_len, Y.shape[2], Y.shape[3])
    # Y: b c l h p -> l p c h b
    Y = Y.permute(2, 4, 1, 3, 0)
    # Fstate_out: b h p n -> p n h b
    Fstate_out.copy_(final_state.permute(2, 3, 1, 0))
    Y_out.copy_(Y)
    return


def ssd_reference_lowprecision_intermediates(
    x, a, delta, B, C, Y_out, Fstate_out, intermediate_dtype, D, has_d, d_has_hdim
):
    """
    Rearrange tensor dimensions from cuda layout to reference layout, then call a reduced intermediate dtype version of ssd implementation
    Arguments:
        X/x: (D, L, C, H, B):(C*L, 1, L, D*C*L, H*D*C*L)
        A/delta: (L, C, H, B):(1, L, C*L, H*C*L)
        a: (H):(1)
        B/C: (L, N, C, G, B):(1, C*L, L, N*C*L, G*N*C*L)
        intermediate_dtype: input and intermediate data type
        D: (1, H):(0, 1) or (D, H):(1, D)
        has_d: bool
        d_has_hdim: bool
    Return:
        Y_out: (L, D, C, H, B):(1, C*L, L, D*C*L, H*D*C*L)
        Fstate_out: (D, N, H, B):(N, 1, D*N, H*D*N)
    """
    assert x.dtype == a.dtype == delta.dtype == B.dtype == C.dtype

    A = delta * a.view(1, 1, -1, 1)

    # Rearrange to match cutlass layout to tridao's layout
    block_len = A.shape[0]
    initial_states = None
    # A: l c h b-> b c l h
    A = A.permute(3, 1, 0, 2)
    # delta: l c h b-> b c l h
    delta = delta.permute(3, 1, 0, 2)
    # x: p l c h b -> b c l h p
    x = x.permute(4, 2, 1, 3, 0)
    # B: l n c g b -> b c l g n
    B = B.permute(4, 2, 0, 3, 1)
    # C: l n c g b -> b c l g n
    C = C.permute(4, 2, 0, 3, 1)
    # x/A/delta/B/C: b c l ... -> b (c l) ...
    x, A, delta, B, C = [
        tensor.reshape(tensor.shape[0], -1, *tensor.shape[3:])
        for tensor in (x, A, delta, B, C)
    ]

    # Ngroup (g to h) mapping
    B_val, CL_val, G_val, N_val = B.shape
    H_val = x.shape[2]
    ngroup_ratio = H_val // G_val
    # B/C: (B, CL, H, N)
    h_to_g_mapping = torch.arange(H_val, device=B.device) // ngroup_ratio
    B = B.gather(2, h_to_g_mapping.view(1, 1, -1, 1).expand(B_val, CL_val, -1, N_val))
    C = C.gather(2, h_to_g_mapping.view(1, 1, -1, 1).expand(B_val, CL_val, -1, N_val))

    # Type convert input tensors to input dtype (same as intermediate dtype)
    x = x.to(intermediate_dtype).to(torch.float32)
    A = A.to(intermediate_dtype).to(torch.float32)
    delta = delta.to(intermediate_dtype).to(torch.float32)
    B = B.to(intermediate_dtype).to(torch.float32)
    C = C.to(intermediate_dtype).to(torch.float32)

    #########################################################################
    # Call reference implementation ssd_minimal_discrete_bf16_intermediates
    Y, final_state = ssd_minimal_discrete_lowprecision_intermediates(
        x, A, delta, B, C, block_len, intermediate_dtype, initial_states
    )
    #########################################################################

    if has_d:
        D = D.to(intermediate_dtype).to(torch.float32)
        D_val = Y.shape[3]
        if not d_has_hdim:
            D = D.expand(D_val, -1)
        Y = Y + torch.einsum("bchp,ph->bchp", x, D)

    # Type convert output tensors to output dtype (same as intermediate dtype)
    Y = Y.to(intermediate_dtype).to(torch.float32)
    final_state = final_state.to(intermediate_dtype).to(torch.float32)

    # Rearrange to match tridao's layout to cutlass layout
    # Y: b (c l) h p -> b c l h p
    Y = Y.reshape(Y.shape[0], -1, block_len, Y.shape[2], Y.shape[3])
    # Y: b c l h p -> l p c h b
    Y = Y.permute(2, 4, 1, 3, 0)
    # Fstate_out: b h p n -> p n h b
    Fstate_out.copy_(final_state.permute(2, 3, 1, 0))
    Y_out.copy_(Y)
    return


def analyze_relative_diffs(actual, expected):
    """
    Print statistics of relative differences between actual and expected tensors
    """
    # Calculate relative differences
    abs_diff = (actual - expected).abs()
    rel_diff = abs_diff / (torch.maximum(expected.abs(), actual.abs()) + 0.00001)

    total_elements = rel_diff.numel()

    # Handle special cases first
    nan_mask = torch.isnan(rel_diff)
    inf_mask = torch.isinf(rel_diff)
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()

    # Find position and value of maximum relative difference
    max_rel_diff = (
        rel_diff[~nan_mask & ~inf_mask].max()
        if (~nan_mask & ~inf_mask).any()
        else float("nan")
    )
    max_rel_diff_pos = (
        rel_diff[~nan_mask & ~inf_mask].argmax()
        if (~nan_mask & ~inf_mask).any()
        else -1
    )

    # Print max relative difference info
    print(f"Maximum relative difference:")
    print(f"Position: {max_rel_diff_pos}")
    print(f"Value: {max_rel_diff:.6e}")
    print(f"Actual value: {actual.flatten()[max_rel_diff_pos]}")
    print(f"Expected value: {expected.flatten()[max_rel_diff_pos]}")
    print(f"NaN values: {nan_count} ({100.0 * nan_count / total_elements:.2f}%)")
    print(f"Inf values: {inf_count} ({100.0 * inf_count / total_elements:.2f}%)\n")

    # Check different rtol thresholds
    rtol_levels = [1e-5, 1e-4, 1e-3, 1e-2, 5e-02, 1e-01]

    for i, rtol in enumerate(rtol_levels):
        if i == 0:
            mask = rel_diff <= rtol
        else:
            mask = (rel_diff <= rtol) & (rel_diff > rtol_levels[i - 1])

        count = mask.sum().item()
        percentage = (count / total_elements) * 100

        if i == 0:
            print(f"Elements with rtol <= {rtol:.0e}: {count} ({percentage:.2f}%)")
        else:
            print(
                f"Elements with {rtol_levels[i-1]:.0e} < rtol <= {rtol:.0e}: {count} ({percentage:.2f}%)"
            )

    # Print elements exceeding the largest rtol
    mask = rel_diff > rtol_levels[-1]
    count = mask.sum().item()
    percentage = (count / total_elements) * 100
    print(f"Elements with rtol > {rtol_levels[-1]:.0e}: {count} ({percentage:.2f}%)\n")


def segsum(x):
    """
    More stable segment sum calculation.
    x: b h c l
    """
    T = x.size(-1)
    # x: b h c l -> b h c l l
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete_fp32_all(X, A, B, C, block_len, initial_states=None):
    """
    This is same with https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
    (all accumulation and intermediate results in fp32)

    Arguments:
        X: (batch(B), length(C*L), n_heads(H), d_head(D))
        A: (batch(B), length(C*L), n_heads(H))
        B: (batch(B), length(C*L), n_heads(H), d_state(N))
        C: (batch(B), length(C*L), n_heads(H), d_state(N))
    Return:
        Y: (batch(B), length(C*L), n_heads(H), d_head(D))
        final_state: (B, H, D, N)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    # X/A/B/C:b (c l) ... -> b c l ...
    X, A, B, C = [
        x.reshape(x.shape[0], -1, block_len, *x.shape[2:]) for x in (X, A, B, C)
    ]

    # A: b c l h -> b h c l
    A = A.permute(0, 3, 1, 2)
    # A_cumsum: (B, H, C, L)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    segsum_A = segsum(A)
    L = torch.exp(segsum_A)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y: b c l h p -> b (c l) h p
    Y = (Y_diag + Y_off).reshape(Y_diag.shape[0], -1, Y_diag.shape[3], Y_diag.shape[4])
    return Y, final_state


def ssd_minimal_discrete_lowprecision_intermediates(
    X, A, delta, B, C, block_len, intermediate_dtype, initial_states=None
):
    """
    This is adjusted from ssd_minimal_discrete_fp32_all, with exceptions:
    1. accumulation in fp32 but intermediates Q/b_tmem/P are in intermediate_dtype
    2. delta is not pre-multiplied with X, delta was applied to generate Q/b_tmem to match GPU implementation

    Arguments:
        X: (batch(B), length(C*L), n_heads(H), d_head(D))
        A: (batch(B), length(C*L), n_heads(H))
        delta: (batch(B), length(C*L), n_heads(H))
        B: (batch(B), length(C*L), n_heads(H), d_state(N))
        C: (batch(B), length(C*L), n_heads(H), d_state(N))
    Return:
        Y: (batch(B), length(C*L), n_heads(H), d_head(D))
        final_state: (B, H, D, N)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    # X/A/delta/B/C: b (c l) ... -> b c l ...
    X, A, delta, B, C = [
        x.reshape(x.shape[0], -1, block_len, *x.shape[2:]) for x in (X, A, delta, B, C)
    ]

    # A: b c l h -> b h c l
    A = A.permute(0, 3, 1, 2)
    # delta: b c l h -> b h c l
    delta = delta.permute(0, 3, 1, 2)
    # A_cumsum: (B, H, C, L)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    segsum_A = segsum(A)
    L = torch.exp(segsum_A)
    intra_acc_0 = torch.einsum("bclhn,bcshn->bclhs", C, B)
    Q = torch.einsum("bclhs,bhcls,bhcs->bclhs", intra_acc_0, L, delta)
    Y_diag = torch.einsum(
        "bclhs,bcshp->bclhp", Q.to(intermediate_dtype).to(torch.float32), X
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    b_tmem = torch.einsum("bclhn,bhcl,bhcl->bclhn", B, decay_states, delta)
    states = torch.einsum(
        "bclhn,bclhp->bchpn", b_tmem.to(intermediate_dtype).to(torch.float32), X
    )

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    final_state = final_state

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off_tmp = torch.einsum(
        "bclhn,bchpn->bclhp", C, states.to(intermediate_dtype).to(torch.float32)
    )
    Y_off = torch.einsum("bclhp,bhcl->bclhp", Y_off_tmp, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y: b c l h p -> b (c l) h p
    Y = (Y_diag + Y_off).reshape(
        Y_diag.shape[0], -1, Y_diag.shape[3], Y_diag.shape[4]
    )  # b (c l) h p
    return Y, final_state
