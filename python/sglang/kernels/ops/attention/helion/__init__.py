"""Helion attention kernels."""

# K3 exposes 12 local value heads at TP=8. Lower value-head counts share the
# same small-head decode regime.
KDA_SMALL_VALUE_HEAD_THRESHOLD = 12


def kda_state_rows(state):
    """View a ``[..., V, K]`` state pool as ``[rows, K]`` for slot-indexed access.

    Triton 3.9 rejects broadcasting a scalar loaded from memory (a slot index)
    as a tensor index, which Helion emits for ``state[slot, h, tile_v, :]``.
    Kernels instead address the pool by row, ``sum(i_d * stride(d) // K) +
    v``, so the loaded slot only ever enters integer arithmetic. Rows keep the
    pool's real strides, including any padded slot pitch.
    """
    K = state.size(-1)
    if state.stride(-1) != 1 or state.stride(-2) != K:
        raise ValueError("KDA state pool rows must be contiguous [V, K] blocks")
    if any(state.stride(d) % K for d in range(state.ndim - 2)):
        raise ValueError("KDA state pool strides must be multiples of K")
    rows = 1 + sum(
        (state.size(d) - 1) * (state.stride(d) // K) for d in range(state.ndim - 1)
    )
    return state.as_strided((rows, K), (K, 1))
