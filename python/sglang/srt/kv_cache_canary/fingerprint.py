"""Python reference implementation of the canary chain hash.

The hash is computed **on device** by the canary kernel (see
``jit_kernel/csrc/kv_cache_canary/canary.cuh``); this module only exists
to (a) document the algorithm in readable Python and (b) provide a
bit-wise reference for a Python <-> CUDA consistency unit test
(``test/registered/canary/test_splitmix64_consistency.py``). It is NOT
used on the hot path.

Algorithm:

- ``splitmix64`` finalizer: the standard variant
  (`x ^= x >> 30; x *= 0xBF58476D1CE4E5B9; x ^= x >> 27; x *= 0x94D049BB133111EB; x ^= x >> 31`).
- ``splitmix64_mix(prev, token_id, position)``: combine the three inputs
  via XOR (``x = prev ^ token_id ^ position``) and run the finalizer.

Chain head seed: comes from ``CanaryConfig.seed`` (default
``0xC0FFEE1234567890``) and is passed into the kernel at every launch.
"""

from __future__ import annotations

_U64_MASK = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """Standard splitmix64 finalizer (no `+= GOLDEN` step).

    Matches the device-side ``splitmix64_finalize`` in ``canary.cuh``.
    """
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def splitmix64_mix(prev_hash: int, token_id: int, position: int) -> int:
    """Chain step: combine prev/token/position via XOR then splitmix64.

    Bit-wise equivalent of the device-side ``splitmix64_mix`` in
    ``canary.cuh``. ``test_splitmix64_consistency.py`` cross-validates the
    two implementations.
    """
    assert (
        0 <= prev_hash <= _U64_MASK
    ), f"kv-canary: splitmix64_mix prev_hash {prev_hash:#x} out of uint64 range"
    combined = (prev_hash & _U64_MASK) ^ (token_id & _U64_MASK) ^ (position & _U64_MASK)
    return splitmix64(combined)


def to_signed_int64(value: int) -> int:
    """Reinterpret an unsigned uint64 as signed int64 (for torch.int64 storage).

    The canary stores ``prev_hash`` (a uint64) into an int64 field. Python's
    arbitrary-precision ints silently promote on overflow, so callers
    construct torch.int64 tensors via this helper to avoid Python's
    ``OverflowError: Python int too large to convert to C long``.
    """
    value &= _U64_MASK
    if value >= (1 << 63):
        value -= 1 << 64
    return value
