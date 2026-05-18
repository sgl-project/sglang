from __future__ import annotations

_U64_MASK = (1 << 64) - 1


def splitmix64(value: int) -> int:
    x = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return x ^ (x >> 31)


def mix_step(prev_hash: int, token_id: int, position: int) -> int:
    h = splitmix64(prev_hash ^ ((token_id & _U64_MASK) * 0xBF58476D1CE4E5B9 & _U64_MASK))
    h = splitmix64(h ^ ((position & _U64_MASK) * 0x94D049BB133111EB & _U64_MASK))
    return h & _U64_MASK
