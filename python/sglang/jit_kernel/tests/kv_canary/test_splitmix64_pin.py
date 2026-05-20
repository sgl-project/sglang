"""Pin splitmix64 magic constants against hand-frozen hex literals."""
from __future__ import annotations

import pytest

from sglang.jit_kernel.kv_canary.verify import CANARY_CHAIN_ANCHOR
from sglang.jit_kernel.tests.kv_canary.canary_helpers import splitmix64
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-unit-1-gpu-large")

_U64_MASK = (1 << 64) - 1


def _splitmix64_independent(value: int) -> int:
    x = value & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


_FROZEN_HEX = {
    0x9E3779B97F4A7C15: 0xE220A8397B1DCDAF,        # published Vigna 2014 test vector — anchors the dict to a known reference
    0x0000000000000001: 0x5692161D100B05E5,
    0xDEADBEEFCAFEBABE: 0x7AD6664F09FFE52C,
    0xFFFFFFFFFFFFFFFF: 0xB4D055FCF2CBBD7B,
    0x8000000000000000: 0x25C26EA579CEA98A,                # sign-bit boundary
    CANARY_CHAIN_ANCHOR: 0xDE7FAE23A9A1B716,
}


def test_splitmix64_helper_matches_frozen_hex() -> None:
    for input_val, expected in _FROZEN_HEX.items():
        actual = splitmix64(input_val)
        assert actual == expected, f"splitmix64({input_val:#x}) expected {expected:#x} got {actual:#x}"


def test_splitmix64_independent_impl_matches_frozen_hex() -> None:
    for input_val, expected in _FROZEN_HEX.items():
        actual = _splitmix64_independent(input_val)
        assert actual == expected, f"_splitmix64_independent({input_val:#x}) expected {expected:#x} got {actual:#x}"


def test_helper_and_independent_match() -> None:
    for input_val in _FROZEN_HEX:
        assert splitmix64(input_val) == _splitmix64_independent(input_val)
