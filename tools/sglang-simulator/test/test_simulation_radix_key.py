"""Differential CPU key comparison against the actual native owner, without GPU imports."""

import ast
import random
from array import array
from pathlib import Path

import pytest
from sglang_simulator.simulation.sglang.radix_key import C_RadixKeyHook


@pytest.fixture
def native_key():
    root = Path(__file__).resolve().parents[3]
    path = root / "python/sglang/srt/mem_cache/radix_cache.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RadixKey"
    )
    scope = {"array": array}
    exec("from __future__ import annotations\n" + ast.unparse(definition), scope)  # noqa: S102 -- trusted checked-out class, no GPU imports
    target = scope["RadixKey"]
    original = target.match_at
    C_RadixKeyHook.hook(target)
    return target, original


@pytest.mark.parametrize("length", [0, 1, 255, 4096, 65536, 262144])
@pytest.mark.parametrize("bigram", [False, True])
def test_same_prefix_mismatch_offset_limit_and_page_boundaries(
    native_key, length, bigram
):
    key, original = native_key
    rng = random.Random(length)
    for offset in (0, 7):
        for divergence in (None, 0, 63, length // 2, max(0, length - 1)):
            raw = array("q", range(length + offset + 1))
            left = key(raw[offset:], is_bigram=bigram)
            right = key(raw[:], is_bigram=bigram)
            if divergence is not None and divergence < length:
                right.token_ids[offset + divergence] = -123
            for limit in (None, max(0, length - rng.randrange(8))):
                left.limit = limit
                for page in (1, 2, 256):
                    assert left.match_at(right, offset, page) == original(
                        left, right, offset, page
                    )


@pytest.mark.parametrize("typecode", ["q", "i"])
def test_invalid_namespaces_modes_and_offsets_keep_native_errors(native_key, typecode):
    key, original = native_key
    left = key(array(typecode, range(8192)))
    for kwargs, offset in (
        ({"extra_key": "other"}, 0),
        ({"cache_salt": "other"}, 0),
        ({"is_bigram": True}, 0),
        ({}, -1),
        ({}, 9000),
    ):
        right = key(array(typecode, range(8192)), **kwargs)
        with pytest.raises((ValueError, IndexError)) as expected:
            original(left, right, offset)
        with pytest.raises(
            type(expected.value),
            match=str(expected.value).replace("[", r"\[").replace("]", r"\]"),
        ):
            left.match_at(right, offset)


def test_comparison_keeps_mutable_storage_and_existing_key_ownership(native_key):
    key, original = native_key
    first, second = array("q", range(16384)), array("q", range(16384))
    left, right = key(first), key(second)
    assert left.match_at(right, 0) == 16384
    second[8192] = -1
    assert left.match_at(right, 0) == original(left, right, 0) == 8192
    assert left.token_ids is first and right.token_ids is second
    # Successful comparison must not leave an exported buffer preventing resize.
    first.append(17)
    second.append(19)
