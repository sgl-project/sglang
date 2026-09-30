import pytest

from sglang.srt.mem_cache.kv_cache_builder import (
    _decode_cache_owns_store,
    _decode_store_contribution_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MOONCAKE_ENV = (
    "MOONCAKE_GLOBAL_SEGMENT_SIZE",
    "MOONCAKE_STANDALONE_STORAGE",
    "MOONCAKE_MASTER",
)


@pytest.fixture(autouse=True)
def _clean_mooncake_env(monkeypatch):
    for name in _MOONCAKE_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("role", "owns_store", "env"),
    [
        # The segment default must not make every decode rank lend memory.
        ("decode", False, {"MOONCAKE_MASTER": "m:50051"}),
        (
            "decode",
            False,
            {"MOONCAKE_MASTER": "m:50051", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0"},
        ),
        # A local store service owns the capacity in standalone mode.
        (
            "decode",
            False,
            {
                "MOONCAKE_MASTER": "m:50051",
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb",
                "MOONCAKE_STANDALONE_STORAGE": "1",
            },
        ),
        # Shared role env without a master must not break decode startup.
        ("decode", False, {"MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb"}),
        # These ranks already mount through their own cache path.
        (
            "decode",
            True,
            {"MOONCAKE_MASTER": "m:50051", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb"},
        ),
        (
            "prefill",
            False,
            {"MOONCAKE_MASTER": "m:50051", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb"},
        ),
    ],
)
def test_decode_lends_nothing_unless_sized_like_prefill(
    monkeypatch, role, owns_store, env
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert (
        _decode_store_contribution_bytes(
            disaggregation_mode=role, cache_owns_store=owns_store
        )
        == 0
    )


def test_decode_lends_the_configured_segment_size(monkeypatch):
    monkeypatch.setenv("MOONCAKE_MASTER", "m:50051")
    monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "140gb")
    assert (
        _decode_store_contribution_bytes(
            disaggregation_mode="decode", cache_owns_store=False
        )
        == 140 * 1024**3
    )


@pytest.mark.parametrize(
    ("disable_radix", "hierarchical", "retraction", "storage", "linker", "owns"),
    [
        # A chunk cache drops store flags copied from prefill; decode must still lend.
        (True, False, "cpu_tensor", "mooncake", True, False),
        # Host-pool retraction builds HiCache, which owns a store only with a backend.
        (True, False, "host_pool", None, True, False),
        (True, False, "host_pool", "mooncake", False, True),
        # A decode radix cache attaches the linker unless HiCache takes precedence.
        (False, False, "cpu_tensor", None, True, True),
        (False, True, "cpu_tensor", None, True, False),
    ],
)
def test_decode_cache_owns_store_matches_tree_cache_selection(
    disable_radix, hierarchical, retraction, storage, linker, owns
):
    assert (
        _decode_cache_owns_store(
            disable_radix_cache=disable_radix,
            enable_hierarchical_cache=hierarchical,
            retraction_backup=retraction,
            hicache_storage_backend=storage,
            external_linker=linker,
        )
        is owns
    )
