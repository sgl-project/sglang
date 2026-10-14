import pytest

from sglang.srt.mem_cache.kv_cache_builder import _decode_store_contribution_bytes
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


_SIZED = {"MOONCAKE_MASTER": "m:50051", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb"}


def _bytes(role="decode", hierarchical=False, retraction="cpu_tensor", linker=False):
    return _decode_store_contribution_bytes(
        disaggregation_mode=role,
        enable_hierarchical_cache=hierarchical,
        retraction_backup=retraction,
        external_linker=linker,
    )


@pytest.mark.parametrize(
    ("kwargs", "env"),
    [
        # The segment default must not make every decode rank lend memory.
        ({}, {"MOONCAKE_MASTER": "m:50051"}),
        ({}, {**_SIZED, "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0"}),
        # A local store service owns the capacity in standalone mode.
        ({}, {**_SIZED, "MOONCAKE_STANDALONE_STORAGE": "1"}),
        # Shared role env without a master must not break decode startup.
        ({}, {"MOONCAKE_GLOBAL_SEGMENT_SIZE": "140gb"}),
        # Prefill mounts through its own cache.
        ({"role": "prefill"}, _SIZED),
        # Decode offload or a linker mounts its own segment; a second one would double-mount.
        ({"linker": True}, _SIZED),
        ({"hierarchical": True}, _SIZED),
        ({"retraction": "host_pool"}, _SIZED),
    ],
)
def test_decode_lends_nothing_unless_sized_like_prefill(monkeypatch, kwargs, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _bytes(**kwargs) == 0


def test_decode_lends_the_configured_segment_size(monkeypatch):
    for name, value in _SIZED.items():
        monkeypatch.setenv(name, value)
    assert _bytes() == 140 * 1024**3
