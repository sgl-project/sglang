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


@pytest.mark.parametrize(
    ("role", "linker", "env"),
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
def test_decode_lends_nothing_unless_sized_like_prefill(monkeypatch, role, linker, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert (
        _decode_store_contribution_bytes(
            disaggregation_mode=role, external_linker=linker
        )
        == 0
    )


def test_decode_lends_the_configured_segment_size(monkeypatch):
    monkeypatch.setenv("MOONCAKE_MASTER", "m:50051")
    monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "140gb")
    assert (
        _decode_store_contribution_bytes(
            disaggregation_mode="decode", external_linker=False
        )
        == 140 * 1024**3
    )
