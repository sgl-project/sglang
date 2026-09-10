from pathlib import Path

from asc_bench.npu import parse_hbm_used_mb

FIXTURE = Path(__file__).parent / "fixtures" / "npu_smi_sample.txt"


def test_fixture_parses_both_dies():
    used = parse_hbm_used_mb(FIXTURE.read_text(encoding="utf-8"))
    assert used == {0: 1024, 1: 512}


def test_hugepage_counters_are_ignored():
    # same rows but chip memory lines removed -> nothing to report
    text = "\n".join(
        line
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if "/ 65536" not in line
    )
    assert parse_hbm_used_mb(text) is None


def test_garbage_returns_none():
    assert parse_hbm_used_mb("no table here") is None
    assert parse_hbm_used_mb("") is None
