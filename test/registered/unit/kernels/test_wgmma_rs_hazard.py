"""CPU tests for the register-input wgmma SASS check and the load_jit hook that runs it on fresh builds."""

from __future__ import annotations

import pathlib

import pytest

from sglang.kernels.jit.utils.compile import cache, loader, ninja
from sglang.kernels.jit.utils.wgmma_rs_hazard import check_sass
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# name: (violations expected, SASS body). Each hazard case is the shape of a real miscompile: a write into a pending
# RS HGMMA's A registers, reached directly, through a wide load, a second destination operand or a loop back edge.
CASES = {
    "retired_then_reused": (
        0,
        """
        /*0000*/  WARPGROUP.ARRIVE ;
        /*0010*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  LDS.128 R40, [R2] ;
        /*0040*/  EXIT ;""",
    ),
    "overwrite_in_flight": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  MOV R43, R2 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;""",
    ),
    "wide_load_overlap": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  LDS.64 R38, [R2] ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;""",
    ),
    "only_via_back_edge": (
        1,
        """
        /*0000*/  LDS.128 R40, [R2] ;
        /*0010*/  WARPGROUP.ARRIVE ;
        /*0020*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0040*/  @P0 BRA 0x0 ;
        /*0050*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0060*/  EXIT ;""",
    ),
    "ping_pong_two_sets": (
        0,
        """
        /*0000*/  LDS.128 R40, [R2] ;
        /*0010*/  WARPGROUP.ARRIVE ;
        /*0020*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0040*/  LDS.128 R44, [R2+0x10] ;
        /*0050*/  WARPGROUP.ARRIVE ;
        /*0060*/  HGMMA.64x32x16.F32.BF16 R24, R44, gdesc[UR8], R24, gsb0 ;
        /*0070*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0080*/  @P0 BRA 0x0 ;
        /*0090*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*00a0*/  EXIT ;""",
    ),
    "chained_no_gsb0_stays_pending": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24 ;
        /*0010*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0020*/  F2FP.BF16.F32.PACK_AB R41, R3, R2 ;
        /*0030*/  HGMMA.64x32x16.F32.BF16 R24, R44, gdesc[UR8], R24, gsb0 ;
        /*0040*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0050*/  EXIT ;""",
    ),
    "acc_read_in_flight": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, gdesc[UR4], R24, gsb0 ;
        /*0010*/  FADD R2, R30, R3 ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;""",
    ),
    "shfl_second_operand_dest": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  SHFL.IDX PT, R42, R2, 0x1, 0x1f ;
        /*0020*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0030*/  EXIT ;""",
    ),
    "predicated_wait_may_not_retire": (
        1,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  @P0 WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0020*/  MOV R40, R2 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0040*/  EXIT ;""",
    ),
    "taken_branch_skips_writer": (
        0,
        """
        /*0000*/  HGMMA.64x32x16.F32.BF16 R24, R40, gdesc[UR4], R24, gsb0 ;
        /*0010*/  BRA 0x30 ;
        /*0020*/  MOV R40, R2 ;
        /*0030*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0040*/  EXIT ;""",
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_check_sass(name):
    """Fails on: a parser that misses a destination width or operand, a model that retires groups too early or
    ignores back edges (false passes), or one that keeps retired groups pending (false failures)."""
    want, body = CASES[name]
    rs, problems = check_sass(f"\t\tFunction : {name}\n{body}")
    assert len(problems) == want, problems
    assert rs == (
        0
        if name == "acc_read_in_flight"
        else body.count("R40, gdesc") + body.count("R44, gdesc")
    )


def test_post_build_rejection_publishes_nothing(tmp_path, monkeypatch):
    """A build whose post_build check raises is neither loaded nor cached. Fails on: a hook that runs after the
    publish (the rejected .so would be reused by every later process) or not at all."""
    monkeypatch.setenv("SGLANG_JIT_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(cache, "cache_root", lambda: tmp_path)

    def fake_build(*, spec, build_dir, build_file):
        build_dir.mkdir(parents=True, exist_ok=True)
        lib = build_dir / f"{spec.module_name}.so"
        lib.write_bytes(b"not a library")
        return lib

    monkeypatch.setattr(ninja, "build", fake_build)
    monkeypatch.setattr(
        loader, "_load", lambda lib: pytest.fail("a rejected build was loaded")
    )
    seen = []

    def reject(lib: pathlib.Path):
        seen.append(lib)
        raise RuntimeError("hazard")

    with pytest.raises(RuntimeError, match="hazard"):
        loader.load_jit("wgmma_rs_hazard_probe", cuda_files=[], post_build=reject)
    assert len(seen) == 1
    assert not [p for p in tmp_path.rglob("*.so")], "the rejected build was cached"
