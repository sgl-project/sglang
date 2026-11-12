# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""CPU test of the TGV per-M range registry (register_tgv_m_ranges) that the
Kimi-K3.1 dense-GEMM tables are registered through, and of the SM107 table
(SGLANG_K31_SM107_GEMM_TACTICS): off by default, registers beside the generic
K3.1 ranges, rejects JSON tables outside the allowed shapes or ranges."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("cutlass")

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as cg  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.models import kimi_k3  # noqa: E402


@pytest.fixture
def registry():
    saved = dict(cg._MODEL_TGV_M_RANGES)
    cg._MODEL_TGV_M_RANGES.clear()
    with (
        patch.object(
            cg,
            "current_arch",
            return_value=SimpleNamespace(num_sms=148, is_sm=lambda *_: False),
        ),
        patch.object(cg.torch.cuda, "current_device", return_value=0),
    ):
        yield
    cg._MODEL_TGV_M_RANGES.clear()
    cg._MODEL_TGV_M_RANGES.update(saved)


def test_k31_tables_merge_into_one_registry(registry):
    n, k = 2112, 7168
    ladder = {m: cg._pick_tactic(m, n, k) for m in (1, 64, 121, 289)}
    for _ in range(2):  # re-registering is a no-op
        cg.register_tgv_m_ranges(kimi_k3._K31_TGV_M_RANGES)
        cg.register_tgv_m_ranges(kimi_k3._K31_GB300_TGV_M_RANGES)
    # Win range (ladder tactic), then the GB300 per-M tactics, then the heuristic.
    for m in (1, 64):
        assert cg.use_cutedsl_bf16_gemm(m, n, k)
        assert cg._pick_tactic(m, n, k) == ladder[m]
    for m, tactic in ((72, 21), (120, 21), (132, 23), (240, 23), (288, 24)):
        assert cg.use_cutedsl_bf16_gemm(m, n, k)
        assert cg._pick_tactic(m, n, k) == tactic
    for m in (121, 289):
        assert not cg._model_takes_tgv(m, n, k)
        assert cg._pick_tactic(m, n, k) == ladder[m]
    # The side-stream [f_a | b] tactics stay off the global registry.
    for (n_side, k_side), ranges in kimi_k3._K31_GB300_BFA_SIDE_TGV_M_RANGES.items():
        for m in range(1, ranges[-1][1] + 1):
            assert not cg.use_cutedsl_bf16_gemm(m, n_side, k_side)


def test_overlapping_ranges_are_rejected(registry):
    cg.register_tgv_m_ranges({(2112, 7168): ((1, 64, None),)})
    with pytest.raises(ValueError):
        cg.register_tgv_m_ranges({(2112, 7168): ((64, 96, 21),)})


_SM107 = SimpleNamespace(device_sm=107, is_blackwell=True)
_SM103 = SimpleNamespace(device_sm=103, is_blackwell=True)


@pytest.fixture
def sm107_registry():
    saved = dict(cg._MODEL_TGV_M_RANGES)
    cg._MODEL_TGV_M_RANGES.clear()
    kimi_k3._k31_sm107_tables.cache_clear()
    with (
        patch.object(
            cg,
            "current_arch",
            return_value=SimpleNamespace(num_sms=148, is_sm=lambda *_: False),
        ),
        patch.object(cg.torch.cuda, "current_device", return_value=0),
        patch.object(kimi_k3, "is_kimi_k31", return_value=True),
    ):
        yield
    kimi_k3._k31_sm107_tables.cache_clear()
    cg._MODEL_TGV_M_RANGES.clear()
    cg._MODEL_TGV_M_RANGES.update(saved)


def test_gate_off_by_default_and_caps(sm107_registry):
    cfg = SimpleNamespace()
    bfa, gate = (
        envs.SGLANG_K31_SM107_BFA_SIDE_MAX_TOKENS,
        envs.SGLANG_K31_SM107_GATE_SIDE_MAX_TOKENS,
    )
    with patch.object(kimi_k3, "get_platform", return_value=_SM107):
        assert not kimi_k3._k31_sm107_tuning(cfg)
        assert kimi_k3._k3_side_stream_max_tokens(cfg, bfa) == 128
        assert kimi_k3._k3_side_stream_max_tokens(cfg, gate) == 128
        # Gate on with no cap env: [f_a | b] side stream to 1152 rows, MLA gate 128.
        with envs.SGLANG_K31_SM107_GEMM_TACTICS.override(True):
            assert kimi_k3._k3_side_stream_max_tokens(cfg, bfa) == 1152
            assert kimi_k3._k3_side_stream_max_tokens(cfg, gate) == 128
        with (
            envs.SGLANG_K31_SM107_GEMM_TACTICS.override(True),
            bfa.override(480),
            gate.override(240),
        ):
            assert kimi_k3._k31_sm107_tuning(cfg)
            assert kimi_k3._k3_side_stream_max_tokens(cfg, bfa) == 480
            assert kimi_k3._k3_side_stream_max_tokens(cfg, gate) == 240
            assert kimi_k3._k3_side_stream_max_tokens(cfg) == 128
    with (
        patch.object(kimi_k3, "get_platform", return_value=_SM103),
        envs.SGLANG_K31_SM107_GEMM_TACTICS.override(True),
        bfa.override(480),
    ):
        assert not kimi_k3._k31_sm107_tuning(cfg)
        assert kimi_k3._k3_side_stream_max_tokens(cfg, bfa) == 288


def test_builtin_table_registers_beside_generic_ranges(sm107_registry):
    dense, side = kimi_k3._k31_sm107_tables()
    for _ in range(2):  # re-registering is a no-op
        cg.register_tgv_m_ranges(kimi_k3._K31_TGV_M_RANGES)
        cg.register_tgv_m_ranges(dense)
    for (n, k), ranges in dense.items():
        assert (n, k) in kimi_k3._K31_SM107_TABLE_SHAPES
        for m_lo, m_hi, tactic in ranges:
            for m in (m_lo, m_hi):
                assert cg.use_cutedsl_bf16_gemm(m, n, k)
                assert cg._pick_tactic(m, n, k) == tactic
    # The side-stream [f_a | b] tactics stay off the global registry.
    for (n, k), ranges in side.items():
        for m in range(1, ranges[-1][1] + 1):
            assert not cg._model_takes_tgv(m, n, k)


def test_json_table(sm107_registry, tmp_path):
    doc = {
        "table": {
            "6144,7168": [[312, 336, 24], [432, 480, 27]],
            "7168,3584": [[33, 48, 23]],
        },
        "side": {"144,7168": [[240, 264, 21]]},
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(doc))
    with envs.SGLANG_K31_SM107_GEMM_TABLE.override(str(path)):
        dense, side = kimi_k3._k31_sm107_tables()
    assert dense == {
        (6144, 7168): ((312, 336, 24), (432, 480, 27)),
        (7168, 3584): ((33, 48, 23),),
    }
    assert side == {(144, 7168): ((240, 264, 21),)}
    cg.register_tgv_m_ranges(kimi_k3._K31_TGV_M_RANGES)
    cg.register_tgv_m_ranges(dense)
    assert cg._pick_tactic(456, 6144, 7168) == 27
    assert not cg._model_takes_tgv(360, 6144, 7168)


@pytest.mark.parametrize(
    "section, allowed",
    [
        ({"6016,7168": [[100, 200, 21]]}, "table"),  # shape not allowed
        ({"144,7168": [[100, 200, 21]]}, "table"),  # side shape in the dense table
        ({"6144,7168": [[64, 200, 21]]}, "table"),  # overlaps the generic 1-64 range
        ({"6144,7168": [[100, 5000, 21]]}, "table"),  # m beyond 4096
        ({"6144,7168": [[100, 200, -1]]}, "table"),  # negative tactic = default
        ({"6144,7168": [[100.0, 200, 21]]}, "table"),  # non-int m
        ({"6144,7168": [[True, 200, 21]]}, "table"),  # bool m
        ({"6144,7168": [[100, 200, 21]]}, "side"),  # dense shape in the side table
    ],
)
def test_bad_tables_rejected(section, allowed):
    shapes = (
        kimi_k3._K31_SM107_TABLE_SHAPES
        if allowed == "table"
        else kimi_k3._K31_SM107_SIDE_SHAPES
    )
    with pytest.raises(ValueError):
        kimi_k3._k31_sm107_parse_ranges(section, shapes)


# The built-in SM107 table is this measured file's "table" / "side" content.
_STABLE_TABLE = (
    Path(__file__).resolve().parents[4]
    / "benchmark/kimi_k31_vr200/sm107_tables/sm107_table-stable.json"
)
_STABLE_TABLE_SHA256 = (
    "358899b130b8329b8d365b181f396c299ed6d8e9c10d7dcbc1e3529d1ecf3e40"
)


def _as_json_section(ranges: dict) -> dict:
    return {f"{n},{k}": [list(r) for r in rows] for (n, k), rows in ranges.items()}


def test_builtin_table_is_the_stable_table(sm107_registry):
    raw = _STABLE_TABLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _STABLE_TABLE_SHA256
    doc = json.loads(raw)
    # Same shapes, ranges, tactics and order as the file.
    assert _as_json_section(kimi_k3._K31_SM107_TGV_M_RANGES) == doc["table"]
    assert list(_as_json_section(kimi_k3._K31_SM107_TGV_M_RANGES)) == list(doc["table"])
    assert _as_json_section(kimi_k3._K31_SM107_BFA_SIDE_TGV_M_RANGES) == doc["side"]
    assert envs.SGLANG_K31_SM107_BFA_SIDE_MAX_TOKENS.default == doc["side_cap_bfa"]
    # The built-in table passes the JSON loader's checks, and loading the file
    # gives the same tables as the built-in path.
    builtin = kimi_k3._k31_sm107_tables()
    assert builtin == (
        kimi_k3._K31_SM107_TGV_M_RANGES,
        kimi_k3._K31_SM107_BFA_SIDE_TGV_M_RANGES,
    )
    kimi_k3._k31_sm107_tables.cache_clear()
    with envs.SGLANG_K31_SM107_GEMM_TABLE.override(str(_STABLE_TABLE)):
        assert kimi_k3._k31_sm107_tables() == builtin
