"""CPU test of the TGV per-M range registry (register_tgv_m_ranges) that the
Kimi-K3.1 dense-GEMM tables are registered through."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("cutlass")

from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as cg  # noqa: E402
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
