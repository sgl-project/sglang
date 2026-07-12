import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_progressive_policy.py"
)


def load_policy_module():
    spec = importlib.util.spec_from_file_location("pd_flip_progressive_policy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_slo_decision_requires_prefill_risk_and_decode_headroom():
    m = load_policy_module()
    assert (
        m.evaluate_slo_decision(14, 20, 19, 20, 0.9, 20, 20)
        is m.ProgressiveDecision.START
    )
    assert (
        m.evaluate_slo_decision(
            14, 20, 18, 20, 0.9, 20, 20, observing=True
        )
        is m.ProgressiveDecision.COMMIT
    )
    assert (
        m.evaluate_slo_decision(18, 20, 19, 20, 0.9, 20, 20)
        is m.ProgressiveDecision.RECOVER
    )
    assert (
        m.evaluate_slo_decision(14, 20, 17, 20, 0.9, 20, 20)
        is m.ProgressiveDecision.RECOVER
    )
    assert (
        m.evaluate_slo_decision(7, 10, 19, 20, 0.9, 20, 20)
        is m.ProgressiveDecision.INSUFFICIENT_SAMPLES
    )


def test_ratio_halves_until_first_n_requests_fit():
    m = load_policy_module()
    reqs = [m.RequestCapacity(str(i), 100) for i in range(8)]
    out = m.select_first_batch(
        reqs,
        0.75,
        target_req_slots=3,
        target_kv_tokens=450,
        reserve_tokens_per_req=25,
    )
    assert out.configured_ratio == 0.75
    assert out.effective_ratio == 0.375
    assert out.selected_rids == ("0", "1", "2")
    assert out.fallback_count == 1


def test_ratio_returns_none_when_one_request_cannot_fit():
    m = load_policy_module()
    reqs = [m.RequestCapacity("r0", 100)]
    assert m.select_first_batch(reqs, 0.5, 1, 99, 0) is None
