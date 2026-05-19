"""MockOracle contract self-unit tests covering forward-mode edge cases."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
    from sglang.srt.mock_mode.oracle import MockOracle
except ImportError:
    MockEngine = None
    MockOracle = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


_PROMPT = [11, 22, 33, 44, 55]


def _launch() -> "MockEngine":
    return MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)


def test_case1_pure_prefill_predict_input_token() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=2)

    for position, expected in enumerate(_PROMPT):
        assert engine.oracle.predict_input_token(req=req, position=position) == expected
    engine.shutdown()


def test_case2_decode_kth_output() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=3)
    engine.step_until(req, n=3)
    prefix_len = len(_PROMPT)

    for k in range(1, 3):
        decode_in = engine.oracle.predict_input_token(req=req, position=prefix_len + k)
        prior_out = engine.oracle.predict_output_token(req=req, step=k - 1)
        assert decode_in == prior_out
    engine.shutdown()


def test_case3_chunked_prefill_transparent() -> None:
    long_prompt = list(range(1, 4097))
    engine = _launch()
    req = engine.admit(prompt=long_prompt, max_new_tokens=1)
    engine.step_until(req, n=1)

    for position in (0, 1024, 2048, 4095):
        assert (
            engine.oracle.predict_input_token(req=req, position=position)
            == long_prompt[position]
        )
    engine.shutdown()


def test_case4_prefix_cache_hit_no_replay() -> None:
    engine = _launch()
    req_a = engine.admit(prompt=_PROMPT, max_new_tokens=2)
    engine.step_until(req_a, n=2)
    req_b = engine.admit(prompt=_PROMPT[:3] + [777, 888], max_new_tokens=1)
    engine.step_until(req_b, n=1)

    for position in range(len(_PROMPT)):
        assert (
            engine.oracle.predict_input_token(req=req_b, position=position)
            == ([_PROMPT[0], _PROMPT[1], _PROMPT[2], 777, 888])[position]
        )
    assert engine.oracle.predict_output_token(req=req_b, step=0) is not None
    engine.shutdown()


def test_case5_preempt_resume_oracle_unchanged() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=4)
    engine.step_until(req, n=2)
    before = engine.oracle.predict_output_token(req=req, step=2)

    engine.force_preempt(req)
    engine.resume(req)
    after = engine.oracle.predict_output_token(req=req, step=2)

    assert before == after
    engine.shutdown()


def test_case6_overlap_scheduler_delayed_sample() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B", num_hidden_layers=1, enable_overlap=True
    )
    req = engine.admit(prompt=_PROMPT, max_new_tokens=4)

    engine.step_until(req, n=3)
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_case7_spec_draft_eagle() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B", num_hidden_layers=1, speculative_algorithm="EAGLE"
    )
    req = engine.admit(prompt=_PROMPT, max_new_tokens=4)

    engine.step_until(req, n=4)
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_case8_eos_oracle_signals_stop() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=10, eos_at=3)

    results = engine.step_until(req, n=10)
    eos_id = engine.oracle.eos_token_id

    assert engine.oracle.predict_output_token(req=req, step=3) == eos_id
    assert len(results) <= 10
    engine.shutdown()


def test_admit_initializes_req_pool_to_id_mapping() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=2)
    engine.step()

    mapping = engine.oracle.req_pool_to_id_snapshot()

    assert req.req_id in mapping.values()
    engine.shutdown()


def test_finish_clears_req_pool_to_id_mapping() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=1)
    engine.step_until_idle(max_steps=10)

    mapping = engine.oracle.req_pool_to_id_snapshot()

    assert req.req_id not in mapping.values()
    engine.shutdown()


def test_case9_dp_attention_oos_marker() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=1)

    with pytest.raises(NotImplementedError):
        engine.oracle.predict_input_token_dp(req=req, position=0)
    engine.shutdown()


def test_case10_multimodal_oos_marker() -> None:
    engine = _launch()
    req = engine.admit(prompt=_PROMPT, max_new_tokens=1)

    with pytest.raises(NotImplementedError):
        engine.oracle.predict_input_token_multimodal(req=req, position=0)
    engine.shutdown()
