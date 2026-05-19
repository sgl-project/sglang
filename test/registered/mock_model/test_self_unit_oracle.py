import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="phase-2; awaits mock_mode/pseudo_mode subsystem reimplementation"
)


def test_case1_pure_prefill_predict_input_token() -> None:
    raise NotImplementedError


def test_case2_decode_kth_output() -> None:
    raise NotImplementedError


def test_case3_chunked_prefill_transparent() -> None:
    raise NotImplementedError


def test_case4_prefix_cache_hit_no_replay() -> None:
    raise NotImplementedError


def test_case5_preempt_resume_oracle_unchanged() -> None:
    raise NotImplementedError


def test_case6_overlap_scheduler_delayed_sample() -> None:
    raise NotImplementedError


def test_case7_spec_draft_eagle() -> None:
    raise NotImplementedError


def test_case8_eos_oracle_signals_stop() -> None:
    raise NotImplementedError


def test_admit_initializes_req_pool_to_id_mapping() -> None:
    raise NotImplementedError


def test_finish_clears_req_pool_to_id_mapping() -> None:
    raise NotImplementedError


def test_case9_dp_attention_oos_marker() -> None:
    raise NotImplementedError


def test_case10_multimodal_oos_marker() -> None:
    raise NotImplementedError
