"""Vectorized oracle batch helpers (predict_input_tokens_for_plan / predict_next_tokens_for_active_batch)."""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def test_predict_input_tokens_for_plan_batch() -> None:
    pass


def test_predict_input_tokens_for_plan_matches_scalar() -> None:
    pass


def test_predict_next_tokens_for_active_batch_batch() -> None:
    pass


def test_predict_next_tokens_matches_scalar() -> None:
    pass


def test_vectorized_handles_chunked_offsets() -> None:
    pass
