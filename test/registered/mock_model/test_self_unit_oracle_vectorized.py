"""Vectorized oracle batch helpers (predict_input_tokens_for_plan / predict_next_tokens_for_active_batch).

The new kv_cache_canary.mock_model design exposes only the per-token Oracle.expected_token API;
there is no vectorized batch oracle. fill_expected_inputs in sampler.py loops over tokens in
Python rather than offering a tensor-level vectorized API. Until a vectorized API is added,
these tests cannot run.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="kv_cache_canary.mock_model has no vectorized batch oracle API; only per-token Oracle.expected_token. Add a vectorized helper before un-skipping."
)


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
