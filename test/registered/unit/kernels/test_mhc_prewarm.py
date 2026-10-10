"""Unit tests for the DeepSeek V4 mHC prenorm prewarm token budget."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
mhc = pytest.importorskip("sglang.kernels.ops.layernorm.mhc")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prewarm_token_counts(chunked_prefill_size, max_prefill_tokens):
    """Run the prewarm with stubbed kernels; return the token counts it replays."""
    calls = []
    schedule = SimpleNamespace(
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
    )
    residual = torch.zeros(1, 4, 8)
    with (
        patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
        # The real split heuristic reads CUDA device properties; bucket by grid
        # size so every 64-token grid is its own representative.
        patch.object(
            mhc,
            "_compute_num_split_for_mhc_pre",
            side_effect=lambda num_tokens, _: (num_tokens + 63) // 64,
        ),
        patch.object(
            mhc,
            "mhc_pre",
            side_effect=lambda residual, *args, **kwargs: calls.append(
                residual.shape[0]
            ),
        ),
    ):
        mhc.prewarm_mhc_pre(
            residual=residual,
            fn=None,
            hc_scale=None,
            hc_base=None,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=1.0,
            sinkhorn_repeat=1,
            n_splits=1,
            n_splits_pre=1,
            norm_weight=None,
            norm_eps=None,
        )
    return calls


def test_prewarm_is_bounded_by_chunk_size():
    counts = _prewarm_token_counts(chunked_prefill_size=256, max_prefill_tokens=4096)
    assert counts == [64, 128, 192, 256]


@pytest.mark.parametrize("chunked_prefill_size", [-1, 0, None])
def test_prewarm_falls_back_to_prefill_budget_without_chunking(chunked_prefill_size):
    counts = _prewarm_token_counts(
        chunked_prefill_size=chunked_prefill_size, max_prefill_tokens=320
    )
    assert counts == [64, 128, 192, 256, 320]
