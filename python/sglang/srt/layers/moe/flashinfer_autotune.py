"""Prefill token coverage for FlashInfer MoE startup tuning."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

from sglang.srt.utils.common import next_power_of_2

_prefill_num_tokens = ContextVar("flashinfer_moe_prefill_num_tokens", default=0)


@contextmanager
def flashinfer_moe_autotune_context(prefill_num_tokens: int):
    """Extend kernel profiles without enlarging the model's dummy forward."""
    token = _prefill_num_tokens.set(prefill_num_tokens)
    try:
        yield
    finally:
        _prefill_num_tokens.reset(token)


def flashinfer_moe_tune_max_num_tokens(num_tokens: int) -> int:
    # Outside target startup, retain the per-call ceiling and cache mapping.
    return next_power_of_2(max(num_tokens, _prefill_num_tokens.get()))


def flashinfer_cutedsl_moe_autotune(num_tokens: int):
    if not _prefill_num_tokens.get():
        return nullcontext()

    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    # CuTe DSL normally generates buckets only up to the live input's M.
    # Preserve its native hybrid spacing and round-up lookup, including the
    # bucket covering an unaligned prefill ceiling. FlashInfer synthesizes the
    # profiling tensors; the live activation/routing/output shapes stay intact.
    return autotune(
        tuning_buckets=get_hybrid_num_tokens_buckets(
            flashinfer_moe_tune_max_num_tokens(num_tokens)
        ),
        round_up=True,
    )
