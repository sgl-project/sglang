"""Decode profiles plus one prefill profile for FlashInfer MoE startup tuning."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import cache, wraps

_active_limits = ContextVar("flashinfer_moe_autotune_limits", default=None)


@contextmanager
def flashinfer_moe_autotune_context(*, decode_num_tokens: int, prefill_num_tokens: int):
    token = _active_limits.set((decode_num_tokens, prefill_num_tokens))
    try:
        yield
    finally:
        _active_limits.reset(token)


@cache
def _tuning_buckets(decode_num_tokens: int, prefill_num_tokens: int):
    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    # Preserve native decode spacing; add only the exact prefill shape.
    return tuple(
        sorted(
            set(get_hybrid_num_tokens_buckets(decode_num_tokens)) | {prefill_num_tokens}
        )
    )


def flashinfer_moe_autotune(num_tokens: int, owner):
    """Install profiles at startup and reuse their mapping during serving.

    The owner is a persistent runner config or layer, never per-call quant_info.
    Draft/PD-decode startup installs no override on its own owners.
    """
    limits = _active_limits.get()
    if limits is not None:
        owner._flashinfer_moe_tuning_buckets = (
            _tuning_buckets(*limits) if all(limits) else None
        )
    buckets = getattr(owner, "_flashinfer_moe_tuning_buckets", None)
    if not buckets:
        return nullcontext()
    if num_tokens > buckets[-1]:
        if limits is not None:
            raise ValueError("FlashInfer MoE startup input exceeds its profile ceiling")
        # Do not clamp an oversized serving input to an undersized profile.
        return nullcontext()

    from flashinfer.autotuner import autotune

    # False inherits the enclosing startup autotune(True), but never enables
    # profiling in serving. Both phases must use the same sparse ceil mapping:
    # every token count above decode and through prefill selects the one prefill.
    return autotune(False, tuning_buckets=buckets, round_up=True)


def flashinfer_moe_autotune_fused_func(func):
    @wraps(func)
    def wrapped(dispatch_output, quant_info, runner_config, *args, **kwargs):
        with flashinfer_moe_autotune(
            dispatch_output.hidden_states.shape[0], runner_config
        ):
            return func(dispatch_output, quant_info, runner_config, *args, **kwargs)

    return wrapped
