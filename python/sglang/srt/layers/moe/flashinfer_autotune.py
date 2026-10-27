"""Decode profiles plus one prefill profile for FlashInfer MoE startup tuning."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import cache, wraps

_active_limits = ContextVar("flashinfer_moe_autotune_limits", default=None)


@contextmanager
def flashinfer_moe_autotune_context(
    *,
    decode_num_tokens: int,
    prefill_num_tokens: int,
    skip_ops: set[str] | frozenset[str] = frozenset(),
):
    token = _active_limits.set(
        (decode_num_tokens, prefill_num_tokens, frozenset(skip_ops))
    )
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
        decode_num_tokens, prefill_num_tokens, skip_ops = limits
        owner._flashinfer_moe_tuning_policy = (
            (_tuning_buckets(decode_num_tokens, prefill_num_tokens), skip_ops)
            if decode_num_tokens and prefill_num_tokens
            else None
        )
    policy = getattr(owner, "_flashinfer_moe_tuning_policy", None)
    if policy is None:
        return nullcontext()

    from flashinfer.autotuner import autotune

    buckets, skip_ops = policy
    if num_tokens > buckets[-1] and limits is None:
        # Do not clamp an oversized serving input to an undersized profile.
        return autotune(False, skip_ops=skip_ops) if skip_ops else nullcontext()

    # False inherits the enclosing startup autotune(True), but never enables
    # profiling in serving. Both phases must use the same sparse ceil mapping:
    # every token count above decode and through prefill selects the one prefill.
    # An opt-in EXTEND dummy can be padded above prefill; still tune only this set.
    # Keep skipped ops on their heuristic path before any mapped cache lookup.
    return autotune(False, tuning_buckets=buckets, round_up=True, skip_ops=skip_ops)


def flashinfer_moe_autotune_fused_func(func):
    @wraps(func)
    def wrapped(dispatch_output, quant_info, runner_config, *args, **kwargs):
        with flashinfer_moe_autotune(
            dispatch_output.hidden_states.shape[0], runner_config
        ):
            return func(dispatch_output, quant_info, runner_config, *args, **kwargs)

    return wrapped
