"""Execution-scoped SM budget for persistent compute kernels."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_compute_num_sms: ContextVar[int | None] = ContextVar("compute_num_sms", default=None)


def get_compute_num_sms() -> int | None:
    return _compute_num_sms.get()


@contextmanager
def compute_sm_budget(num_sms: int | None) -> Iterator[None]:
    """Apply a compute budget; None leaves any enclosing budget unchanged.

    Explicit-argument backends read get_compute_num_sms(). DeepGEMM instead
    needs its process-wide setting applied across both metadata and compute,
    including attention/indexer calls that bypass the MoE runner.
    """
    if num_sms is None:
        yield
        return

    from sglang.srt.layers.deep_gemm_wrapper import configure_deep_gemm_num_sms

    token = _compute_num_sms.set(num_sms)
    try:
        with configure_deep_gemm_num_sms(num_sms):
            yield
    finally:
        _compute_num_sms.reset(token)
