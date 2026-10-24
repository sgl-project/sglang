"""Opt-in Kimi K3 MHA region between prefill graph segments."""

from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("k3_bcg_mha_eager", default=False)


def in_bcg_mha_eager():
    return _active.get()


@contextmanager
def bcg_mha_eager_scope():
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)
