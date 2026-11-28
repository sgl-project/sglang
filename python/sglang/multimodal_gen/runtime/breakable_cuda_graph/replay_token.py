"""Replay-token tracking for diffusion BCG replays.

The SRT BCG core does not stamp replays; the diffusion runner sets a fresh
token around each graph replay so replay-local caches (e.g. varlen attention
mask metadata in ``DynamicVarlenMaskMeta``) can be rebuilt once per replay
while still being reused across the break points of that same replay.
``get_current_replay_token`` returns ``None`` outside a replay (including
during capture).
"""

import itertools
from contextlib import contextmanager
from contextvars import ContextVar

_current_replay_token_var: ContextVar[int | None] = ContextVar(
    "mm_bcg_replay_token", default=None
)
_replay_token_counter = itertools.count(1)


def get_current_replay_token() -> int | None:
    return _current_replay_token_var.get()


@contextmanager
def replay_token_scope():
    token = _current_replay_token_var.set(next(_replay_token_counter))
    try:
        yield
    finally:
        _current_replay_token_var.reset(token)
