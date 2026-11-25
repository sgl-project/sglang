"""Account for media bytes while request workers materialize their inputs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Lock

# same encoded ingress allowance as sglang-mm's MAX_REQUEST_BYTES
MAX_REQUEST_BYTES = 1280 << 20


# Limit the materialized PCM or RGB output of a decode independently of ingress.
MAX_DECODED_BYTES = 1280 << 20


class MediaBudgetExceeded(ValueError):
    pass


def check_decoded_bytes(size: int):
    if size > MAX_DECODED_BYTES:
        raise MediaBudgetExceeded("Decoded media exceeds the byte budget")


class ByteBudget:
    def __init__(self, limit: int):
        self.remaining = limit
        self.lock = Lock()

    def charge(self, size: int):
        with self.lock:
            if size > self.remaining:
                raise MediaBudgetExceeded("Media exceeds the request byte budget")
            self.remaining -= size


_active: ContextVar[ByteBudget | None] = ContextVar("media_byte_budget", default=None)


@contextmanager
def media_budget(limit: int = MAX_REQUEST_BYTES):
    budget = ByteBudget(limit)
    token = _active.set(budget)
    try:
        yield budget
    finally:
        _active.reset(token)


def charge_media_bytes(size: int):
    budget = _active.get()
    if budget is not None:
        budget.charge(size)


# A snapshot has already paid for this exact immutable buffer. Other sources
# decoded by the same worker still consume the active request's allowance.
_admitted = ContextVar("admitted_media_buffer", default=None)


@contextmanager
def decoding_admitted_media(data):
    token = _admitted.set(data)
    try:
        yield
    finally:
        _admitted.reset(token)


def charge_media_buffer(data: bytes):
    if data is not _admitted.get():
        charge_media_bytes(len(data))


def charge_base64(encoded: str):
    padding = 2 if encoded.endswith("==") else int(encoded.endswith("="))
    charge_media_bytes(max(0, ((len(encoded) + 3) // 4) * 3 - padding))


class MediaThreadPool(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(copy_context().run, fn, *args, **kwargs)


def clear_inherited_media_budget():
    _active.set(None)
    _admitted.set(None)


@contextmanager
def ensure_media_budget():
    if _active.get() is not None:
        yield
    else:
        with media_budget():
            yield
