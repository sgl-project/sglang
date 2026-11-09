"""What page moves wait for.

A page move must not copy a page some unfinished work still writes, and must not
hand out a page it vacated while unfinished work still reads it. Forwards and
host-cache copies run on CUDA streams and signal with events, which a stream
waits on the GPU.
"""

from typing import Iterator, Optional

import torch


class StreamTransfer:
    """The key of a copy a CUDA stream runs on a band's pages (a host-cache
    backup or load), around its event (or a tuple of events). It is no
    forward: no decision waits for it, since whatever writes its pages next
    waits its events on a stream once the decision is made."""

    __slots__ = ("events",)

    def __init__(self, events):
        self.events = events


def keys_of(key) -> Iterator:
    """The members of a release key, each a forward's event or a transfer's
    key: a tuple holds several."""
    if isinstance(key, tuple):
        for k in key:
            yield from keys_of(k)
    elif key is not None:
        yield key


def signals_of(key) -> Iterator:
    """The signals a wait or release key holds: a tuple holds several, and a
    stream transfer its events."""
    for k in keys_of(key):
        if isinstance(k, StreamTransfer):
            yield from signals_of(k.events)
        else:
            yield k


def signal_fired(key) -> bool:
    """Whether every signal of ``key`` has completed: its events fired."""
    return all(s.query() for s in signals_of(key))


def wait_signals(key, stream: Optional[torch.cuda.Stream] = None) -> None:
    """Order ``stream``'s later work after every signal of ``key``, on the GPU."""
    for s in signals_of(key):
        if s.query():
            continue
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(s)
