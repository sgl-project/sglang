"""What page moves wait for.

A page move must not copy a page some unfinished work still writes, and must not
hand out a page it vacated while unfinished work still reads it. Forwards and
host-cache copies run on CUDA streams and signal with events, which a stream
waits on the GPU. A network transfer is driven by its engine outside every
stream: only the host learns that it ended, so a move that must wait for one
waits on the host (`NetworkTransfer`), never on the GPU.
"""

from typing import Callable, Iterator, List, Optional, Sequence

import torch


class TransferRelease:
    """The logical end of a transfer whose completion the host observes: set
    where every rank observes it, so move and reuse decisions that read it agree
    across ranks."""

    __slots__ = ("released",)

    def __init__(self):
        self.released = False


class ReceiveHold(TransferRelease):
    """Receive rows held in place until their destination is published: no
    engine touches them yet, so nothing waits on them, but a relief keeps them
    where they are when it can and an opportunistic compaction leaves them."""

    __slots__ = ()


class NetworkTransfer(TransferRelease):
    """One send or receive a network engine runs on a request's rows. It is
    released where the transfer's owner observes its end on every rank; a move
    that cannot wait for that asks the storage's `NetworkWaiter`. ``peer`` is
    what the waiter judges the transfer by, opaque here."""

    __slots__ = ("peer",)

    def __init__(self, peer=None):
        super().__init__()
        self.peer = peer


# Blocks on the host, releasing the GIL, until the engine is done with every
# transfer given, alike on every rank; ranks give the same set in any order.
NetworkWaiter = Callable[[Sequence[NetworkTransfer]], None]


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


def unreleased_transfers(key) -> List[NetworkTransfer]:
    """The unreleased network transfers of ``key``, each once: how many fences
    hold one can differ across ranks."""
    seen = {}
    for s in signals_of(key):
        if isinstance(s, NetworkTransfer) and not s.released:
            seen.setdefault(id(s), s)
    return list(seen.values())


def signal_fired(key) -> bool:
    """Whether every signal of ``key`` has completed: its events fired and its
    network transfers were released. A receive hold has nothing to wait for."""
    for s in signals_of(key):
        if isinstance(s, TransferRelease):
            if isinstance(s, NetworkTransfer) and not s.released:
                return False
        elif not s.query():
            return False
    return True


def wait_signals(
    key,
    stream: Optional[torch.cuda.Stream] = None,
    *,
    network_waiter: Optional[NetworkWaiter] = None,
) -> None:
    """Order ``stream``'s later work after every signal of ``key``: network
    transfers on the host (``network_waiter``), then events on the GPU."""
    pending = unreleased_transfers(key)
    if pending:
        if network_waiter is None:
            raise RuntimeError(
                "a page move must wait for a network transfer, but no network "
                "waiter is installed on the storage"
            )
        network_waiter(pending)
        for transfer in pending:
            transfer.released = True
    for s in signals_of(key):
        if isinstance(s, TransferRelease) or s.query():
            continue
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(s)
