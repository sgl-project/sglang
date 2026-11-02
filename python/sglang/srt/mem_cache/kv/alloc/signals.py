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


# Blocks on the host until the engine is done with every transfer given, as
# judged alike on every rank of the storage's group, releasing the GIL while it
# waits. Installed by whatever runs the transfers (`KVStorage.set_network_waiter`).
NetworkWaiter = Callable[[Sequence[NetworkTransfer]], None]


def signals_of(key) -> Iterator:
    """The signals a wait or release key holds: a tuple holds several."""
    if isinstance(key, tuple):
        for k in key:
            yield from signals_of(k)
    elif key is not None:
        yield key


def unreleased_transfers(key) -> List[NetworkTransfer]:
    """The network transfers of ``key`` that have not been released."""
    return [
        s for s in signals_of(key) if isinstance(s, NetworkTransfer) and not s.released
    ]


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
    transfers first, on the host through ``network_waiter`` (which every rank
    reaches at the same point, since the decision reads releases only), then
    events, on the GPU."""
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
