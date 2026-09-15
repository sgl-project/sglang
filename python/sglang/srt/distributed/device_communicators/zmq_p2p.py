"""Deadline-free point-to-point channel between scheduler ranks.

Adjacent pipeline stages exchange small Python objects (request lists,
grammar/HiRadix sync markers, tensor metadata) that used to ride gloo
``send``/``irecv``. Gloo waits carry a CLOCK_MONOTONIC deadline, so a rank
suspended past its budget — e.g. a VM snapshot/restore that jumps the
guest clock forward — fails every in-flight and subsequent wait on
resume. This channel uses a ZMQ PUSH/PULL pair instead: there is no
application-level deadline, PUSH queues until the connection is up, and
TCP state survives a suspend, so a frozen rank only delays the exchange
instead of failing it.

The channel is addressed by global rank and created once per process at
scheduler init. Every rank binds a PULL socket and publishes its
endpoint through one ``all_gather_object`` on the world CPU group (an
init-time collective, so a bounded gloo wait there is acceptable);
PUSH sockets connect lazily to the ranks this process actually sends to.

Multiple logical protocols may share a peer pair. Like gloo tags, each
message carries a tag so overlapping protocols do not consume each
other's messages; ``recv_from`` buffers messages for other tags until
they are requested.
"""

import logging
import pickle
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch.distributed as dist
import zmq
from zmq import IPV6, PULL, PUSH, Context  # type: ignore

from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)


class ZmqP2PWork:
    """Awaitable handle for a queued PUSH send.

    ``socket.send`` hands the frame to ZMQ's queue, which delivers it once
    the connection is up, so there is nothing left to wait for. Callers
    drain it either as a ``P2PWork`` (``work.work.wait()`` — ``.work``
    resolves to self) or as a raw ``torch.distributed.Work``
    (``work.wait()``).
    """

    @property
    def work(self) -> "ZmqP2PWork":
        return self

    def wait(self):
        return None

    def is_completed(self) -> bool:
        return True


class ZmqP2PChannel:
    """ZMQ PUSH/PULL channel for global-rank-addressed pyobj traffic."""

    def __init__(self, rank: int, connect_ip: Optional[str] = None):
        if connect_ip is None:
            connect_ip = get_local_ip_auto("0.0.0.0")
        self.rank = rank
        self.connect_ip = connect_ip
        self.context = Context()
        self.pull_socket = self.context.socket(PULL)
        na_is_ipv6 = ":" in connect_ip
        if na_is_ipv6:
            self.pull_socket.setsockopt(IPV6, 1)
            port = self.pull_socket.bind_to_random_port(f"tcp://[{connect_ip}]")
            self.endpoint = f"tcp://[{connect_ip}]:{port}"
        else:
            port = self.pull_socket.bind_to_random_port(f"tcp://{connect_ip}")
            self.endpoint = f"tcp://{connect_ip}:{port}"
        self.push_sockets: Dict[int, zmq.Socket] = {}
        self.endpoints: Dict[int, str] = {}
        # Out-of-order arrivals stashed per (src_rank, tag).
        self._pending: Dict[Tuple[int, int], deque] = defaultdict(deque)

    @classmethod
    def create(cls, cpu_group: dist.ProcessGroup, rank: int) -> "ZmqP2PChannel":
        """Create a channel and exchange endpoints across ``cpu_group``.

        Every member of the group must call this collectively; the
        endpoint exchange is a single init-time gloo allgather.
        """
        channel = cls(rank)
        gathered = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(gathered, (rank, channel.endpoint), group=cpu_group)
        channel.endpoints = dict(gathered)
        return channel

    def _push(self, dst_rank: int) -> zmq.Socket:
        sock = self.push_sockets.get(dst_rank)
        if sock is None:
            assert dst_rank in self.endpoints, f"No endpoint for global rank {dst_rank}"
            assert dst_rank != self.rank, "Cannot send to self"
            sock = self.context.socket(PUSH)
            if self.endpoints[dst_rank].startswith("tcp://["):
                sock.setsockopt(IPV6, 1)
            sock.connect(self.endpoints[dst_rank])
            self.push_sockets[dst_rank] = sock
        return sock

    def send_to(
        self, dst_rank: int, obj: Any, tag: int = 0, async_send: bool = False
    ) -> List[ZmqP2PWork]:
        """Queue ``obj`` for delivery to ``dst_rank``'s channel.

        ``async_send`` only changes the return value: a list with a
        completed work handle, matching the ``List[P2PWork]`` shape gloo
        callers drain with ``work.wait()``.
        """
        self._push(dst_rank).send(
            pickle.dumps((self.rank, tag, obj), protocol=pickle.HIGHEST_PROTOCOL)
        )
        return [ZmqP2PWork()] if async_send else []

    def recv_from(self, src_rank: int, tag: int = 0) -> Any:
        """Return the next object sent by ``src_rank`` under ``tag``.

        Blocks on the PULL socket; there is no deadline to exceed.
        Messages for other (src, tag) pairs are buffered and replayed on
        later calls.
        """
        key = (src_rank, tag)
        if self._pending[key]:
            return self._pending[key].popleft()
        while True:
            src, t, obj = pickle.loads(self.pull_socket.recv())
            if (src, t) == key:
                return obj
            self._pending[(src, t)].append(obj)

    def close(self):
        for sock in self.push_sockets.values():
            sock.close(linger=0)
        self.pull_socket.close(linger=0)
        self.context.term()
