"""Internal scheduler-to-launcher metadata for node-local sidecars.

These records travel over the scheduler-ready pipe, never through server info.
Version 1 describes a static set of local KV-event publishers.

With ``--sidecar-scope local-telemetry``, the launcher serializes SidecarContext
as JSON in ``SGLANG_SIDECAR_CONTEXT`` before importing the provider. Providers
must check ``version`` and use only the listed sources. ``main(argv)`` remains
the provider entrypoint.

``mode=full`` retains ``SGLANG_GRPC_ENDPOINT``; ``mode=telemetry`` removes it.
``dist_init_addr`` is the engine rendezvous address shared by the nodes in one
multinode instance. Consumers resolve it when correlating nodes and must handle
worker generations themselves. It is not a Dynamo worker ID.

SGLang launches the provider without waiting for its subscriptions and treats
an unexpected sidecar exit as a node failure. The consumer owns initialization,
cross-node coordination, attribution, replay and recovery. Process launch does
not guarantee lossless ZMQ PUB delivery.
Independent follower-sidecar restarts and live topology changes are unsupported.
"""

from dataclasses import dataclass
from typing import Any, Iterable

LOCAL_KV_EVENT_SOURCES = "_local_kv_event_sources"


@dataclass(frozen=True)
class KvEventSource:
    dp_rank: int
    endpoint: str
    topic: str
    block_size: int
    replay_endpoint: str | None = None


@dataclass(frozen=True)
class SidecarContext:
    mode: str
    node_rank: int
    nnodes: int
    dp_size: int
    dist_init_addr: str | None
    kv_event_sources: list[KvEventSource]
    version: int = 1


def take_local_kv_event_sources(infos: Iterable[dict[str, Any]]) -> list[KvEventSource]:
    """Extract private metadata before the remaining info becomes public.

    Used at both ready-pipe boundaries, including the path without a DP
    controller. Duplicate ranks indicate conflicting publisher ownership.
    """
    sources = [
        source for info in infos for source in info.pop(LOCAL_KV_EVENT_SOURCES, [])
    ]
    sources.sort(key=lambda source: source.dp_rank)
    if len({source.dp_rank for source in sources}) != len(sources):
        raise ValueError("Duplicate local KV-event publisher DP rank")
    if len({source.endpoint for source in sources}) != len(sources):
        raise ValueError("Duplicate local KV-event publisher endpoint")
    return sources
