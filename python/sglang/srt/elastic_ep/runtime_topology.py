from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from typing import Optional
from urllib.parse import quote

from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.environ import envs

_RUNTIME_TOPOLOGY_KEY = "elastic_ep/runtime_topology"
_MEMBERSHIP_OBSERVATION_KEY_PREFIX = "elastic_ep/membership_observation"
_MEMBERSHIP_EPOCH_KEY_PREFIX = "elastic_ep/membership_epoch"
_MEMBERSHIP_LATEST_KEY_PREFIX = "elastic_ep/membership_latest"
_MEMBERSHIP_SLOT_KEY_PREFIX = "elastic_ep/membership_slot"
_MEMBERSHIP_PUBLICATION_LOCK = threading.RLock()


class ElasticEPRecoveryRequiredError(RuntimeError):
    """The requested rank range belonged to an earlier topology."""


@dataclass(frozen=True)
class RuntimeTopology:
    runtime_instance_id: str
    initial_ep_size: int
    allocation_width: int
    effective_ep_size: int
    max_committed_ep_size: int
    topology_generation: int = 0

    def validate(self) -> None:
        if not self.runtime_instance_id:
            raise ValueError("Elastic EP runtime instance ID must not be empty.")
        if self.initial_ep_size <= 0:
            raise ValueError("Elastic EP initial size must be positive.")
        if self.allocation_width <= 0:
            raise ValueError("Elastic EP allocation width must be positive.")
        if self.initial_ep_size % self.allocation_width != 0:
            raise ValueError(
                "Elastic EP initial size must be divisible by the allocation "
                f"width (initial={self.initial_ep_size}, "
                f"width={self.allocation_width})."
            )
        if not (0 < self.effective_ep_size <= self.max_committed_ep_size):
            raise ValueError(
                "Elastic EP runtime topology requires "
                "0 < effective_ep_size <= max_committed_ep_size "
                f"(effective={self.effective_ep_size}, "
                f"max_committed={self.max_committed_ep_size})."
            )
        if self.topology_generation < 0:
            raise ValueError("Elastic EP topology generation must be non-negative.")


@dataclass(frozen=True)
class MembershipObservation:
    runtime_instance_id: str
    topology_generation: int
    rank: int
    failed_allocation_incarnation: str
    membership_epoch: int

    def validate(self, topology: RuntimeTopology) -> None:
        if self.runtime_instance_id != topology.runtime_instance_id:
            raise ValueError("Membership observation runtime instance is stale.")
        if self.topology_generation != topology.topology_generation:
            raise ValueError("Membership observation topology generation is stale.")
        if self.rank < 0:
            raise ValueError("Membership observation rank must be non-negative.")
        if self.rank >= topology.max_committed_ep_size:
            raise ValueError("Membership observation rank is outside the topology.")
        if not self.failed_allocation_incarnation:
            raise ValueError(
                "Membership observation allocation incarnation is required."
            )
        if self.membership_epoch < 0:
            raise ValueError("Membership observation epoch must be non-negative.")


def _membership_observation_key(observation: MembershipObservation) -> str:
    return (
        f"{_MEMBERSHIP_OBSERVATION_KEY_PREFIX}/"
        f"{quote(observation.runtime_instance_id, safe='')}/"
        f"{observation.topology_generation}/"
        f"{observation.rank}/"
        f"{quote(observation.failed_allocation_incarnation, safe='')}/"
        f"{observation.membership_epoch}"
    )


def _membership_epoch_key(observation: MembershipObservation) -> str:
    return (
        f"{_MEMBERSHIP_EPOCH_KEY_PREFIX}/"
        f"{quote(observation.runtime_instance_id, safe='')}/"
        f"{observation.topology_generation}/"
        f"{observation.rank}/"
        f"{quote(observation.failed_allocation_incarnation, safe='')}"
    )


def _membership_latest_key(observation: MembershipObservation) -> str:
    return (
        f"{_MEMBERSHIP_LATEST_KEY_PREFIX}/"
        f"{quote(observation.runtime_instance_id, safe='')}/"
        f"{observation.topology_generation}"
    )


def _membership_slot_key(observation: MembershipObservation) -> str:
    return (
        f"{_MEMBERSHIP_SLOT_KEY_PREFIX}/"
        f"{quote(observation.runtime_instance_id, safe='')}/"
        f"{observation.topology_generation}/{observation.rank}"
    )


def _membership_incarnation_key(observation: MembershipObservation) -> str:
    return (
        f"{_MEMBERSHIP_SLOT_KEY_PREFIX}/seen/"
        f"{quote(observation.runtime_instance_id, safe='')}/"
        f"{observation.topology_generation}/{observation.rank}/"
        f"{quote(observation.failed_allocation_incarnation, safe='')}"
    )


def publish_membership_observation(
    observation: MembershipObservation,
    *,
    deadline_unix_seconds: Optional[float] = None,
) -> str:
    """Publish one CPU-only, monotonic membership observation.

    Publication is serialized so the per-slot incarnation and latest snapshot
    cannot be observed halfway through an update. Replaying the exact same
    observation is idempotent so every surviving scheduler can acknowledge a
    shared publication. The epoch watermark still rejects stale and ABA
    observations.
    """
    topology = get_runtime_topology()
    if topology is None:
        raise RuntimeError("Elastic EP runtime topology has not been published.")
    observation.validate(topology)
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP requires the global TCPStore.")

    with _MEMBERSHIP_PUBLICATION_LOCK:
        if deadline_unix_seconds is not None and time.time() >= deadline_unix_seconds:
            raise TimeoutError(
                "Membership observation expired before publication; "
                "no state was changed."
            )
        observation_key = _membership_observation_key(observation)
        encoded = json.dumps(asdict(observation), sort_keys=True).encode()
        if store.check([observation_key]):
            if store.get(observation_key) != encoded:
                raise ValueError("Conflicting Elastic EP membership observation.")
        epoch_key = _membership_epoch_key(observation)
        slot_key = _membership_slot_key(observation)
        incarnation_key = _membership_incarnation_key(observation)
        if store.check([incarnation_key]):
            current = json.loads(store.get(slot_key).decode())
            if current["failed_allocation_incarnation"] != (
                observation.failed_allocation_incarnation
            ):
                raise ValueError("Stale or ABA Elastic EP membership observation.")
        if store.check([epoch_key]):
            previous_epoch = int(store.get(epoch_key).decode())
            if observation.membership_epoch < previous_epoch:
                raise ValueError("Stale or ABA Elastic EP membership observation.")
            if (
                observation.membership_epoch == previous_epoch
                and store.check([slot_key])
                and store.get(slot_key) == encoded
            ):
                return observation_key
        if hasattr(store, "compare_set"):
            existing = store.compare_set(observation_key, b"", encoded)
            if existing != encoded:
                raise ValueError("Conflicting Elastic EP membership observation.")
            store.set(epoch_key, str(observation.membership_epoch).encode())
        else:
            store.set(observation_key, encoded)
            store.set(epoch_key, str(observation.membership_epoch).encode())
        store.set(incarnation_key, b"1")
        store.set(slot_key, encoded)
        store.set(_membership_latest_key(observation), encoded)
        return observation_key


def get_membership_observation() -> Optional[MembershipObservation]:
    topology = get_runtime_topology()
    store = get_global_tcp_store()
    if topology is None or store is None:
        return None
    key = (
        f"{_MEMBERSHIP_LATEST_KEY_PREFIX}/"
        f"{quote(topology.runtime_instance_id, safe='')}/"
        f"{topology.topology_generation}"
    )
    if not store.check([key]):
        return None
    observation = MembershipObservation(**json.loads(store.get(key).decode()))
    observation.validate(topology)
    return observation


def _decode_runtime_topology(value: bytes) -> RuntimeTopology:
    topology = RuntimeTopology(**json.loads(value.decode()))
    topology.validate()
    return topology


def _read_runtime_topology(store) -> Optional[RuntimeTopology]:
    if not store.check([_RUNTIME_TOPOLOGY_KEY]):
        return None
    return _decode_runtime_topology(store.get(_RUNTIME_TOPOLOGY_KEY))


def get_runtime_topology() -> Optional[RuntimeTopology]:
    store = get_global_tcp_store()
    if store is None:
        return None
    return _read_runtime_topology(store)


def publish_runtime_topology(topology: RuntimeTopology) -> None:
    topology.validate()
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP requires the global TCPStore.")

    existing = _read_runtime_topology(store)
    if (
        existing is not None
        and existing.runtime_instance_id != topology.runtime_instance_id
    ):
        raise RuntimeError(
            "Elastic EP runtime topology already belongs to runtime instance "
            f"{existing.runtime_instance_id}, not {topology.runtime_instance_id}."
        )
    store.set(
        _RUNTIME_TOPOLOGY_KEY,
        json.dumps(asdict(topology), sort_keys=True).encode(),
    )


def commit_runtime_topology(effective_ep_size: int) -> RuntimeTopology:
    topology = get_runtime_topology()
    if topology is None:
        raise RuntimeError("Elastic EP runtime topology has not been published.")
    committed = replace(
        topology,
        effective_ep_size=effective_ep_size,
        max_committed_ep_size=max(
            topology.max_committed_ep_size,
            effective_ep_size,
        ),
        topology_generation=topology.topology_generation + 1,
    )
    publish_runtime_topology(committed)
    return committed


def validate_append_candidate(
    topology: RuntimeTopology,
    *,
    rank_offset: int,
    allocation_width: int,
    initial_ep_size: int,
) -> None:
    if topology.initial_ep_size != initial_ep_size:
        raise RuntimeError(
            "Joining allocation initial EP size does not match the running world "
            f"(joiner={initial_ep_size}, runtime={topology.initial_ep_size})."
        )
    if topology.allocation_width != allocation_width:
        raise RuntimeError(
            "Joining allocation width does not match the running world "
            f"(joiner={allocation_width}, runtime={topology.allocation_width})."
        )
    if (
        rank_offset != topology.effective_ep_size
        or rank_offset != topology.max_committed_ep_size
    ):
        raise ElasticEPRecoveryRequiredError(
            "Elastic EP allocation offset "
            f"{rank_offset} was previously occupied or is not the next append "
            "position; recovery mode is required "
            f"(effective={topology.effective_ep_size}, "
            f"max_committed={topology.max_committed_ep_size})."
        )


def probe_runtime_topology(
    dist_init_addr: Optional[str],
    *,
    timeout_seconds: float = 1.0,
    attempts: int = 3,
    retry_interval_seconds: float = 0.25,
) -> Optional[RuntimeTopology]:
    """Read a running primary's record without creating a CUDA context.

    Absence is expected while a new initial world is still forming. Later
    append candidates require an authoritative record and reject absence.
    """

    master_ip = None
    if dist_init_addr:
        address = dist_init_addr.removeprefix("tcp://")
        master_ip = address.rsplit(":", 1)[0]
    if not master_ip:
        master_ip = os.environ.get("MASTER_ADDR")
    if not master_ip:
        return None
    if attempts <= 0:
        raise ValueError("Runtime-topology probe attempts must be positive.")

    from torch.distributed import TCPStore

    for attempt in range(attempts):
        try:
            store = TCPStore(
                host_name=master_ip,
                port=envs.SGLANG_TCP_STORE_PORT.get(),
                is_master=False,
                wait_for_workers=False,
                timeout=timedelta(seconds=timeout_seconds),
            )
        except Exception:
            # A new world has no coordination store until its primary has
            # formed the distributed runtime. Retry to avoid treating a brief
            # connection failure as proof that no runtime exists.
            if attempt + 1 < attempts:
                time.sleep(retry_interval_seconds)
            continue
        return _read_runtime_topology(store)
    return None
