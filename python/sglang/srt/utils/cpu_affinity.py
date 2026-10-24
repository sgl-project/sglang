"""CPU allocation shared by accelerator topology providers."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for item in value.strip().split(","):
        bounds = item.split("-")
        if not 1 <= len(bounds) <= 2 or not all(x.isdecimal() for x in bounds):
            raise ValueError(f"Invalid CPU list: {value!r}")
        first, last = int(bounds[0]), int(bounds[-1])
        if first > last:
            raise ValueError(f"Invalid CPU range: {item!r}")
        cpus.update(range(first, last + 1))
    return cpus


def allocate_cpu_cores(cpus: set[int], index: int, count: int) -> set[int]:
    """Keep SMT siblings together, including non-contiguous CPU numbering."""
    if not 0 <= index < count:
        raise ValueError(f"Invalid local device index {index} / {count}")
    groups = {}
    for cpu in sorted(cpus):
        siblings = parse_cpu_list(
            Path(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ).read_text()
        )
        if cpu not in siblings:
            raise ValueError(f"CPU {cpu} missing from its SMT sibling list")
        groups.setdefault(tuple(sorted(siblings)), set()).add(cpu)
    cores = sorted(groups.values(), key=min)
    if len(cores) < count:
        raise ValueError(
            f"Only {len(cores)} allowed physical cores for {count} local devices"
        )
    start, stop = len(cores) * index // count, len(cores) * (index + 1) // count
    return set().union(*cores[start:stop])


def bind_cpu_affinity(provider, device_id: int, device_count: int, strict=False):
    """Bind within the inherited affinity; discovery failures leave it unchanged.

    Providers return a physical ID and a CPU set for each process-visible device.
    Partition cores among visible devices with the same locality, so independently
    started workers make the same allocation without cross-process coordination.
    No NUMA memory policy or additional capabilities are required.
    """
    allowed = os.sched_getaffinity(0)
    applied = False
    try:
        if not 0 <= device_id < device_count:
            raise ValueError(f"Invalid device index {device_id} / {device_count}")
        devices = [provider.device(i) for i in range(device_count)]
        physical_id, local_cpus = devices[device_id]
        if len({physical for physical, _ in devices}) != device_count:
            raise ValueError("Topology returned duplicate physical devices")
        peers = [i for i, (_, cpus) in enumerate(devices) if cpus == local_cpus]
        if any(cpus != local_cpus and cpus & local_cpus for _, cpus in devices):
            raise ValueError("NPU CPU localities partially overlap")
        candidates = allowed & local_cpus
        requested = allocate_cpu_cores(candidates, peers.index(device_id), len(peers))
        os.sched_setaffinity(0, requested)
        applied = True
        effective = os.sched_getaffinity(0)
        if effective != requested:
            raise RuntimeError(f"Affinity readback differs: {sorted(effective)}")
        logger.info(
            "%s CPU affinity: device=%s physical_device=%s numa_nodes=%s "
            "requested=%s allowed=%s effective=%s",
            provider.name,
            device_id,
            physical_id,
            provider.numa_nodes(local_cpus),
            sorted(requested),
            sorted(allowed),
            sorted(effective),
        )
        return True
    except (OSError, ValueError, RuntimeError) as exc:
        # A failed discovery never changes affinity. Restore it if a readback or
        # post-bind check fails; do not fall through to a host-wide flat split.
        if applied:
            try:
                os.sched_setaffinity(0, allowed)
            except OSError as restore_exc:
                logger.warning("Could not restore CPU affinity: %s", restore_exc)
        message = f"Skipping {provider.name} CPU affinity for device {device_id}: {exc}"
        if strict:
            raise RuntimeError(message) from exc
        logger.warning(message)
        return False
