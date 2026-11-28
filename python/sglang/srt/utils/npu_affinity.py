"""Topology-aware CPU affinity helpers for Ascend NPU scheduler processes.

This module intentionally does not import ``torch`` or ``torch_npu``.  CPU
assignment is planned before the accelerator runtime is initialized, so the
scheduler's ``gpu_id`` argument is the authoritative runtime NPU id.  The physical
NPU id used by ``npu-smi`` is resolved from the Ascend visibility environment.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import shutil
import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import psutil

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_NPU_ROW_RE = re.compile(r"^NPU(?P<id>\d+)$")
_NUMA_NODE_RE = re.compile(r"^node(?P<id>\d+)$")
_VALID_TOPOLOGY_TOKENS = frozenset({"X", "SYS", "PHB", "PIX", "PXB", "SIO", "UB", "NA"})
_TOPOLOGY_SOURCE = "npu-smi info -t topo"
_TOPOLOGY_LOG_LOCK = threading.Lock()
_LOGGED_TOPOLOGY_SCOPES: set[str] = set()


class NpuAffinityError(RuntimeError):
    """A recoverable NPU topology or CPU-affinity failure."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        logical_npu_id: Optional[int] = None,
        physical_npu_id: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.logical_npu_id = logical_npu_id
        self.physical_npu_id = physical_npu_id


@dataclass(frozen=True)
class PhysicalCore:
    socket_id: int
    core_id: int
    logical_cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class NpuTopologyEntry:
    physical_npu_id: int
    raw_cpu_affinity: str
    local_cpu_ids: frozenset[int]
    numa_node: int
    # Every core contains all of its online SMT siblings from global lscpu,
    # including siblings omitted by this entry's raw CPU affinity.
    physical_cores: tuple[PhysicalCore, ...]


@dataclass(frozen=True)
class NpuTopology:
    source: str
    entries: dict[int, NpuTopologyEntry]
    online_cpu_ids: frozenset[int]


@dataclass(frozen=True)
class NpuAffinityAssignment:
    # This is the runtime device id passed to run_scheduler_process as gpu_id.
    logical_npu_id: int
    physical_npu_id: int
    numa_node: int
    slot_index: int
    slots_on_node: int
    requested_pcores: int
    effective_pcores: int
    physical_core_keys: tuple[tuple[int, int], ...]
    logical_cpu_ids: tuple[int, ...]
    topology_source: str
    raw_cpu_affinity: str = ""
    local_cpu_ids: tuple[int, ...] = ()
    allowed_cpu_ids: tuple[int, ...] = ()
    owned_physical_core_keys: tuple[tuple[int, int], ...] = ()
    group_physical_npu_ids: tuple[int, ...] = ()
    ownership_start: int = 0
    ownership_end: int = 0
    visible_devices: str = "unset"

    @property
    def runtime_npu_id(self) -> int:
        return self.logical_npu_id


@dataclass(frozen=True)
class NpuPdAffinityAssignment:
    physical_npu_id: int
    physical_core_keys: tuple[tuple[int, int], ...]
    logical_cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class NpuAffinityThreadResult:
    thread_id: Optional[int]
    status: Literal["bound", "exited", "failed", "mismatched"]
    actual_cpu_ids: tuple[int, ...] = ()
    error: str = ""
    expected_cpu_ids: tuple[int, ...] = ()
    role: str = "compute"
    source: str = "default"
    name: str = ""
    start_time: Optional[int] = None


@dataclass(frozen=True)
class NpuAffinityApplyResult:
    requested_cpu_ids: tuple[int, ...]
    main_actual_cpu_ids: tuple[int, ...]
    main_matched: bool
    threads_total: int
    threads_bound: int
    threads_exited: int
    threads_failed: int
    threads_mismatched: int
    thread_results: tuple[NpuAffinityThreadResult, ...] = ()
    bind_all_threads: bool = False
    pd_assignment: Optional[NpuPdAffinityAssignment] = None

    @property
    def success(self) -> bool:
        return (
            self.main_matched
            and self.threads_failed == 0
            and self.threads_mismatched == 0
        )


def parse_cpu_list(value: str) -> frozenset[int]:
    """Parse a Linux CPU-list string such as ``0-7,16,32-39``."""

    if value is None or not value.strip():
        raise ValueError("CPU list must not be empty")

    cpu_ids: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Invalid empty token in CPU list {value!r}")
        if "-" in token:
            if token.count("-") != 1:
                raise ValueError(f"Invalid CPU range {token!r}")
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid CPU range {token!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"CPU range end precedes start in {token!r}")
            cpu_ids.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid CPU id {token!r}")
            cpu_ids.add(int(token))

    if not cpu_ids:
        raise ValueError("CPU list must contain at least one CPU")
    return frozenset(cpu_ids)


def format_cpu_list(cpu_ids: Iterable[int]) -> str:
    """Format CPU ids using Linux's compact range notation."""

    values = sorted(set(cpu_ids))
    if not values:
        return ""
    if values[0] < 0:
        raise ValueError("CPU ids must be non-negative")

    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _parse_npu_smi_topology_output(
    output: str,
) -> dict[int, tuple[str, frozenset[int]]]:
    """Parse validated NPU rows from ``npu-smi info -t topo`` output."""

    header_npu_ids: Optional[tuple[int, ...]] = None
    parsed: dict[int, tuple[str, frozenset[int]]] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()

        if len(tokens) >= 3 and tokens[-2:] == ["CPU", "Affinity"]:
            header_ids: list[int] = []
            for token in tokens[:-2]:
                match = _NPU_ROW_RE.fullmatch(token)
                if match is None:
                    raise NpuAffinityError(
                        f"Invalid npu-smi topology header token {token!r}",
                        stage="parse_npu_smi_header",
                    )
                header_ids.append(int(match.group("id")))
            if not header_ids or len(set(header_ids)) != len(header_ids):
                raise NpuAffinityError(
                    "npu-smi topology header has no NPU ids or contains duplicates",
                    stage="parse_npu_smi_header",
                )
            header_npu_ids = tuple(header_ids)
            continue

        first_match = _NPU_ROW_RE.fullmatch(tokens[0])
        if first_match is None:
            continue
        if len(tokens) < 3:
            raise NpuAffinityError(
                f"Malformed npu-smi topology row: {line!r}",
                stage="parse_npu_smi_row",
            )

        physical_npu_id = int(first_match.group("id"))
        if physical_npu_id in parsed:
            raise NpuAffinityError(
                f"Duplicate NPU id NPU{physical_npu_id} in npu-smi topology",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )

        topology_tokens = tokens[1:-1]
        invalid_tokens = [
            token for token in topology_tokens if token not in _VALID_TOPOLOGY_TOKENS
        ]
        if invalid_tokens:
            raise NpuAffinityError(
                f"Unsupported topology token(s) {invalid_tokens!r} in row {line!r}",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )
        if header_npu_ids is not None and len(topology_tokens) != len(header_npu_ids):
            raise NpuAffinityError(
                f"NPU{physical_npu_id} has {len(topology_tokens)} topology fields; "
                f"expected {len(header_npu_ids)}",
                stage="parse_npu_smi_row",
                physical_npu_id=physical_npu_id,
            )

        raw_cpu_affinity = tokens[-1]
        try:
            local_cpu_ids = parse_cpu_list(raw_cpu_affinity)
        except ValueError as exc:
            raise NpuAffinityError(
                f"Invalid CPU affinity {raw_cpu_affinity!r} for "
                f"NPU{physical_npu_id}: {exc}",
                stage="parse_npu_smi_cpu_affinity",
                physical_npu_id=physical_npu_id,
            ) from exc
        parsed[physical_npu_id] = (raw_cpu_affinity, local_cpu_ids)

    if not parsed:
        raise NpuAffinityError(
            "No NPU topology rows were found in npu-smi output",
            stage="parse_npu_smi_output",
        )
    if header_npu_ids is not None and set(parsed) != set(header_npu_ids):
        raise NpuAffinityError(
            "npu-smi topology rows do not match the NPU ids declared by the header: "
            f"header={list(header_npu_ids)}, rows={sorted(parsed)}",
            stage="parse_npu_smi_output",
        )
    return parsed


def read_online_cpu_ids() -> frozenset[int]:
    path = Path("/sys/devices/system/cpu/online")
    try:
        return parse_cpu_list(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise NpuAffinityError(
            f"Failed to read online CPUs from {path}: {exc}",
            stage="read_online_cpus",
        ) from exc


def read_numa_cpu_sets() -> dict[int, frozenset[int]]:
    result: dict[int, frozenset[int]] = {}
    root = Path("/sys/devices/system/node")
    try:
        node_paths = sorted(root.glob("node[0-9]*"))
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to enumerate NUMA nodes under {root}: {exc}",
            stage="read_numa_cpus",
        ) from exc

    for node_path in node_paths:
        match = _NUMA_NODE_RE.fullmatch(node_path.name)
        if match is None:
            continue
        cpulist_path = node_path / "cpulist"
        try:
            cpu_ids = parse_cpu_list(cpulist_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise NpuAffinityError(
                f"Failed to read NUMA CPU list from {cpulist_path}: {exc}",
                stage="read_numa_cpus",
            ) from exc
        result[int(match.group("id"))] = cpu_ids

    if not result:
        raise NpuAffinityError(
            f"No NUMA node CPU lists were found under {root}",
            stage="read_numa_cpus",
        )
    return result


def read_lscpu_core_topology() -> dict[int, tuple[int, int, int]]:
    """Return logical CPU -> (core id, socket id, NUMA node)."""

    try:
        proc = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NpuAffinityError(
            f"Failed to execute lscpu: {exc}", stage="query_lscpu"
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1000]
        raise NpuAffinityError(
            f"lscpu exited with code {proc.returncode}: {stderr}",
            stage="query_lscpu",
        )

    result: dict[int, tuple[int, int, int]] = {}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4 or any(not field.isdigit() for field in fields):
            raise NpuAffinityError(
                f"Malformed lscpu topology row {line!r}",
                stage="parse_lscpu",
            )
        cpu_id, core_id, socket_id, numa_node = map(int, fields)
        if cpu_id in result:
            raise NpuAffinityError(
                f"Duplicate logical CPU {cpu_id} in lscpu output",
                stage="parse_lscpu",
            )
        result[cpu_id] = (core_id, socket_id, numa_node)
    if not result:
        raise NpuAffinityError(
            "lscpu returned no CPU topology rows", stage="parse_lscpu"
        )
    return result


def _infer_unique_numa_node(
    local_cpu_ids: frozenset[int], numa_cpu_sets: dict[int, frozenset[int]]
) -> int:
    candidates = [
        node
        for node, node_cpu_ids in numa_cpu_sets.items()
        if local_cpu_ids <= node_cpu_ids
    ]
    if len(candidates) != 1:
        raise NpuAffinityError(
            "NPU CPU affinity does not map to exactly one NUMA node: "
            f"cpus={format_cpu_list(local_cpu_ids)}, candidates={sorted(candidates)}",
            stage="infer_numa_node",
        )
    return candidates[0]


def _build_npu_topology(
    parsed_rows: dict[int, tuple[str, frozenset[int]]],
    *,
    online_cpu_ids: frozenset[int],
    numa_cpu_sets: dict[int, frozenset[int]],
    lscpu_topology: dict[int, tuple[int, int, int]],
) -> NpuTopology:
    # Build sibling groups from the whole machine before inspecting an NPU's
    # local mask. Otherwise a mask containing only one of two online siblings
    # would incorrectly appear to describe a complete physical core.
    core_to_cpus: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cpu_id in sorted(online_cpu_ids):
        cpu_info = lscpu_topology.get(cpu_id)
        if cpu_info is None:
            raise NpuAffinityError(
                f"Online logical CPU {cpu_id} is missing from lscpu",
                stage="map_cpu_to_core",
            )
        core_id, socket_id, _ = cpu_info
        core_to_cpus[(socket_id, core_id)].append(cpu_id)

    entries: dict[int, NpuTopologyEntry] = {}
    for physical_npu_id, (raw_cpu_affinity, local_cpu_ids) in parsed_rows.items():
        offline_cpu_ids = local_cpu_ids - online_cpu_ids
        if offline_cpu_ids:
            raise NpuAffinityError(
                f"NPU{physical_npu_id} CPU affinity contains offline CPUs: "
                f"{format_cpu_list(offline_cpu_ids)}",
                stage="validate_online_cpus",
                physical_npu_id=physical_npu_id,
            )

        try:
            numa_node = _infer_unique_numa_node(local_cpu_ids, numa_cpu_sets)
        except NpuAffinityError as exc:
            exc.physical_npu_id = physical_npu_id
            raise

        local_core_keys: set[tuple[int, int]] = set()
        for cpu_id in sorted(local_cpu_ids):
            cpu_info = lscpu_topology.get(cpu_id)
            if cpu_info is None:
                raise NpuAffinityError(
                    f"Logical CPU {cpu_id} for NPU{physical_npu_id} is missing "
                    "from lscpu",
                    stage="map_cpu_to_core",
                    physical_npu_id=physical_npu_id,
                )
            core_id, socket_id, cpu_numa_node = cpu_info
            if cpu_numa_node != numa_node:
                raise NpuAffinityError(
                    f"Logical CPU {cpu_id} maps to NUMA {cpu_numa_node} in lscpu but "
                    f"NPU{physical_npu_id} affinity maps to NUMA {numa_node}",
                    stage="map_cpu_to_core",
                    physical_npu_id=physical_npu_id,
                )
            local_core_keys.add((socket_id, core_id))

        physical_cores = tuple(
            PhysicalCore(
                socket_id=socket_id,
                core_id=core_id,
                logical_cpu_ids=tuple(core_to_cpus[(socket_id, core_id)]),
            )
            for socket_id, core_id in sorted(local_core_keys)
        )
        entries[physical_npu_id] = NpuTopologyEntry(
            physical_npu_id=physical_npu_id,
            raw_cpu_affinity=raw_cpu_affinity,
            local_cpu_ids=local_cpu_ids,
            numa_node=numa_node,
            physical_cores=physical_cores,
        )

    return NpuTopology(
        source=_TOPOLOGY_SOURCE,
        entries=entries,
        online_cpu_ids=online_cpu_ids,
    )


@lru_cache(maxsize=1)
def query_npu_smi_topology() -> NpuTopology:
    try:
        proc = subprocess.run(
            ["npu-smi", "info", "-t", "topo"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NpuAffinityError(
            "npu-smi topology query timed out after 10 seconds",
            stage="query_npu_smi",
        ) from exc
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to execute npu-smi: {exc}", stage="query_npu_smi"
        ) from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:1000]
        raise NpuAffinityError(
            f"npu-smi topology query exited with code {proc.returncode}: {stderr}",
            stage="query_npu_smi",
        )

    parsed_rows = _parse_npu_smi_topology_output(proc.stdout)
    online_cpu_ids = read_online_cpu_ids()
    return _build_npu_topology(
        parsed_rows,
        online_cpu_ids=online_cpu_ids,
        numa_cpu_sets=read_numa_cpu_sets(),
        lscpu_topology=read_lscpu_core_topology(),
    )


def _get_visible_devices() -> tuple[Optional[str], Optional[str]]:
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return name, value.strip()
    return None, None


def resolve_physical_npu_id(logical_npu_id: int) -> int:
    if logical_npu_id < 0:
        raise NpuAffinityError(
            f"Runtime NPU id must be non-negative, got {logical_npu_id}",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )

    env_name, raw_devices = _get_visible_devices()
    if raw_devices is None:
        return logical_npu_id

    tokens = [token.strip() for token in raw_devices.split(",")]
    if any(not token or not token.isdigit() for token in tokens):
        raise NpuAffinityError(
            f"Invalid {env_name}={raw_devices!r}; expected comma-separated "
            "physical NPU ids",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    physical_ids = [int(token) for token in tokens]
    if len(set(physical_ids)) != len(physical_ids):
        raise NpuAffinityError(
            f"Invalid {env_name}={raw_devices!r}; physical NPU ids must be unique",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    if logical_npu_id >= len(physical_ids):
        raise NpuAffinityError(
            f"Runtime NPU id {logical_npu_id} is outside {env_name}={raw_devices!r}",
            stage="resolve_visible_device",
            logical_npu_id=logical_npu_id,
        )
    return physical_ids[logical_npu_id]


def _get_allowed_cpu_ids() -> frozenset[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise NpuAffinityError(
            "os.sched_getaffinity is unavailable; NPU CPU binding requires Linux",
            stage="read_process_cpuset",
        )
    try:
        return frozenset(os.sched_getaffinity(0))
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to read current process CPU affinity: {exc}",
            stage="read_process_cpuset",
        ) from exc


def _log_topology_once(
    topology: NpuTopology, allowed_cpu_ids: frozenset[int], *, scope: str
) -> None:
    with _TOPOLOGY_LOG_LOCK:
        if scope in _LOGGED_TOPOLOGY_SCOPES:
            return
        _LOGGED_TOPOLOGY_SCOPES.add(scope)

    lines = [
        "=============== NPU CPU AFFINITY TOPOLOGY ===============",
        f"scope={scope} source={topology.source!r}",
        f"physical_npu_count={len(topology.entries)}",
        f"online_cpus={format_cpu_list(topology.online_cpu_ids)}",
        f"allowed_cpus_before_binding={format_cpu_list(allowed_cpu_ids)}",
        "--------------- Physical NPU CPU affinity ---------------",
    ]
    for physical_npu_id, entry in sorted(topology.entries.items()):
        lines.append(
            f"physical_npu=NPU{physical_npu_id} numa={entry.numa_node}\n"
            f"  raw_cpu_affinity={entry.raw_cpu_affinity}\n"
            f"  physical_cores={len(entry.physical_cores)} "
            f"logical_cpus={len(entry.local_cpu_ids)}"
        )

    groups: dict[tuple[int, frozenset[int]], list[int]] = defaultdict(list)
    for physical_npu_id, entry in topology.entries.items():
        groups[(entry.numa_node, entry.local_cpu_ids)].append(physical_npu_id)
    lines.append("--------------- Fixed ownership groups ----------------")
    for (numa_node, local_cpu_ids), physical_ids in sorted(
        groups.items(), key=lambda item: (item[0][0], tuple(sorted(item[0][1])))
    ):
        lines.append(
            f"NPU locality group: numa={numa_node} npus={sorted(physical_ids)} "
            f"local_cpus={format_cpu_list(local_cpu_ids)}"
        )
    lines.append("=============== END NPU CPU AFFINITY TOPOLOGY ===============")
    logger.info("\n".join(lines))


def _build_affinity_assignment(
    topology: NpuTopology,
    *,
    logical_npu_id: int,
    physical_npu_id: int,
    allowed_cpu_ids: frozenset[int],
    requested_pcores: int,
) -> NpuAffinityAssignment:
    if requested_pcores < 0:
        raise ValueError("SGLANG_NPU_AFFINITY_PCORES_PER_PROC must be >= 0")
    entry = topology.entries.get(physical_npu_id)
    if entry is None:
        raise NpuAffinityError(
            f"Physical NPU{physical_npu_id} is absent from npu-smi topology; "
            f"available={sorted(topology.entries)}",
            stage="select_npu_topology",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    # The complete npu-smi topology, including devices outside this process's
    # visible-device list, determines physical CPU ownership. Neither the
    # process cpuset nor its requested CPU budget can move these boundaries.
    group_entries = sorted(
        (
            candidate
            for candidate in topology.entries.values()
            if candidate.numa_node == entry.numa_node
            and candidate.local_cpu_ids == entry.local_cpu_ids
        ),
        key=lambda candidate: candidate.physical_npu_id,
    )
    group_ids = [candidate.physical_npu_id for candidate in group_entries]
    try:
        slot_index = group_ids.index(physical_npu_id)
    except ValueError as exc:
        raise NpuAffinityError(
            f"NPU{physical_npu_id} is missing from its locality group",
            stage="assign_locality_slot",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        ) from exc

    slots_on_node = len(group_entries)
    all_cores = sorted(
        entry.physical_cores, key=lambda core: (core.socket_id, core.core_id)
    )
    start = slot_index * len(all_cores) // slots_on_node
    end = (slot_index + 1) * len(all_cores) // slots_on_node
    owned_cores = all_cores[start:end]
    if not owned_cores:
        raise NpuAffinityError(
            f"NPU{physical_npu_id} has an empty fixed ownership slice [{start}:{end}] "
            f"of {len(all_cores)} physical cores shared by {slots_on_node} NPUs "
            f"in NUMA node {entry.numa_node}",
            stage="assign_locality_slot",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    candidate_cpu_ids = entry.local_cpu_ids & topology.online_cpu_ids & allowed_cpu_ids
    eligible_cores = [
        core
        for core in owned_cores
        if core.logical_cpu_ids and set(core.logical_cpu_ids) <= candidate_cpu_ids
    ]
    if not eligible_cores:
        owned_cpu_ids = (
            cpu_id for core in owned_cores for cpu_id in core.logical_cpu_ids
        )
        raise NpuAffinityError(
            f"NPU{physical_npu_id} has no complete online SMT core in its fixed "
            f"ownership slice [{start}:{end}]: "
            f"owned={format_cpu_list(owned_cpu_ids)}, "
            f"local={format_cpu_list(entry.local_cpu_ids)}, "
            f"allowed={format_cpu_list(allowed_cpu_ids)}; "
            "all online siblings of an owned core must be local and allowed",
            stage="filter_process_cpuset",
            logical_npu_id=logical_npu_id,
            physical_npu_id=physical_npu_id,
        )

    if requested_pcores == 0:
        effective_pcores = len(eligible_cores)
    else:
        effective_pcores = min(requested_pcores, len(eligible_cores))
        if requested_pcores > len(eligible_cores):
            logger.warning(
                "SGLANG_NPU_AFFINITY_PCORES_PER_PROC=%s exceeds %s available "
                "complete SMT cores in NPU%s fixed ownership slice [%s:%s]; "
                "clipping to %s without changing ownership boundaries",
                requested_pcores,
                len(eligible_cores),
                physical_npu_id,
                start,
                end,
                effective_pcores,
            )

    selected_cores = eligible_cores[:effective_pcores]

    logical_cpu_ids = tuple(
        sorted(cpu_id for core in selected_cores for cpu_id in core.logical_cpu_ids)
    )
    env_name, visible_devices = _get_visible_devices()
    return NpuAffinityAssignment(
        logical_npu_id=logical_npu_id,
        physical_npu_id=physical_npu_id,
        numa_node=entry.numa_node,
        slot_index=slot_index,
        slots_on_node=slots_on_node,
        requested_pcores=requested_pcores,
        effective_pcores=effective_pcores,
        physical_core_keys=tuple(
            (core.socket_id, core.core_id) for core in selected_cores
        ),
        logical_cpu_ids=logical_cpu_ids,
        topology_source=topology.source,
        raw_cpu_affinity=entry.raw_cpu_affinity,
        local_cpu_ids=tuple(sorted(entry.local_cpu_ids)),
        allowed_cpu_ids=tuple(sorted(allowed_cpu_ids)),
        owned_physical_core_keys=tuple(
            (core.socket_id, core.core_id) for core in owned_cores
        ),
        group_physical_npu_ids=tuple(group_ids),
        ownership_start=start,
        ownership_end=end,
        visible_devices=f"{env_name}={visible_devices}" if env_name else "unset",
    )


def resolve_npu_affinity_assignment(
    *,
    logical_npu_id: int,
    emit_topology_log: bool,
) -> NpuAffinityAssignment:
    requested_pcores = envs.SGLANG_NPU_AFFINITY_PCORES_PER_PROC.get()
    if requested_pcores < 0:
        raise ValueError("SGLANG_NPU_AFFINITY_PCORES_PER_PROC must be >= 0")

    physical_npu_id = resolve_physical_npu_id(logical_npu_id)
    try:
        topology = query_npu_smi_topology()
        allowed_cpu_ids = _get_allowed_cpu_ids()
    except NpuAffinityError as exc:
        if exc.logical_npu_id is None:
            exc.logical_npu_id = logical_npu_id
        if exc.physical_npu_id is None:
            exc.physical_npu_id = physical_npu_id
        raise
    if emit_topology_log:
        _log_topology_once(topology, allowed_cpu_ids, scope="launcher")

    return _build_affinity_assignment(
        topology,
        logical_npu_id=logical_npu_id,
        physical_npu_id=physical_npu_id,
        allowed_cpu_ids=allowed_cpu_ids,
        requested_pcores=requested_pcores,
    )


def build_npu_pd_affinity_assignment(
    topology: NpuTopology,
    compute: NpuAffinityAssignment,
    requested_pcores: int,
) -> NpuPdAffinityAssignment:
    """Use only unused, complete cores from the compute plan's saved slice."""
    if requested_pcores <= 0:
        raise NpuAffinityError(
            "PD core budget must be positive", stage="plan_pd_affinity"
        )
    entry = topology.entries[compute.physical_npu_id]
    owned_keys = set(compute.owned_physical_core_keys)
    compute_keys = set(compute.physical_core_keys)
    unused_keys = owned_keys - compute_keys
    allowed = (
        entry.local_cpu_ids & topology.online_cpu_ids & set(compute.allowed_cpu_ids)
    )
    remaining = [
        core
        for core in sorted(entry.physical_cores, key=lambda c: (c.socket_id, c.core_id))
        if (core.socket_id, core.core_id) in unused_keys
        and core.logical_cpu_ids
        and set(core.logical_cpu_ids) <= allowed
    ]
    if len(remaining) < requested_pcores:
        raise NpuAffinityError(
            f"NPU{compute.physical_npu_id} cannot allocate {requested_pcores} extra "
            f"PD physical cores: compute={compute.effective_pcores}, "
            f"remaining={len(remaining)} in its fixed ownership slice; "
            "the compute assignment will not be reduced",
            stage="plan_pd_affinity",
            logical_npu_id=compute.logical_npu_id,
            physical_npu_id=compute.physical_npu_id,
        )
    selected = remaining[:requested_pcores]
    return NpuPdAffinityAssignment(
        physical_npu_id=compute.physical_npu_id,
        physical_core_keys=tuple((c.socket_id, c.core_id) for c in selected),
        logical_cpu_ids=tuple(
            sorted(cpu for core in selected for cpu in core.logical_cpu_ids)
        ),
    )


def apply_npu_cpu_affinity(
    assignment: NpuAffinityAssignment,
    *,
    phase: Literal["early", "final"],
    bind_all_threads: bool,
    emit_log: bool = True,
    pid: Optional[int] = None,
    thread_binder: Optional[Callable[[int, int], NpuAffinityThreadResult]] = None,
    pd_assignment: Optional[NpuPdAffinityAssignment] = None,
) -> NpuAffinityApplyResult:
    """Apply the saved target exactly and verify every surviving thread.

    ``assignment`` must be resolved before initialization and reused for final
    binding. A runtime may narrow the main thread's mask in between; intersecting
    with that mask here would prevent restoring the originally selected CPUs.
    ``pid`` allows the HTTP launcher to bind a scheduler after HTTP warmup.
    An optional ``thread_binder`` applies and verifies per-thread targets for
    PD. With no binder the original uniform compute binding is unchanged.
    """

    if phase not in ("early", "final"):
        raise ValueError(f"Unsupported NPU affinity phase {phase!r}")
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise NpuAffinityError(
            "Linux sched affinity APIs are unavailable",
            stage="apply_cpu_affinity",
            logical_npu_id=assignment.logical_npu_id,
            physical_npu_id=assignment.physical_npu_id,
        )

    target_cpu_ids = set(assignment.logical_cpu_ids)
    if not target_cpu_ids:
        raise NpuAffinityError(
            "Saved NPU affinity target is empty",
            stage="apply_cpu_affinity",
            logical_npu_id=assignment.logical_npu_id,
            physical_npu_id=assignment.physical_npu_id,
        )
    process_pid = os.getpid() if pid is None else pid
    main_tid = 0 if pid is None else pid
    try:
        os.sched_setaffinity(main_tid, target_cpu_ids)
        main_actual_cpu_ids = tuple(sorted(os.sched_getaffinity(main_tid)))
    except OSError as exc:
        raise NpuAffinityError(
            f"Failed to bind scheduler main thread: {exc}",
            stage="apply_cpu_affinity",
            logical_npu_id=assignment.logical_npu_id,
            physical_npu_id=assignment.physical_npu_id,
        ) from exc

    target_tuple = tuple(sorted(target_cpu_ids))
    main_matched = main_actual_cpu_ids == target_tuple
    threads_total = 0
    threads_bound = 0
    threads_exited = 0
    threads_failed = 0
    threads_mismatched = 0
    thread_results: list[NpuAffinityThreadResult] = []
    if bind_all_threads:
        try:
            threads = psutil.Process(process_pid).threads()
        except (psutil.Error, OSError) as exc:
            threads = []
            threads_failed += 1
            thread_results.append(
                NpuAffinityThreadResult(
                    thread_id=None,
                    status="failed",
                    error=f"Failed to enumerate scheduler threads: {exc}",
                )
            )
        threads_total = len(threads)
        for thread in threads:
            tid = thread.id
            try:
                if thread_binder is not None:
                    thread_result = thread_binder(process_pid, tid)
                    thread_results.append(thread_result)
                    threads_bound += thread_result.status in ("bound", "mismatched")
                    threads_mismatched += thread_result.status == "mismatched"
                    threads_failed += thread_result.status == "failed"
                    threads_exited += thread_result.status == "exited"
                    continue
                os.sched_setaffinity(tid, target_cpu_ids)
                actual_cpu_ids = tuple(sorted(os.sched_getaffinity(tid)))
                threads_bound += 1
                if actual_cpu_ids != target_tuple:
                    threads_mismatched += 1
                    thread_results.append(
                        NpuAffinityThreadResult(tid, "mismatched", actual_cpu_ids)
                    )
                else:
                    thread_results.append(
                        NpuAffinityThreadResult(tid, "bound", actual_cpu_ids)
                    )
            except ProcessLookupError:
                threads_exited += 1
                thread_results.append(NpuAffinityThreadResult(tid, "exited"))
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    threads_exited += 1
                    thread_results.append(NpuAffinityThreadResult(tid, "exited"))
                else:
                    threads_failed += 1
                    thread_results.append(
                        NpuAffinityThreadResult(tid, "failed", error=str(exc))
                    )

    result = NpuAffinityApplyResult(
        requested_cpu_ids=target_tuple,
        main_actual_cpu_ids=main_actual_cpu_ids,
        main_matched=main_matched,
        threads_total=threads_total,
        threads_bound=threads_bound,
        threads_exited=threads_exited,
        threads_failed=threads_failed,
        threads_mismatched=threads_mismatched,
        thread_results=tuple(thread_results),
        bind_all_threads=bind_all_threads,
        pd_assignment=pd_assignment,
    )
    if emit_log:
        log_npu_affinity_result(assignment, result, phase=phase, pid=pid)
    return result


def log_npu_affinity_result(
    assignment: NpuAffinityAssignment,
    result: NpuAffinityApplyResult,
    phase: Literal["early", "final"],
    *,
    pid: Optional[int] = None,
) -> None:
    """Log binding results, with an optional final thread snapshot for debugging."""

    if phase not in ("early", "final"):
        raise ValueError(f"Unsupported NPU affinity phase {phase!r}")
    status = "SUCCESS" if result.success else "INCOMPLETE"
    process_pid = os.getpid() if pid is None else pid
    raw_affinity = assignment.raw_cpu_affinity or format_cpu_list(
        assignment.local_cpu_ids
    )
    binding = (
        f"main_cpu_mask={format_cpu_list(result.main_actual_cpu_ids)} "
        f"assigned_cores={assignment.effective_pcores}P/{len(result.requested_cpu_ids)}L"
    )
    if result.bind_all_threads:
        binding += (
            f" threads_matched={result.threads_bound - result.threads_mismatched}"
            f"/{result.threads_total} threads_failed={result.threads_failed}"
            f" threads_mismatched={result.threads_mismatched}"
            f" threads_exited={result.threads_exited}"
        )
    else:
        binding += " threads=main-only"
    if not result.main_matched:
        binding += " main_matched=False expected_cpu_mask=" + format_cpu_list(
            result.requested_cpu_ids
        )
    if 0 < assignment.effective_pcores < assignment.requested_pcores:
        binding += f" requested_pcores={assignment.requested_pcores} (clipped)"
    if result.pd_assignment is not None:
        pd = result.pd_assignment
        overlap = set(assignment.physical_core_keys) & set(pd.physical_core_keys)
        binding += (
            f"\npd_cpu_mask={format_cpu_list(pd.logical_cpu_ids)} "
            f"pd_physical_cores={len(pd.physical_core_keys)} "
            f"compute_pd_core_overlap={len(overlap)}"
        )
        for source in ("python_pd", "native_pd", "default"):
            matched = sum(
                t.source == source and t.status == "bound"
                for t in result.thread_results
            )
            binding += f" {source}_threads={matched}"
    lines = [
        "=============== NPU CPU AFFINITY RESULT ===============",
        f"phase={phase} status={status} pid={process_pid} "
        f"runtime_npu={assignment.runtime_npu_id} "
        f"physical_npu={assignment.physical_npu_id} numa={assignment.numa_node} "
        f"raw_cpu_affinity={raw_affinity}",
        binding,
    ]
    if (
        phase == "final"
        and result.bind_all_threads
        and envs.SGLANG_NPU_AFFINITY_DEBUG_THREADS.get()
    ):
        # Reuse the PD binder's identity snapshot when available. Otherwise
        # inspect /proc names only in debug mode (native threads aren't in
        # threading.enumerate()).
        python_names = (
            {t.native_id: t.name for t in threading.enumerate()}
            if process_pid == os.getpid()
            else {}
        )
        lines.append("thread_details (final binding snapshot):")
        for thread in result.thread_results:
            tid = thread.thread_id
            name = thread.name or "<unavailable>"
            if tid is not None and not thread.name:
                try:
                    name = (
                        Path(f"/proc/{process_pid}/task/{tid}/comm")
                        .read_text(encoding="utf-8", errors="replace")
                        .rstrip("\n")
                    )
                except OSError:
                    # A short-lived runtime thread may exit after binding.
                    pass
            detail = (
                f"  tid={tid} name={name!r} main={tid == process_pid}"
                f" cpu_mask={format_cpu_list(thread.actual_cpu_ids) or 'unavailable'}"
                f" status={thread.status}"
            )
            if tid in python_names:
                detail += f" python_name={python_names[tid]!r}"
            if result.pd_assignment is not None:
                detail += (
                    f" role={thread.role} source={thread.source}"
                    f" starttime={thread.start_time}"
                    f" expected_cpu_mask={format_cpu_list(thread.expected_cpu_ids)}"
                )
            if thread.error:
                detail += f" error={thread.error!r}"
            lines.append(detail)
    lines.append("=============== END NPU CPU AFFINITY RESULT ===============")
    log = logger.info if result.success else logger.warning
    log("\n" + "\n".join(lines) + "\n")


def build_npu_affinity_report(
    assignment: Optional[NpuAffinityAssignment],
    result: Optional[NpuAffinityApplyResult],
    *,
    runtime_npu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    error: Optional[NpuAffinityError] = None,
    pid: Optional[int] = None,
) -> dict:
    """Attach the actual final read-back to the existing scheduler ready message."""
    if error is not None or result is None:
        status = "FAILED"
    elif result.success and result.bind_all_threads:
        status = "SUCCESS"
    else:
        status = "INCOMPLETE"
    report = {
        "pid": os.getpid() if pid is None else pid,
        "runtime_npu_id": runtime_npu_id,
        "physical_npu_id": (
            assignment.physical_npu_id
            if assignment is not None
            else getattr(error, "physical_npu_id", None)
        ),
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": dp_rank,
        "numa_node": assignment.numa_node if assignment is not None else None,
        "raw_cpu_affinity": (
            assignment.raw_cpu_affinity if assignment is not None else None
        ),
        "expected_cpu_ids": (
            list(assignment.logical_cpu_ids) if assignment is not None else None
        ),
        "actual_cpu_ids": (
            list(result.main_actual_cpu_ids) if result is not None else None
        ),
        "effective_pcores": (
            assignment.effective_pcores if assignment is not None else None
        ),
        "status": status,
        "threads_total": result.threads_total if result is not None else None,
        "threads_bound": result.threads_bound if result is not None else None,
        "threads_failed": result.threads_failed if result is not None else None,
        "threads_mismatched": (
            result.threads_mismatched if result is not None else None
        ),
        "threads_exited": result.threads_exited if result is not None else None,
        "thread_issues": (
            [
                {
                    "tid": thread.thread_id,
                    "status": thread.status,
                    "actual_cpu_ids": list(thread.actual_cpu_ids),
                    "error": thread.error,
                }
                for thread in result.thread_results
                if thread.status in ("failed", "mismatched")
            ]
            if result is not None
            else []
        ),
        "error": str(error) if error is not None else None,
    }
    if result is not None and result.pd_assignment is not None:
        pd = result.pd_assignment
        report["pd_cpu_ids"] = list(pd.logical_cpu_ids)
        report["pd_physical_cores"] = len(pd.physical_core_keys)
        report["pd_threads_bound"] = sum(
            t.source in ("python_pd", "native_pd") and t.status == "bound"
            for t in result.thread_results
        )
        issues = {t.thread_id: t for t in result.thread_results}
        for issue in report["thread_issues"]:
            thread = issues[issue["tid"]]
            issue.update(
                expected_cpu_ids=list(thread.expected_cpu_ids),
                role=thread.role,
                source=thread.source,
            )
    return report


def migrate_npu_pages(pid: int, numa_node: int) -> None:
    """Best-effort, opt-in migration of existing host pages; no memory policy."""
    if not envs.SGLANG_NPU_MIGRATE_PAGES.get():
        return
    executable = shutil.which("migratepages")
    if executable is None:
        logger.warning("NPU page migration skipped: migratepages is not installed")
        return
    try:
        result = subprocess.run(
            [executable, str(pid), "all", str(numa_node)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        # A successful command does not guarantee that pinned/shared pages moved.
        log = logger.info if result.returncode == 0 else logger.warning
        log(
            "NPU page migration: pid=%s numa=%s exit_code=%s output=%s",
            pid,
            numa_node,
            result.returncode,
            ((result.stdout or "") + (result.stderr or "")).strip()[:2000] or "none",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "NPU page migration failed: pid=%s numa=%s: %s", pid, numa_node, exc
        )


def finalize_npu_cpu_affinity(plan: dict) -> dict:
    """Apply a scheduler's saved plan after HTTP warmup, then optionally migrate."""
    assignment = plan["assignment"]
    pid = plan["pid"]
    result = None
    error = None
    try:
        # Do not apply a stale plan to an unrelated process after PID reuse.
        if psutil.Process(pid).create_time() != plan["create_time"]:
            raise NpuAffinityError(
                f"Scheduler PID {pid} was reused", stage="finalize_scheduler"
            )
        binding_options = {}
        if plan.get("pd_assignment") is not None:
            from sglang.srt.utils.npu_pd_affinity import make_pd_thread_binder

            binding_options = {
                "pd_assignment": plan["pd_assignment"],
                "thread_binder": make_pd_thread_binder(
                    assignment, plan["pd_assignment"], plan["python_pd_threads"]
                ),
            }
        result = apply_npu_cpu_affinity(
            assignment, phase="final", bind_all_threads=True, pid=pid, **binding_options
        )
        if result.success:
            migrate_npu_pages(pid, assignment.numa_node)
    except (NpuAffinityError, psutil.Error) as exc:
        error = (
            exc
            if isinstance(exc, NpuAffinityError)
            else NpuAffinityError(str(exc), stage="finalize_scheduler")
        )
        logger.warning("NPU final CPU binding failed: pid=%s: %s", pid, error)
    return build_npu_affinity_report(
        assignment,
        result,
        runtime_npu_id=assignment.runtime_npu_id,
        tp_rank=plan["tp_rank"],
        pp_rank=plan["pp_rank"],
        dp_rank=plan["dp_rank"],
        error=error,
        pid=pid,
    )


def log_npu_affinity_summary(
    scheduler_infos: list[dict], *, base_gpu_id: int, tp_size: int, port: int
) -> None:
    """Print one TP-group overview from child reports, then consume the metadata.

    Do not infer other schedulers' masks from their ranks or from topology: the
    ready pipes carry their own final read-backs, including failed bindings.
    """
    reports = [
        report
        for info in scheduler_infos
        if (report := info.pop("npu_cpu_affinity", None)) is not None
    ]
    if not reports:
        return
    reports.sort(
        key=lambda report: (
            report["physical_npu_id"] is None,
            report["physical_npu_id"] if report["physical_npu_id"] is not None else -1,
            report["runtime_npu_id"],
        )
    )

    def cpu_mask(cpu_ids):
        return format_cpu_list(cpu_ids) if cpu_ids is not None else "unavailable"

    lines = [
        "=============== NPU CPU AFFINITY SUMMARY ===============",
        f"instance: base_gpu_id={base_gpu_id} tp_size={tp_size} port={port}",
        f"local_schedulers={len(reports)}",
        "Each card below reports its own FINAL binding read-back.",
    ]
    for report in reports:
        physical_id = report["physical_npu_id"]
        lines.extend(
            [
                "",
                "--------------- Physical NPU "
                f"{physical_id if physical_id is not None else 'unknown'} ---------------",
                f"runtime_npu={report['runtime_npu_id']} pid={report['pid']} "
                f"numa={report['numa_node']}",
                f"tp_rank={report['tp_rank']} pp_rank={report['pp_rank']} "
                f"dp_rank={report['dp_rank']} status={report['status']}",
                "npu-smi raw_cpu_affinity="
                + (report["raw_cpu_affinity"] or "unavailable"),
                "expected_cpu_mask=" + cpu_mask(report["expected_cpu_ids"]),
                "actual_main_cpu_mask=" + cpu_mask(report["actual_cpu_ids"]),
                f"assigned_physical_cores={report['effective_pcores']}",
                "actual_main_logical_cpus="
                + (
                    str(len(report["actual_cpu_ids"]))
                    if report["actual_cpu_ids"] is not None
                    else "unavailable"
                ),
                f"threads_total={report['threads_total']} "
                f"threads_bound={report['threads_bound']}",
                f"threads_failed={report['threads_failed']} "
                f"threads_mismatched={report['threads_mismatched']} "
                f"threads_exited={report['threads_exited']}",
            ]
        )
        if "pd_cpu_ids" in report:
            lines.append(
                f"pd_cpu_mask={cpu_mask(report['pd_cpu_ids'])} "
                f"pd_physical_cores={report['pd_physical_cores']} "
                f"pd_threads_bound={report['pd_threads_bound']}"
            )
        for issue in report["thread_issues"]:
            lines.extend(
                [
                    f"  tid={issue['tid']} status={issue['status']}",
                    "  actual_cpu_mask=" + cpu_mask(issue["actual_cpu_ids"]),
                ]
            )
            if issue["error"]:
                lines.append(f"  error={issue['error']}")
            if "expected_cpu_ids" in issue:
                lines.append(
                    f"  role={issue['role']} source={issue['source']} "
                    f"expected_cpu_mask={cpu_mask(issue['expected_cpu_ids'])}"
                )
        if report["error"]:
            lines.append(f"error={report['error']}")
    lines.append("=============== END NPU CPU AFFINITY SUMMARY ===============")
    log = (
        logger.info
        if all(report["status"] == "SUCCESS" for report in reports)
        else logger.warning
    )
    log("\n" + "\n".join(lines) + "\n")


def format_npu_affinity_error(
    exc: NpuAffinityError,
    *,
    fallback: Literal["defer_to_final", "skip_cpu_bind", "skip_memory_bind"],
) -> str:
    return (
        f"topology_source={_TOPOLOGY_SOURCE!r} "
        f"runtime_npu={exc.logical_npu_id} physical_npu={exc.physical_npu_id} "
        f"failure_stage={exc.stage} reason={exc} fallback={fallback}"
    )
