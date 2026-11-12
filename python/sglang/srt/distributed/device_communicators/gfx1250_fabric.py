"""Whether a process group sits in one gfx1250 UALink/UALoE fabric clique.

A multi-node ``CustomAllReduceV2`` workspace on ROCm is shared through HIP
FABRIC handles, which only map between accelerators of one VPOD. The checks
here only read sysfs and a device attribute; ``HipSymmetricBuffer`` does the
allocation and mapping once every rank has passed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_LINK_TYPES = {"ualink", "ualoe"}
_READY_STATES = {"active", "ready"}
_UALINK_FIELDS = (
    "link_type",
    "accel_state",
    "accel_id",
    "ppod_id",
    "vpod_id",
    "vpod_size",
    "vpod_active_accels",
)


@dataclass(frozen=True)
class FabricCapability:
    rank: int
    hostname: str
    accel_id: Optional[int]
    ppod_id: Optional[str]
    vpod_id: Optional[str]
    vpod_size: Optional[int]
    reasons: tuple[str, ...]


def _parse_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value, 0) if value is not None else None
    except ValueError:
        return None


def _parse_int_list(value: Optional[str]) -> tuple[int, ...]:
    try:
        return tuple(int(item, 0) for item in (value or "").split())
    except ValueError:
        return ()


def _is_zero_uuid(value: str) -> bool:
    return not value.replace("-", "").strip("0")


def pci_function0_bus_id(properties) -> Optional[str]:
    """Every DPX/CPX partition of an accelerator reports the UALink record of
    PCI function 0."""
    if not all(
        hasattr(properties, name)
        for name in ("pci_domain_id", "pci_bus_id", "pci_device_id")
    ):
        return None
    return (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )


def read_ualink_sysfs(
    pci_bus_id: Optional[str], sysfs_root: Path = Path("/sys/class/drm")
) -> Optional[dict[str, Optional[str]]]:
    """The UALink attributes of the DRM card at ``pci_bus_id``, or None."""
    if pci_bus_id is None:
        return None
    for card in sorted(sysfs_root.glob("card[0-9]*")):
        try:
            bus_id = (card / "device").resolve().name
        except OSError:
            continue
        if bus_id.removeprefix("0000:").lower() != pci_bus_id.removeprefix("0000:"):
            continue
        values = {}
        for name in _UALINK_FIELDS:
            try:
                values[name] = (card / "device/ualink" / name).read_text().strip()
            except (OSError, UnicodeDecodeError):
                values[name] = None
        return values
    return None


def probe_local_fabric_capability(
    device: int,
    *,
    rank: int = 0,
    sysfs_root: Path = Path("/sys/class/drm"),
) -> FabricCapability:
    from sglang.srt.utils.hip_vmm_utils import is_fabric_handle_supported

    properties = torch.cuda.get_device_properties(device)
    arch = getattr(properties, "gcnArchName", "")
    ualink = read_ualink_sysfs(pci_function0_bus_id(properties), sysfs_root)
    fields = ualink or dict.fromkeys(_UALINK_FIELDS)
    accel_id = _parse_int(fields["accel_id"])
    active_accels = _parse_int_list(fields["vpod_active_accels"])

    reasons = []
    if "gfx1250" not in arch:
        reasons.append(f"device architecture is {arch or 'unknown'}, not gfx1250")
    if ualink is None:
        reasons.append("no UALink sysfs record matches the device")
    if (fields["link_type"] or "").lower() not in _LINK_TYPES:
        reasons.append(f"unsupported fabric link type: {fields['link_type']!r}")
    if (fields["accel_state"] or "").lower() not in _READY_STATES:
        reasons.append(f"fabric state is not active: {fields['accel_state']!r}")
    if not fields["ppod_id"] or _is_zero_uuid(fields["ppod_id"]):
        reasons.append(f"no PPOD UUID: {fields['ppod_id']!r}")
    if not fields["vpod_id"]:
        reasons.append("no VPOD ID")
    if _parse_int(fields["vpod_size"]) is None:
        reasons.append(f"no VPOD size: {fields['vpod_size']!r}")
    if accel_id is None:
        reasons.append("no accelerator ID")
    elif accel_id not in active_accels:
        reasons.append(f"accelerator {accel_id} is not active in its VPOD")
    if not is_fabric_handle_supported(device):
        reasons.append("device does not support FABRIC handles")

    return FabricCapability(
        rank=rank,
        hostname=os.uname().nodename,
        accel_id=accel_id,
        ppod_id=fields["ppod_id"],
        vpod_id=fields["vpod_id"],
        vpod_size=_parse_int(fields["vpod_size"]),
        reasons=tuple(reasons),
    )


def validate_fabric_clique(
    capabilities: Iterable[FabricCapability],
) -> tuple[bool, tuple[str, ...]]:
    capabilities = tuple(capabilities)
    if not capabilities:
        return False, ("no fabric capability records",)
    reasons = [
        f"rank {c.rank} ({c.hostname}): {reason}"
        for c in capabilities
        for reason in c.reasons
    ]
    for name in ("ppod_id", "vpod_id", "vpod_size"):
        values = {getattr(c, name) for c in capabilities}
        if len(values) != 1:
            reasons.append(f"ranks disagree on {name}: {sorted(values, key=str)}")
    # VPOD size counts physical accelerators, and DPX/CPX partitions of one
    # accelerator share its ID, so a group may hold more ranks than that
    accels = {c.accel_id for c in capabilities if c.accel_id is not None}
    vpod_size = capabilities[0].vpod_size
    if vpod_size is None or len(accels) > vpod_size:
        reasons.append(f"{len(accels)} accelerators exceed the VPOD size {vpod_size}")
    return not reasons, tuple(reasons)


def probe_fabric_clique(group: ProcessGroup, device: torch.device) -> bool:
    """Whether every rank of ``group`` is in one gfx1250 fabric clique.

    Collective: every rank probes its own device and all ranks reach the same
    answer, so none is left alone in the handle exchange.
    """
    rank = dist.get_rank(group=group)
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        local: object = probe_local_fabric_capability(index, rank=rank)
    except Exception as e:
        local = f"rank {rank}: fabric probe failed: {e!r}"
    capabilities: List[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(capabilities, local, group=group)
    failures = tuple(c for c in capabilities if isinstance(c, str))
    supported, reasons = (
        (False, failures) if failures else validate_fabric_clique(capabilities)
    )
    if not supported and rank == 0:
        logger.warning(
            "CustomAllReduceV2 has no fabric clique; falling back to RCCL: %s",
            reasons,
        )
    return supported
