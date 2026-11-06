"""Capture stable device and system software identity outside timed intervals."""

import hashlib
import os
import platform
from pathlib import Path


def version_record(paths):
    for path in paths:
        if path.is_file():
            try:
                data = path.read_bytes()
            except OSError:
                continue
            fields = dict(
                line.split("=", 1) for line in data.decode().splitlines() if "=" in line
            )
            version = fields.get("version") or fields.get("Version")
            if version:
                return dict(version=version, sha256=hashlib.sha256(data).hexdigest())
    return None


def hardware_metadata():
    """Keep missing identities explicit; summaries reject incomplete captures.

    Device UUIDs distinguish physical chips even across hosts and container
    restarts. Logical indices describe visibility; rank placement is captured
    separately after distributed initialization. No utilization or free-memory
    counters enter the identity because those vary during an experiment.
    """
    home = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    hardware = dict(
        machine=platform.machine(),
        device_nodes=sorted(p.name for p in Path("/dev").glob("davinci[0-9]*")),
        visible_devices={
            key: os.environ[key]
            for key in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
            if key in os.environ
        },
        cann=version_record(
            [
                home / f"{platform.machine()}-linux/ascend_toolkit_install.info",
                home / "version.cfg",
                home / "version.info",
            ]
        ),
        driver=version_record([Path("/usr/local/Ascend/driver/version.info")]),
        devices=[],
    )
    try:
        import torch
        import torch_npu  # noqa: F401

        for index in range(torch.npu.device_count()):
            props = torch.npu.get_device_properties(index)
            hardware["devices"].append(
                dict(
                    logical_id=index,
                    model=props.name,
                    uuid=str(props.uuid) if getattr(props, "uuid", None) else None,
                )
            )
    except (ImportError, RuntimeError) as exc:
        hardware["capture_error"] = f"{type(exc).__name__}: {exc}"
    return hardware


def rank_mapping(hardware):
    """All ranks must call this; never collect inside rank-zero-only writers."""
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    local = dict(
        rank=dist.get_rank(),
        local_rank=local_rank,
        device=hardware["devices"][local_rank],
    )
    placement = [None] * dist.get_world_size()
    dist.all_gather_object(placement, local)
    return placement


def hardware_identity(metadata):
    hardware = metadata.get("hardware")
    if not hardware or hardware.get("capture_error"):
        raise ValueError("missing hardware identity")
    for key in ("machine", "cann", "driver", "devices"):
        if not hardware.get(key):
            raise ValueError(f"missing hardware identity: {key}")
    for key in ("cann", "driver"):
        if not all(hardware[key].get(field) for field in ("version", "sha256")):
            raise ValueError(f"incomplete hardware identity: {key}")
    for device in hardware["devices"]:
        if not device.get("model") or not device.get("uuid"):
            raise ValueError("incomplete device model/UUID")
    return hardware
