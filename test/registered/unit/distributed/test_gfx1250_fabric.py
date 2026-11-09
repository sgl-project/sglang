from pathlib import Path
from unittest.mock import Mock

import pytest

from sglang.srt.distributed.device_communicators import gfx1250_fabric
from sglang.srt.distributed.device_communicators.gfx1250_fabric import (
    FabricCapability,
    pci_function0_bus_id,
    read_ualink_sysfs,
    validate_fabric_clique,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _capability(rank: int, **overrides) -> FabricCapability:
    fields = dict(
        rank=rank,
        hostname=f"host-{rank // 8}",
        accel_id=rank,
        ppod_id="5ddd19b0-c9cd-49cc-b55a-ea08459ed150",
        vpod_id="1",
        vpod_size=8,
        reasons=(),
    )
    fields.update(overrides)
    return FabricCapability(**fields)


def test_read_ualink_sysfs(tmp_path: Path):
    ualink = tmp_path / "devices" / "0001:04:00.0" / "ualink"
    ualink.mkdir(parents=True)
    values = {
        "link_type": "UALoE",
        "accel_state": "active",
        "accel_id": "52",
        "ppod_id": "uuid",
        "vpod_id": "1",
        "vpod_size": "60",
        "vpod_active_accels": "0 1 52 53",
    }
    for name, value in values.items():
        (ualink / name).write_text(value, encoding="utf-8")
    (tmp_path / "card0").mkdir()
    (tmp_path / "card0" / "device").symlink_to(ualink.parent, target_is_directory=True)

    assert read_ualink_sysfs("0001:04:00.0", tmp_path) == values
    assert read_ualink_sysfs("0002:04:00.0", tmp_path) is None


def test_pci_function0_bus_id():
    properties = Mock(pci_domain_id=1, pci_bus_id=4, pci_device_id=0)
    assert pci_function0_bus_id(properties) == "0001:04:00.0"


def test_validate_fabric_clique():
    assert validate_fabric_clique([_capability(0), _capability(1)]) == (True, ())

    supported, reasons = validate_fabric_clique(
        [_capability(0), _capability(1, vpod_id="3")]
    )
    assert not supported
    assert any("vpod_id" in reason for reason in reasons)

    supported, reasons = validate_fabric_clique(
        [_capability(0), _capability(1, reasons=("no VPOD ID",))]
    )
    assert not supported
    assert "rank 1 (host-0): no VPOD ID" in reasons

    supported, reasons = validate_fabric_clique(
        [_capability(0), _capability(1, vpod_size=16)]
    )
    assert not supported
    assert any("vpod_size" in reason for reason in reasons)


def test_validate_fabric_clique_counts_dpx_partitions_once():
    # TP16 over two DPX hosts: 16 ranks on 8 physical accelerators
    dpx = [_capability(rank, accel_id=rank // 2) for rank in range(16)]
    assert validate_fabric_clique(dpx) == (True, ())

    spx = [_capability(rank) for rank in range(16)]
    supported, reasons = validate_fabric_clique(spx)
    assert not supported
    assert any("VPOD size" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("peer", "expected"),
    [
        (_capability(1), True),
        (_capability(1, vpod_id="3"), False),
        ("rank 1: fabric probe failed: OSError()", False),
    ],
)
def test_probe_fabric_clique_agrees(monkeypatch, peer, expected):
    """Every rank gets the same answer, including when a peer's probe raised."""
    monkeypatch.setattr(gfx1250_fabric.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(gfx1250_fabric.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        gfx1250_fabric,
        "probe_local_fabric_capability",
        lambda device, rank: _capability(0),
    )

    def all_gather_object(gathered, local, group):
        gathered[:] = [local, peer]

    monkeypatch.setattr(gfx1250_fabric.dist, "all_gather_object", all_gather_object)
    assert gfx1250_fabric.probe_fabric_clique(object(), Mock(index=0)) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
