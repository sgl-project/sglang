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


@pytest.mark.parametrize(
    "accel_ids, expected",
    [
        # two DPX hosts, partitions of one SoC adjacent
        (
            [52, 52, 53, 53, 56, 56, 57, 57],
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        ),
        # pairs found by accel_id, not by rank parity
        ([53, 52, 52, 53], [[0, 3], [1, 2]]),
        ([52, 53, 54, 55], None),  # SPX
        ([52, 52, 52, 52], None),  # CPX
        ([52, 52, 53], None),  # half a SoC in the group
        ([52, None], None),
        ([], None),
    ],
)
def test_dpx_pairs(accel_ids, expected):
    assert gfx1250_fabric.dpx_pairs(accel_ids) == expected


@pytest.mark.parametrize(
    ("accel_ids", "expected"),
    [
        ([52, 52, 53, 53], [0, 1, 2, 3]),
        # an accelerator's partitions apart in the group: the ring keeps them together
        ([52, 53, 52, 53], [0, 2, 1, 3]),
        ([52, 53, 54, 55, 52, 53, 54, 55], [0, 4, 1, 5, 2, 6, 3, 7]),
        # one partition of an accelerator in the group, or four
        ([52, 53, 53], [0, 1, 2]),
        ([53, 52, 52, 52, 52, 53], [0, 5, 1, 2, 3, 4]),
        ([52, 53, 54, 55], [0, 1, 2, 3]),  # one partition per accelerator
        ([52, None, 52], [0, 1, 2]),
    ],
)
def test_accelerator_ring_order(accel_ids, expected):
    assert gfx1250_fabric.accelerator_ring_order(accel_ids) == expected


@pytest.mark.parametrize(("peer_accel_id", "expected"), [(52, [[0, 1]]), (None, None)])
def test_probe_dpx_pairs_gathers_every_rank(monkeypatch, peer_accel_id, expected):
    """Pairs come from every rank's ID; one rank without an ID disables them."""
    monkeypatch.setattr(gfx1250_fabric.torch.cuda, "get_device_properties", Mock())
    monkeypatch.setattr(
        gfx1250_fabric, "pci_function0_bus_id", lambda p: "0001:04:00.0"
    )
    monkeypatch.setattr(
        gfx1250_fabric, "read_ualink_sysfs", lambda bus_id: {"accel_id": "52"}
    )
    monkeypatch.setattr(gfx1250_fabric.dist, "get_world_size", lambda group: 2)

    def all_gather_object(gathered, local, group):
        gathered[:] = [local, peer_accel_id]

    monkeypatch.setattr(gfx1250_fabric.dist, "all_gather_object", all_gather_object)
    assert gfx1250_fabric.probe_dpx_pairs(object(), Mock(index=0)) == expected


def test_probe_dpx_pairs_survives_a_raising_probe(monkeypatch):
    def boom(index):
        raise RuntimeError("no device")

    monkeypatch.setattr(gfx1250_fabric.torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(gfx1250_fabric.dist, "get_world_size", lambda group: 2)
    gathered_local = []

    def all_gather_object(gathered, local, group):
        gathered_local.append(local)
        gathered[:] = [local, 52]

    monkeypatch.setattr(gfx1250_fabric.dist, "all_gather_object", all_gather_object)
    # the failing rank still joins the gather, so no rank is left waiting
    assert gfx1250_fabric.probe_dpx_pairs(object(), Mock(index=0)) is None
    assert gathered_local == [None]


def _drm_card(sysfs_root: Path, card: str, bus_id: str, partition=None) -> None:
    device = sysfs_root / "devices" / bus_id
    device.mkdir(parents=True)
    if partition is not None:
        (device / "current_compute_partition").write_text(partition, encoding="utf-8")
    (sysfs_root / card).mkdir()
    (sysfs_root / card / "device").symlink_to(device, target_is_directory=True)


def test_read_compute_partition(tmp_path: Path):
    _drm_card(tmp_path, "card1", "0001:01:00.0", "SPX\n")
    _drm_card(tmp_path, "card9", "0002:01:00.0", "dpx")
    _drm_card(tmp_path, "card17", "0003:01:00.0")

    read = gfx1250_fabric.read_compute_partition
    assert read("0001:01:00.0", tmp_path) == "SPX"
    assert read("0002:01:00.0", tmp_path) == "DPX"
    assert read("0003:01:00.0", tmp_path) is None  # no attribute
    assert read("0004:01:00.0", tmp_path) is None  # no card
    assert read(None, tmp_path) is None


@pytest.mark.parametrize(
    ("local", "peer", "expected"),
    [("SPX", "SPX", True), ("SPX", "DPX", False), ("SPX", None, False)],
)
def test_probe_spx_needs_every_rank(monkeypatch, local, peer, expected):
    monkeypatch.setattr(gfx1250_fabric.torch.cuda, "get_device_properties", Mock())
    monkeypatch.setattr(
        gfx1250_fabric, "pci_function0_bus_id", lambda p: "0001:01:00.0"
    )
    monkeypatch.setattr(gfx1250_fabric, "read_compute_partition", lambda bus_id: local)
    monkeypatch.setattr(gfx1250_fabric.dist, "get_world_size", lambda group: 2)

    def all_gather_object(gathered, local, group):
        gathered[:] = [local, peer]

    monkeypatch.setattr(gfx1250_fabric.dist, "all_gather_object", all_gather_object)
    assert gfx1250_fabric.probe_spx(object(), Mock(index=0)) is expected


def test_probe_spx_survives_a_raising_probe(monkeypatch):
    def boom(index):
        raise RuntimeError("no device")

    monkeypatch.setattr(gfx1250_fabric.torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(gfx1250_fabric.dist, "get_world_size", lambda group: 2)
    gathered_local = []

    def all_gather_object(gathered, local, group):
        gathered_local.append(local)
        gathered[:] = [local, "SPX"]

    monkeypatch.setattr(gfx1250_fabric.dist, "all_gather_object", all_gather_object)
    assert gfx1250_fabric.probe_spx(object(), Mock(index=0)) is False
    assert gathered_local == [None]


# The rows follow the CU count the device reports, whatever it is.
@pytest.mark.parametrize("num_sm", [96, 128, 256])
def test_gfx1250_spx_rows(num_sm):
    from sglang.srt.distributed.device_communicators.configs import (
        custom_all_reduce_v2 as cfg,
    )

    MB = cfg.MB
    base = cfg._rocm_gfx1250_configs(num_sm, True)
    spx = cfg._rocm_gfx1250_configs(num_sm, True, True)
    # SPX rows exist only with HIP_FORCE_DEV_KERNARG=1, where they were tuned
    assert cfg._rocm_gfx1250_configs(num_sm, False, True) == cfg._rocm_gfx1250_configs(
        num_sm, False
    )
    for ws in (3, 5, 6, 7, 8, 16):
        assert spx[ws] == base[ws]
    assert base[2].default_max_size == base[4].default_max_size == 0
    assert base[2].ring_simple_buffer_size == base[4].ring_simple_buffer_size == 0

    for ws, one_shot, two_shot, slot in (
        (2, 6 * MB, 96 * MB, 96 * MB),
        (4, 2 * MB, 160 * MB, 80 * MB),
    ):
        row = spx[ws]
        assert row.graph == row.eager
        assert row.num_push_blocks == num_sm
        assert row.graph.one_shot_push_threshold == one_shot
        assert row.graph.two_shot_push_threshold == two_shot
        assert row.graph.ring_simple.min_bytes == two_shot + 16
        assert row.graph.hier_2shot_push == row.graph.hier_ring_simple == (0, 0)
        assert row.default_max_size == slot
        assert row.ring_simple_buffer_size == 2 * MB
        # the default cap keeps 2shot_push up to its threshold
        clipped = row.clip(max_push_bytes=slot, max_pull_bytes=0, world_size=ws)
        assert clipped.graph.two_shot_push_threshold == two_shot
        assert row.max_push_bytes(ws) == slot


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
