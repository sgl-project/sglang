import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_monitor.py"
)


def load_monitor_module():
    spec = importlib.util.spec_from_file_location("pd_flip_observation_monitor", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reset_window_excludes_triggering_samples():
    m = load_monitor_module()
    monitor = m.PDFlipSLOMonitor.__new__(m.PDFlipSLOMonitor)
    monitor.window = m.SLOWindow(10.0)
    monitor.window.add(
        m.NodeSLOSample(1.0, "p0", "prefill", ttft=m.SampleCounts(5, 10))
    )

    monitor.reset_window()
    snapshot = monitor.window.snapshot(timestamp=2.0)

    assert monitor.window.window_seconds == 10.0
    assert snapshot.prefill_counts == m.SampleCounts()
    assert snapshot.decode_counts == m.SampleCounts()


def test_snapshot_attainment_uses_aggregate_sample_counts():
    m = load_monitor_module()
    window = m.SLOWindow(10.0)
    window.add(
        m.NodeSLOSample(1.0, "p0", "prefill", ttft=m.SampleCounts(1, 1))
    )
    window.add(
        m.NodeSLOSample(1.5, "p1", "prefill", ttft=m.SampleCounts(0, 9))
    )

    snapshot = window.snapshot(timestamp=2.0)

    assert snapshot.prefill_counts == m.SampleCounts(1, 10)
    assert snapshot.prefill_slo_attainment == 0.1


def test_legacy_snapshot_constructor_infers_counts_from_nodes():
    m = load_monitor_module()
    nodes = [
        m.NodeSLOSample(
            1.0,
            "p0",
            "prefill",
            ttft=m.SampleCounts(0, 0),
        ),
        m.NodeSLOSample(
            1.0,
            "d0",
            "decode",
            ttft=m.SampleCounts(3, 4),
            tpot=m.SampleCounts(8, 10),
        ),
    ]

    snapshot = m.ClusterSLOSnapshot(
        timestamp=1.0,
        prefill_nodes=1,
        decode_nodes=1,
        prefill_slo_attainment=None,
        decode_slo_attainment=None,
        nodes=nodes,
    )

    assert snapshot.prefill_counts == m.SampleCounts(3, 4)
    assert snapshot.decode_counts == m.SampleCounts(8, 10)
    assert snapshot.prefill_slo_attainment == 0.75
    assert snapshot.decode_slo_attainment == 0.8


def test_snapshot_constructor_preserves_explicit_counts():
    m = load_monitor_module()
    explicit_prefill = m.SampleCounts(1, 2)
    explicit_decode = m.SampleCounts(2, 5)

    snapshot = m.ClusterSLOSnapshot(
        timestamp=1.0,
        prefill_nodes=0,
        decode_nodes=0,
        prefill_slo_attainment=0.5,
        decode_slo_attainment=0.4,
        nodes=[],
        prefill_counts=explicit_prefill,
        decode_counts=explicit_decode,
    )

    assert snapshot.prefill_counts is explicit_prefill
    assert snapshot.decode_counts is explicit_decode
    assert snapshot.prefill_slo_attainment == 0.5
    assert snapshot.decode_slo_attainment == 0.4
