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
