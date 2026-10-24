import runpy
from pathlib import Path


def test_observed_suite_prefers_thirty_minutes_over_extra_short_shard(monkeypatch):
    """The 154-minute CI suite should use five shards, rather than six at 26m."""
    script_dir = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(script_dir))
    namespace = runpy.run_path(str(script_dir / "compute_diffusion_partitions.py"))
    assert namespace["compute_partition_count"](9255.4, 1200, 1800, 2400, 10) == 5
