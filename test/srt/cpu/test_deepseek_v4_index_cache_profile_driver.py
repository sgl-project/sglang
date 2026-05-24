import importlib.util
import json
import sys
from pathlib import Path


def _load_profile_driver_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/profile_dsv4_indexcache_endpoint.py"
    )
    spec = importlib.util.spec_from_file_location("profile_dsv4_indexcache_endpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_driver = _load_profile_driver_module()


def test_dsv4_index_cache_profile_driver_finds_and_summarizes_traces(tmp_path):
    trace_path = tmp_path / "dsv4-indexcache-rank0.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "dsv4_indexcache.csa_indexer.layer_2", "dur": 1000},
                    {"name": "dsv4_indexcache.cuda_graph.decode.replay", "dur": 0},
                ]
            }
        )
    )

    traces = profile_driver.find_trace_files(tmp_path, "dsv4-indexcache")

    assert traces == [trace_path]
    summary = profile_driver.summarize_trace(traces[0])
    assert summary["categories"]["csa_indexer"]["total_ms"] == 1.0
    assert summary["cuda_graph_paths"] == {"decode.replay": 1}
