import importlib.util
import json
import sys
from pathlib import Path


def _load_analyzer_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/analyze_dsv4_indexcache_profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_dsv4_indexcache_profile", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load_analyzer_module()


def test_dsv4_index_cache_profile_analysis_groups_trace_regions(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "dsv4_indexcache.csa_indexer.layer_2", "dur": 1000},
                    {
                        "name": "dsv4_indexcache.raw_to_page_translation.layer_4",
                        "dur": 500,
                    },
                    {"name": "dsv4_indexcache.core_attention_c4.layer_4", "dur": 2000},
                    {"name": "dsv4_indexcache.ffn_moe.layer_4", "dur": 4000},
                    {"name": "dsv4_indexcache.cuda_graph.decode.replay", "dur": 0},
                    {
                        "name": "dsv4_indexcache.cuda_graph.decode.indexcache_on.replay",
                        "dur": 0,
                    },
                    {"name": "dsv4_indexcache.cuda_graph.decode.fallback", "dur": 0},
                    {"name": "dsv4_indexcache.cuda_graph.decode.fallback", "dur": 0},
                    {"name": "unrelated", "dur": 999999},
                ]
            }
        )
    )

    summary = analyzer.summarize_trace(trace_path)

    assert summary["total_ms"] == 7.5
    assert summary["categories"]["csa_indexer"]["total_ms"] == 1.0
    assert summary["categories"]["raw_to_page_translation"]["total_ms"] == 0.5
    assert summary["categories"]["core_attention_c4"]["total_ms"] == 2.0
    assert summary["categories"]["ffn_moe"]["total_ms"] == 4.0
    assert summary["categories"]["cuda_graph"]["count"] == 4
    assert summary["cuda_graph_paths"] == {
        "decode.fallback": 2,
        "decode.indexcache_on.replay": 1,
        "decode.replay": 1,
    }
    assert summary["layers"]["4"]["raw_to_page_translation"] == 0.5
