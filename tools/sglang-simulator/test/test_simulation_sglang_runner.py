import atexit
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def make_fixed_dataset(
    start_token: int,
    count: int,
    *,
    input_length: int = 1025,
    output_length: int = 1,
) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[start_token + i] * input_length,
                input_length=input_length,
                output_length=output_length,
                custom_params={"created_time": i / 10},
            )
            for i in range(count)
        ]
    )


def _write_sim_config(tmp_path: Path) -> Path:
    table_path = tmp_path / "replay.json"
    table_path.write_text(
        json.dumps({"[[1, 1024]]": 0.001, "[[1025, 0]]": 0.01}),
        encoding="utf-8",
    )
    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "replay",
            "database_path": str(table_path),
            "miss_strategy": "knn",
            "miss_knn_k": 1,
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def make_sglang_runner(tmp_path: Path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            max_total_tokens=10 * 1024,
            page_size=256,
            skip_tokenizer_init=True,
        )
    )
    runner.clear_hicache_storage()
    return runner


def test_benchmark_sglang_runs_paged_decode(tmp_path):
    runner = make_sglang_runner(tmp_path)
    dataset = make_fixed_dataset(
        1000,
        2,
        input_length=1024,
        output_length=2,
    )

    try:
        metrics = runner.benchmark(
            BenchmarkConfig(request_rate=10, ignore_request_timestamp=False),
            dataset=dataset,
        )
        request_stats = runner.get_request_stats()
    finally:
        with patch.object(atexit, "unregister", wraps=atexit.unregister) as unregister:
            runner.shutdown()
            runner.shutdown()

        unregister.assert_called_once_with(runner.engine.shutdown)

    assert metrics["completed"] == len(dataset)
    assert metrics["total_input"] == 2 * 1024
    assert metrics["total_output"] == 2 * 2
    assert metrics["mean_tpot_ms"] > 0
    assert all(
        idx == 0 or req["created_time"] > 0 for idx, req in enumerate(request_stats)
    )
