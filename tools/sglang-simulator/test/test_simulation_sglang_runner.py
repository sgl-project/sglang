import atexit
import json
import os
from pathlib import Path
from unittest.mock import patch

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def make_fixed_dataset(start_token: int, count: int) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[start_token + i] * 1025,
                input_length=1025,
                output_length=1,
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


def test_benchmark_sglang(tmp_path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from sglang_simulator.simulation.sglang.bench_runner import SGLangBenchmarkRunner

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

    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=False)
    cached_ds = make_fixed_dataset(1000, 8)
    evict_l1_ds = make_fixed_dataset(2000, 10)
    evict_l2_ds = make_fixed_dataset(3000, 20)

    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["completed"] == len(cached_ds)
    assert metrics["prefix_cache_reused_ratio"] == 0
    assert all(
        idx == 0 or req["created_time"] > 0
        for idx, req in enumerate(runner.get_request_stats())
    )

    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_device_hit_ratio"] > 0.95

    runner.benchmark(benchmark_config, dataset=evict_l1_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_host_hit_ratio"] > 0.95

    runner.benchmark(benchmark_config, dataset=evict_l2_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_storage_hit_ratio"] > 0.95
    with patch.object(atexit, "unregister", wraps=atexit.unregister) as unregister:
        runner.shutdown()
        runner.shutdown()

    unregister.assert_called_once_with(runner.engine.shutdown)
