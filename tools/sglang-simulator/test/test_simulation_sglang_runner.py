import os

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner,
)


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


def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    model_path = os.path.join(os.path.dirname(__file__), "assets/qwen3-8b")
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
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

    benchmark_config = BenchmarkConfig(
        request_rate=10,
        ignore_request_timestamp=False,
    )
    cached_ds = make_fixed_dataset(1000, 8)
    evict_l1_ds = make_fixed_dataset(2000, 10)
    evict_l2_ds = make_fixed_dataset(3000, 20)

    # First run: warm up cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["completed"] == len(cached_ds)
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_e2e_latency_ms"] >= 0

    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] > 0
        ), "created_time should not be zero when request_rate=10"

    assert metrics["prefix_cache_reused_ratio"] == 0

    # Second run: hit device cache
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_device_hit_ratio"] > 0.95
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_e2e_latency_ms"] >= 0

    # Evict from device cache, then hit host cache
    _ = runner.benchmark(benchmark_config, dataset=evict_l1_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_host_hit_ratio"] > 0.95
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_e2e_latency_ms"] >= 0

    # Evict from host cache, then hit storage cache
    _ = runner.benchmark(benchmark_config, dataset=evict_l2_ds)
    metrics = runner.benchmark(benchmark_config, dataset=cached_ds)
    assert metrics["kv_cache_storage_hit_ratio"] > 0.95
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_e2e_latency_ms"] >= 0

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
