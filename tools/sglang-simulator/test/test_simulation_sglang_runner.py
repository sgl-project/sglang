import os

from sglang_simulator.dataset import DatasetArgs, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from transformers import AutoTokenizer

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from sglang_simulator.simulation.sglang.bench_runner import (
    SGLangBenchmarkRunner,
)


def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    model_path = "Qwen/Qwen3-8B"
    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_storage_backend="file",
            max_total_tokens=8192,
            page_size=2,
        )
    )

    # Test with benchmark config
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=10,
        min_input_len=100,
        max_input_len=101,
        min_output_len=1,
        max_output_len=2,
    )
    dataset = get_dataset(
        dataset_args, tokenizer=AutoTokenizer.from_pretrained(model_path)
    )
    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)
    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] != 0
        ), "The created time should not be zero due to request_rate equal to 10"
        assert (
            dataset_args.min_input_len
            <= req["input_length"]
            <= dataset_args.max_input_len
        )

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
