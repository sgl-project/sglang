import copy
import json
import random
import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    run_bench_serving_multi,
)

class TestPrioritySchedulingPerf(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def tearDownClass(cls):
        pass

    def test_priority_scheduling_perf(self):
        # run_traffic_based_benchmark(MAX_CONCURRENCY_MODE)
        run_traffic_based_benchmark(FIXED_QPS_MODE)
        run_traffic_based_benchmark_with_preemption(MAX_CONCURRENCY_MODE)

def run_traffic_based_benchmark_with_preemption(mode: int):
    # 1. Build test configs
    bench_args = []
    bench_args_with_priority = []
    max_concurrency_or_fixed_qps = [2**i for i in range(7)]
    # We want to use the same distribution of priorities and ordering across different tests.
    prioritiy_distribution_base = [1] * 5 + [2] * 5 + [3] * 5
    random.shuffle(prioritiy_distribution_base)

    # Preemption debug - start
    prioritiy_distribution_base = [1] * 5 + [2] * 5 + [3] * 5
    # Preemption debug - end


    for c_or_qps in max_concurrency_or_fixed_qps:
        num_prompts = 15 * c_or_qps
        request_rate = 2 * c_or_qps
        if mode == FIXED_QPS_MODE:
            # Send the load for 60 seconds 
            num_prompts = 60 * c_or_qps
            request_rate = c_or_qps            
        priorities = prioritiy_distribution_base * c_or_qps
        bench_arg = get_benchmark_args(
            base_url="http://localhost:30000",
            dataset_name="random",
            tokenizer="openai/gpt-oss-120b",
            num_prompts=num_prompts,
            sharegpt_output_len=None,
            random_input_len=1024,
            random_output_len=512,
            request_rate=request_rate,
        )
        bench_arg.model = "openai/gpt-oss-120b"
        bench_arg.backend = "sglang-oai-chat"
        bench_arg.max_concurrency = c_or_qps

        bench_args.append(bench_arg)

        bench_arg_with_priority = copy.deepcopy(bench_arg)
        bench_arg_with_priority.priorities = priorities
        bench_args_with_priority.append(bench_arg_with_priority)
    
    # Preemption debug - start
    bench_args_with_priority = bench_args_with_priority[3:]
    # Preemption debug - end
    
    # Test setting dictionary that combines test setting name, corresponding server config, benchmark arguments and metadata.
    tests = {
        "priority-scheduling-with-preemption": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
                "--enable-priority-scheduling",
                "--priority-scheduling-preemption-threshold",
                "1"
            ],
            "bench_args": bench_args_with_priority,
            "use_priority": True,
        },
        "default": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
            ],
            "bench_args": bench_args,
            "use_priority": False,
        },
        "priority-scheduling": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
                "--enable-priority-scheduling",
                "--priority_scheduling_preemption_threshold",
                "1"
            ],
            "bench_args": bench_args_with_priority,
            "use_priority": True,
        },
    }

    # 2. Run Tests
    benchmark_metric_collection = dict()
    for server_setting_name, cfg in tests.items():
        benchmark_metric_collection[server_setting_name] = dict()
        results = run_bench_serving_multi(
            model="openai/gpt-oss-120b",
            base_url="http://localhost:30000",
            other_server_args=cfg["other_server_args"],
            benchmark_args=cfg["bench_args"],
        )
        if cfg["use_priority"]:
            for c_or_qps, result in zip(max_concurrency_or_fixed_qps, results):
                _, metric = result
                benchmark_metric_collection[server_setting_name][str(c_or_qps)] = metric
        else:
            for c_or_qps, result in zip(max_concurrency_or_fixed_qps, results):
                _, metric = result
                # this is to have uniform metric data format
                benchmark_metric_collection[server_setting_name][str(c_or_qps)] = dict()
                benchmark_metric_collection[server_setting_name][str(c_or_qps)]["all"] = metric

    # 3. Print results
    for c_or_qps in max_concurrency_or_fixed_qps:
        if mode == MAX_CONCURRENCY_MODE:
            print(f"\nConcurrency: {c_or_qps}")
        elif mode == FIXED_QPS_MODE:
            print(f"\nQPS: {c_or_qps}")
        header = (
            f"{'Setting':<{40}} "
            f"{'Priority':<{10}} "
            f"{'Median TTFT (ms)':<{20}} "
            f"{'Median TPOT (ms)':<{20}} "
            f"{'Median E2E Latency (ms)':<{20}} "
            f"{'Completed':>{10}}"
        )
        dashed_line = "-" * len(header)
        print(header)
        print(dashed_line)

        for i, setting_name in enumerate(tests.keys()):
            metric = benchmark_metric_collection[setting_name][str(c_or_qps)]["all"]

            # Print the main row for the setting
            print(
                f"{setting_name:<{40}} "
                f"{'Overall':<{10}} "
                f"{metric['median_ttft_ms']:<{20}.2f} "
                f"{metric['median_tpot_ms']:<{20}.2f} "
                f"{metric['median_e2e_latency_ms']:<{20}.2f} "
                f"{metric['completed']:>{10}}"
            )

            # Print per-priority metrics if they exist
            if setting_name != "default":
                for p in sorted(set(prioritiy_distribution_base)):
                    metric = benchmark_metric_collection[setting_name][str(c_or_qps)][p]
                    print(
                        f"{'':<{40}} "
                        f"{p:<{10}} "
                        f"{metric['median_ttft_ms']:<{20}.2f} "
                        f"{metric['median_tpot_ms']:<{20}.2f} "
                        f"{metric['median_e2e_latency_ms']:<{20}.2f} "
                        f"{metric['completed']:>{10}}"
                    )

            if i < len(tests) - 1:
                print(dashed_line)

    # save the data.
    with open("preemption_fixed_concurrency_bench_result_" + str(mode) + ".json" , "w") as f:
        json.dump(benchmark_metric_collection, f, indent=4)


def run_traffic_based_benchmark(mode: int):
    # 1. Build test configs
    bench_args = []
    bench_args_with_priority = []
    max_concurrency_or_fixed_qps = [2**i for i in range(7)]
    # We want to use the same distribution of priorities and ordering across different tests.
    prioritiy_distribution_base = [1] * 5 + [2] * 5 + [3] * 5
    random.shuffle(prioritiy_distribution_base)
    for c_or_qps in max_concurrency_or_fixed_qps:
        num_prompts = 15 * c_or_qps
        request_rate = 2 * c_or_qps
        if mode == FIXED_QPS_MODE:
            # Send the load for 60 seconds 
            num_prompts = 60 * c_or_qps
            request_rate = c_or_qps            
        priorities = prioritiy_distribution_base * c_or_qps
        bench_arg = get_benchmark_args(
            base_url="http://localhost:30000",
            dataset_name="random",
            tokenizer="openai/gpt-oss-120b",
            num_prompts=num_prompts,
            sharegpt_output_len=None,
            random_input_len=1024,
            random_output_len=512,
            request_rate=request_rate,
        )
        bench_arg.model = "openai/gpt-oss-120b"
        bench_arg.backend = "sglang-oai-chat"
        bench_arg.max_concurrency = c_or_qps

        bench_args.append(bench_arg)

        bench_arg_with_priority = copy.deepcopy(bench_arg)
        bench_arg_with_priority.priorities = priorities
        bench_args_with_priority.append(bench_arg_with_priority)
    # Test setting dictionary that combines test setting name, corresponding server config, benchmark arguments and metadata.
    tests = {
        "default": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
            ],
            "bench_args": bench_args,
            "use_priority": False,
        },
        "priority-scheduling": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
                "--enable-priority-scheduling",
            ],
            "bench_args": bench_args_with_priority,
            "use_priority": True,
        },
        "priority-scheduling-low-value-first": {
            "other_server_args": [
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--max-running-requests",
                "8",
                "--enable-priority-scheduling",
                "--schedule-low-priority-values-first",
            ],
            "bench_args": bench_args_with_priority,
            "use_priority": True,
        },
    }

    # 2. Run Tests
    benchmark_metric_collection = dict()
    for server_setting_name, cfg in tests.items():
        benchmark_metric_collection[server_setting_name] = dict()
        results = run_bench_serving_multi(
            model="openai/gpt-oss-120b",
            base_url="http://localhost:30000",
            other_server_args=cfg["other_server_args"],
            benchmark_args=cfg["bench_args"],
        )
        if cfg["use_priority"]:
            for c_or_qps, result in zip(max_concurrency_or_fixed_qps, results):
                _, metric = result
                benchmark_metric_collection[server_setting_name][str(c_or_qps)] = metric
        else:
            for c_or_qps, result in zip(max_concurrency_or_fixed_qps, results):
                _, metric = result
                # this is to have uniform metric data format
                benchmark_metric_collection[server_setting_name][str(c_or_qps)] = dict()
                benchmark_metric_collection[server_setting_name][str(c_or_qps)]["all"] = metric

    # 3. Print results
    for c_or_qps in max_concurrency_or_fixed_qps:
        if mode == MAX_CONCURRENCY_MODE:
            print(f"\nConcurrency: {c_or_qps}")
        elif mode == FIXED_QPS_MODE:
            print(f"\nQPS: {c_or_qps}")
        header = (
            f"{'Setting':<{40}} "
            f"{'Priority':<{10}} "
            f"{'Median TTFT (ms)':<{20}} "
            f"{'Median TPOT (ms)':<{20}} "
            f"{'Median E2E Latency (ms)':<{20}} "
            f"{'Completed':>{10}}"
        )
        dashed_line = "-" * len(header)
        print(header)
        print(dashed_line)

        for i, setting_name in enumerate(tests.keys()):
            metric = benchmark_metric_collection[setting_name][str(c_or_qps)]["all"]

            # Print the main row for the setting
            print(
                f"{setting_name:<{40}} "
                f"{'Overall':<{10}} "
                f"{metric['median_ttft_ms']:<{20}.2f} "
                f"{metric['median_tpot_ms']:<{20}.2f} "
                f"{metric['median_e2e_latency_ms']:<{20}.2f} "
                f"{metric['completed']:>{10}}"
            )

            # Print per-priority metrics if they exist
            if setting_name != "default":
                for p in sorted(set(prioritiy_distribution_base)):
                    metric = benchmark_metric_collection[setting_name][str(c_or_qps)][p]
                    print(
                        f"{'':<{40}} "
                        f"{p:<{10}} "
                        f"{metric['median_ttft_ms']:<{20}.2f} "
                        f"{metric['median_tpot_ms']:<{20}.2f} "
                        f"{metric['median_e2e_latency_ms']:<{20}.2f} "
                        f"{metric['completed']:>{10}}"
                    )

            if i < len(tests) - 1:
                print(dashed_line)

    # save the data.
    with open("fixed_concurrency_bench_result_" + str(mode) + ".json" , "w") as f:
        json.dump(benchmark_metric_collection, f, indent=4)


if __name__ == "__main__":
    unittest.main()
