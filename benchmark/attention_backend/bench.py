import argparse
import itertools
import re
import subprocess
import threading
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from sglang.utils import launch_server_cmd, terminate_process, wait_for_server


@dataclass
class ExperimentConfig:
    experiment_name: str
    # regex pattern -> metrics name dict
    benchmark_metric_pattern_dict: dict[str, str]


@dataclass
class TaskConfig:
    server_cmd: str
    client_cmd: str
    task_name: str
    experiment_config: ExperimentConfig


@dataclass
class TaskResult:
    config: TaskConfig
    # metrics name -> metrics value dict
    benchmark_metrics: dict[str, str]


def extract_benchmark_metrics(
    benchmark_metric_pattern_dict: dict[str, str], text: str
) -> dict[str, str]:
    benchmark_metrics = {}

    for line in text.splitlines():
        for pattern_str, metric_name in benchmark_metric_pattern_dict.items():
            pattern = re.compile(pattern_str)
            match = pattern.search(line)
            if match:
                benchmark_metrics[metric_name] = match.group(1)

    return benchmark_metrics


def launch_client_cmd(command: str) -> Tuple[subprocess.Popen, StringIO]:
    def stream_log(stream):
        for line in iter(stream.readline, ""):
            output_buffer.write(line)
            print(line)

    process = subprocess.Popen(
        command.replace("\\\n", " ").replace("\\", " "),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_buffer = StringIO()

    logging_thread = threading.Thread(
        target=stream_log, args=(process.stdout,), daemon=True
    )
    logging_thread.start()
    return process, output_buffer


def run_task(config: TaskConfig) -> TaskResult:
    try:
        print(f"Starting server: {config.server_cmd}")
        server_process, port = launch_server_cmd(config.server_cmd)

        # timeout = 300s
        wait_for_server(f"http://localhost:{port}", 300)

        client_cmd = f"{config.client_cmd} --port {port}"
        print(f"Starting client: {client_cmd}")
        client_process, output_buffer = launch_client_cmd(client_cmd)

        returncode = client_process.wait()
        if returncode != 0:
            raise RuntimeError(f"Client failed with code {returncode}")
        terminate_process(server_process)

        # Parse and format the output
        benchmark_metrics = extract_benchmark_metrics(
            config.experiment_config.benchmark_metric_pattern_dict,
            output_buffer.getvalue(),
        )

        return TaskResult(config=config, benchmark_metrics=benchmark_metrics)
    except Exception as e:
        print(e)
        return TaskResult(config=config, benchmark_metrics={"FAILED": str(e)})
    finally:
        try:
            print(f"Killing process {client_process}")
            terminate_process(client_process)
        except Exception as e:
            print(f"Failed to kill client process: {e}")

        try:
            print(f"Killing process {server_process}")
            terminate_process(server_process)
        except Exception as e:
            print(f"Failed to kill server process: {e}")


def load_config(config_path: str) -> List[TaskConfig]:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    config_name = Path(config_path).stem
    global_config = config_data.get("global", {})
    client_cmd = global_config.get("client_cmd")
    server_cmd = global_config.get("server_cmd")
    benchmark_metric_patterns = global_config.get("benchmark_metric_patterns")
    benchmark_metric_pattern_dict = {
        entry["regex"]: entry["name"] for entry in benchmark_metric_patterns
    }

    configs = []
    for (
        in_out_len_pair,
        model,
        attention_backend,
    ) in itertools.product(
        config_data.get("random_input_output_lens", [None]),
        config_data.get("models", []),
        config_data.get("attention_backend", []),
    ):
        model_name = model.get("name")
        additional_server_arguments = model.get("additional_server_arguments", "")
        experiment_name = f"{config_name} | {model_name}"
        task_name = f"attn={attention_backend}"
        _client_cmd = client_cmd

        if in_out_len_pair != None:
            random_input_len = in_out_len_pair.get("random_input_len")
            random_output_len = in_out_len_pair.get("random_output_len")
            _client_cmd = client_cmd.format(
                random_input_len=random_input_len,
                random_output_len=random_output_len,
            )
            task_name = f"{task_name},random_input_len={random_input_len},random_output_len={random_output_len}"

        config = TaskConfig(
            client_cmd=_client_cmd,
            server_cmd=f"{server_cmd} {additional_server_arguments}".format(
                attention_backend=attention_backend,
            ),
            task_name=task_name,
            experiment_config=ExperimentConfig(
                experiment_name=experiment_name,
                benchmark_metric_pattern_dict=benchmark_metric_pattern_dict,
            ),
        )
        configs.append(config)

    return configs


def format_results(all_results: List[TaskResult]) -> str:
    def group_results_by_experiment(
        results: List[TaskResult],
    ) -> dict[str, List[TaskResult]]:
        grouped = defaultdict(list)
        for result in results:
            experiment_name = result.config.experiment_config.experiment_name
            grouped[experiment_name].append(result)
        return grouped

    grouped_results = group_results_by_experiment(all_results)
    output = []

    for experiment_name, task_results in grouped_results.items():
        output.append(f"## Experiment - {experiment_name}\n")

        # Collect all unique metric names
        all_metric_names = sorted(
            {metric for r in task_results for metric in r.benchmark_metrics}
        )

        # Table header
        header = "| Configuration | " + " | ".join(all_metric_names) + " |"
        separator = (
            "|-------------|"
            + "|".join(["----------------" for _ in all_metric_names])
            + "|"
        )
        output.append(header)
        output.append(separator)

        # Table rows
        for result in task_results:
            config = result.config
            row = [config.task_name]
            for metric in all_metric_names:
                value = result.benchmark_metrics.get(metric, "")
                row.append(str(value))
            output.append("| " + " | ".join(row) + " |")

        output.append(f"\n### Commands\n")
        for result in task_results:
            output.append(f"#### {result.config.task_name}\n")
            output.append(f"```\n{result.config.server_cmd}")
            output.append(f"{result.config.client_cmd}\n```\n")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config files", nargs="+"
    )
    args = parser.parse_args()

    with open(f"{torch.cuda.get_device_name(0).replace(' ', '_')}.md", "w") as file:
        for config_path in args.config:
            results = []
            configs = load_config(config_path)
            for config in configs:
                result = run_task(config)
                results.append(result)

            formated_results = format_results(results)
            file.write(formated_results)


if __name__ == "__main__":
    main()
