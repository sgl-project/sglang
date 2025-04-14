import argparse
import logging
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import requests
import yaml


@dataclass
class ServerConfig:
    command: str
    process_names: List[str]
    default_port: int


@dataclass
class TaskConfig:
    server_cmd: str
    client_cmd: str
    name: Optional[str] = None
    server_type: Optional[str] = None


@dataclass
class TaskResult:
    name: str
    success: bool
    output: str
    runtime: float
    timestamp: str


SERVER_DEFAULTS = {
    "sglang": ServerConfig(
        command="sglang.launch_server",
        process_names=["sglang.launch_server"],
        default_port=30000,
    ),
    "vllm": ServerConfig(
        command="vllm.entrypoints.openai.api_server",
        process_names=["vllm.entrypoints.openai.api_server"],
        default_port=8000,
    ),
}


def parse_key_info(output: str) -> str:
    """Extract and format key information from the output"""
    key_info = []

    # Extract Args namespace
    args_match = re.search(r"Namespace\(.*?\)", output, re.DOTALL)
    if args_match:
        key_info.append(args_match.group(0))

    # Extract input/output token counts
    token_matches = re.findall(r"#(Input|Output) tokens: \d+", output)
    key_info.extend(token_matches)

    # Extract benchmark result section
    result_match = re.search(
        r"============ Serving Benchmark Result ============.*?={50,}",
        output,
        re.DOTALL,
    )
    if result_match:
        key_info.append(result_match.group(0))

    return "\n\n".join(key_info)


def extract_port_from_command(cmd: str, server_type: str) -> int:
    port_match = re.search(r"--port[= ](\d+)", cmd)
    if port_match:
        return int(port_match.group(1))
    return SERVER_DEFAULTS.get(server_type, ServerConfig("", [], 8000)).default_port


def detect_server_type(cmd: str) -> str:
    for server_type, config in SERVER_DEFAULTS.items():
        if config.command in cmd:
            return server_type
    return "unknown"


def stream_output(
    process: subprocess.Popen, prefix: str, logger: logging.Logger
) -> queue.Queue:
    output_queue = queue.Queue()

    def stream_pipe(pipe, prefix):
        for line in iter(pipe.readline, ""):
            if prefix == "CLIENT":
                output_queue.put(line.rstrip())
            logger.debug(f"{prefix} | {line.rstrip()}")

    stdout_thread = threading.Thread(
        target=stream_pipe, args=(process.stdout, prefix), daemon=True
    )
    stderr_thread = threading.Thread(
        target=stream_pipe, args=(process.stderr, prefix), daemon=True
    )

    stdout_thread.start()
    stderr_thread.start()
    return output_queue, (stdout_thread, stderr_thread)


class ProcessManager:
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.client_process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(__name__)

    def start_process(
        self, command: str, prefix: str
    ) -> Tuple[subprocess.Popen, queue.Queue]:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        output_queue, threads = stream_output(process, prefix, self.logger)
        return process, output_queue, threads

    def kill_process_tree(self, process: subprocess.Popen):
        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)

            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            parent.kill()
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)

            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    def cleanup(self, process_names: List[str]):
        if self.client_process:
            self.kill_process_tree(self.client_process)
            self.client_process = None

        if self.server_process:
            self.kill_process_tree(self.server_process)
            self.server_process = None

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.cmdline())
                if any(name in cmdline for name in process_names):
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


class ExperimentRunner:
    def __init__(self):
        self.process_manager = ProcessManager()
        self.logger = logging.getLogger(__name__)

    def wait_for_server(self, port: int, timeout: int = 300) -> bool:
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    self.logger.debug(f"Server ready on port {port}")
                    return True
            except requests.RequestException:
                time.sleep(2)
        return False

    def run_task(self, config: TaskConfig) -> TaskResult:
        start_time = time.time()
        client_output = []

        try:
            if not config.server_type:
                config.server_type = detect_server_type(config.server_cmd)

            server_config = SERVER_DEFAULTS.get(config.server_type)
            if not server_config:
                raise ValueError(f"Unknown server type: {config.server_type}")

            port = extract_port_from_command(config.server_cmd, config.server_type)

            self.process_manager.cleanup(server_config.process_names)

            self.logger.debug(f"Starting server: {config.name}")
            self.process_manager.server_process, _, server_threads = (
                self.process_manager.start_process(config.server_cmd, "SERVER")
            )

            if not self.wait_for_server(port):
                raise TimeoutError("Server startup timeout")

            time.sleep(10)

            self.logger.debug("Starting client")
            self.process_manager.client_process, output_queue, client_threads = (
                self.process_manager.start_process(config.client_cmd, "CLIENT")
            )

            returncode = self.process_manager.client_process.wait()

            while True:
                try:
                    line = output_queue.get_nowait()
                    client_output.append(line)
                except queue.Empty:
                    break

            if returncode != 0:
                raise RuntimeError(f"Client failed with code {returncode}")

            # Parse and format the output
            full_output = "\n".join(client_output)
            formatted_output = parse_key_info(full_output)

            return TaskResult(
                name=config.name,
                success=True,
                output=formatted_output,
                runtime=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            return TaskResult(
                name=config.name,
                success=False,
                output=str(e),
                runtime=time.time() - start_time,
                timestamp=datetime.now().isoformat(),
            )

        finally:
            if config.server_type in SERVER_DEFAULTS:
                self.process_manager.cleanup(
                    SERVER_DEFAULTS[config.server_type].process_names
                )
            time.sleep(10)


def load_config(config_path: str) -> List[TaskConfig]:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    configs = []
    for idx, entry in enumerate(config_data.get("tasks", [])):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid entry at index {idx}")

        config = TaskConfig(
            server_cmd=entry.get("server_cmd"),
            client_cmd=entry.get("client_cmd"),
            name=entry.get("name", f"task-{idx+1}"),
            server_type=entry.get("server_type"),
        )

        if not config.server_cmd or not config.client_cmd:
            raise ValueError(f"Missing commands in {config.name}")

        configs.append(config)

    return configs


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("experiment.log")],
    )


def format_results(results: List[TaskResult]) -> str:
    """Format experiment results in Markdown for GitHub step summary."""
    output = ["# Experiment Results\n"]

    for result in results:
        output.append(f"## {result.name}")
        output.append(f"**Status**: {'✅ Success' if result.success else '❌ Failed'}")
        output.append(f"**Runtime**: {result.runtime:.2f} seconds")
        output.append(f"**Timestamp**: {result.timestamp}")
        output.append("\n**Output**:\n```")
        output.append(result.output)
        output.append("```\n")

    return "\n".join(output)


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")


def write_in_github_step_summary(results: List[TaskResult]):
    """Write formatted results to GitHub step summary."""
    if not os.environ.get("GITHUB_STEP_SUMMARY"):
        logging.warning("GITHUB_STEP_SUMMARY environment variable not set")
        return

    formatted_content = format_results(results)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(formatted_content)


def main():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    results = []

    try:
        configs = load_config(args.config)
        runner = ExperimentRunner()

        for config in configs:
            logger.info(f"Running {config.name}")
            result = runner.run_task(config)
            results.append(result)

        if get_bool_env_var("SGLANG_IS_IN_CI"):
            write_in_github_step_summary(results)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
