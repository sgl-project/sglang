"""In-process benchmark runner for SGLang Simulator."""

import asyncio
import atexit
import json
import os
from dataclasses import asdict
from typing import Iterator

import numpy as np
from sglang_simulator.compat import apply_simulator_server_args
from sglang_simulator.dataset import BaseDataset, GenericRequest
from sglang_simulator.simulation.benchmark import BaseBenchmarkRunner, BenchmarkConfig
from sglang_simulator.utils.logger import get_logger

SGLANG_SIMULATOR_OUTPUT_DIR = os.getenv(
    "SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator/output"
)
SIMULATION_METRICS_PATH = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/metrics.json"
os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = SGLANG_SIMULATOR_OUTPUT_DIR

if os.getenv("SGLANG_SIMULATOR_OUTPUT_MODE") is None:
    os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"

# Import the simulator engine only after configuring its worker environment.
from sglang_simulator.simulation.sglang.engine import (  # noqa: E402
    SGLangSimulationEngine,
)

# SGLang must be imported after the simulator hooks are installed by engine.py.
from sglang.srt.server_args import ServerArgs  # noqa: E402

logger = get_logger("sglang_simulator")


class SGLangBenchmarkRunner(BaseBenchmarkRunner):
    """Run a simulator workload directly through SGLang's in-process Engine."""

    def __init__(self, server_args: ServerArgs):
        # Disable features that are unnecessary for simulation.
        server_args_kwargs = asdict(server_args)
        apply_simulator_server_args(server_args_kwargs)
        self.engine = SGLangSimulationEngine(**server_args_kwargs)
        self.server_args = self.engine.server_args
        self._shutdown = False

    def flush_cache(self):
        self.engine.flush_cache()

    def clear_hicache_storage(self):
        self.engine.loop.run_until_complete(
            self.engine.tokenizer_manager.clear_hicache_storage()
        )

    def get_request(
        self,
        dataset: BaseDataset,
        ignore_timestamp: bool = False,
        request_rate: float = float("inf"),
    ) -> Iterator[tuple[GenericRequest, dict]]:
        yield_delay = 0
        for req in dataset:
            if ignore_timestamp:
                created_time = yield_delay
                yield_delay += np.random.exponential(1.0 / request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)

            simulation_params = {
                "total_request": len(dataset),  # Include the warmup requests.
                "created_time": created_time,
            }

            yield (req, simulation_params)

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: BaseDataset,
    ):
        await self.engine.tokenizer_manager.start_profile()

        if os.path.exists(SIMULATION_METRICS_PATH):
            with open(SIMULATION_METRICS_PATH, "w") as metrics_file:
                # Clear data from a previous benchmark in the same process.
                pass

        tasks = []
        logger.info(f"Created {len(dataset)} request tasks.")
        for req, simulation_params in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ):
            task = asyncio.create_task(
                self.engine.async_generate(
                    prompt=req.prompt,
                    input_ids=req.token_ids,
                    sampling_params={
                        "ignore_eos": True,
                        "max_new_tokens": req.output_length,
                        "custom_params": {
                            # Transfer simulation arguments through sampling params.
                            "simulation": simulation_params
                        },
                    },
                )
            )
            tasks.append(task)

        _ = await asyncio.gather(*tasks)

        # Trigger the simulator's profile handler to flush final metrics.
        await self.engine.tokenizer_manager.start_profile()

        if os.path.exists(SIMULATION_METRICS_PATH):
            with open(SIMULATION_METRICS_PATH) as metrics_file:
                metrics = json.load(metrics_file)
        else:
            logger.error(
                f"Failed to load metrics from serving backend. The metrics file "
                f"should be loaded from {SIMULATION_METRICS_PATH}."
            )
            return None

        return metrics

    def benchmark(self, benchmark_config: BenchmarkConfig, dataset: BaseDataset):
        return self.engine.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset)
        )

    def get_iteration_stats(self) -> list[dict]:
        data = []
        file_path = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/iteration.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as stats_file:
                line = stats_file.readline()
                while line:
                    data.append(json.loads(line))
                    line = stats_file.readline()
        else:
            logger.error(f"The iteration statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        data = []
        file_path = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/request.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as stats_file:
                line = stats_file.readline()
                while line:
                    data.append(json.loads(line))
                    line = stats_file.readline()
        else:
            logger.error(f"The request statistics data({file_path}) does not exist.")
        return data

    def shutdown(self):
        if self._shutdown:
            return None

        logger.info("Attempting to shut down the SGLang backend engine.")
        try:
            return self.engine.shutdown()
        finally:
            self._shutdown = True
            atexit.unregister(self.engine.shutdown)
