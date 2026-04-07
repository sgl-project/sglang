import asyncio
import json
import os
from dataclasses import asdict
from typing import Iterator

import numpy as np
import sglang_simulator.hook as sglang_simulator_hook
import torch
from sglang_simulator.dataset import (
    BaseDataset,
    GenericRequest,
)
from sglang_simulator.simulation.benchmark import BaseBenchmarkRunner, BenchmarkConfig
from sglang_simulator.simulation.sglang import (
    cache_controller,
    hicache_storage,
    hiradix_cache,
    mem_cache_allocator,
    mem_pool_host,
    model_runner,
    scheduler,
    sgl_kernel_hook,
)
from sglang_simulator.utils.logger import get_logger

# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
    sglang_simulator_hook.install_module_hooks(
        [sgl_kernel_hook.M_SGLangKernelLoadUtilHook]
    )
sglang_simulator_hook.install_class_hooks(
    [
        scheduler.C_SchedulerHook,
        model_runner.C_ModelRunnerHook,
        hicache_storage.C_StorageBackendFactory,
        cache_controller.C_HiCacheController,
        hiradix_cache.C_HiRadixCacheHook,
        mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
        mem_pool_host.C_MHATokenToKVPoolHostHook,
        mem_pool_host.C_HostKVCacheHook,
    ]
)


SGLANG_SIMULATOR_OUTPUT_DIR = "/tmp/sglang_simulator/output"
HISIM_METRICS_PATH = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/metrics.json"
os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = SGLANG_SIMULATOR_OUTPUT_DIR

if os.getenv("HISIM_SIMULATION_MODE") is None:
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"

from transformers import AutoTokenizer  # noqa

# The sglang must be imported after the hook installer
from sglang.srt.entrypoints.engine import Engine  # noqa
from sglang.srt.server_args import ServerArgs  # noqa

logger = get_logger("sglang_simulator")


class SGLangBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: ServerArgs):
        # disable some features which is not necessary for simulation.
        server_args.disable_cuda_graph = True
        self.server_args = server_args
        self.engine = Engine(**asdict(server_args))

        self._tokenizer: AutoTokenizer = None

    def flush_cache(self):
        self.engine.flush_cache()

    def clear_hicache_storage(self):
        self.engine.tokenizer_manager.clear_hicache_storage()

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
                "total_request": len(dataset),  # include the warmup requests.
                "created_time": created_time,
            }

            yield (req, simulation_params)

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: BaseDataset,
    ):
        await self.engine.tokenizer_manager.start_profile(profile_prefix="reset")

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "w") as f:
                # clear data
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
                            # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                            "simulation": simulation_params
                        },
                    },
                )
            )
            tasks.append(task)

        _ = await asyncio.gather(*tasks)

        # dump result
        await self.engine.tokenizer_manager.start_profile()

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "r") as f:
                metrics = json.loads(f.readline())
        else:
            logger.error(
                f"Failed to load metrics from serving backend. The metrics file should be loaded from {HISIM_METRICS_PATH}."
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
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The iteration statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        data = []
        file_path = f"{SGLANG_SIMULATOR_OUTPUT_DIR}/request.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The request statistics data({file_path}) does not exist.")
        return data

    def shutdown(self):
        logger.info("Attempting to shut down the SGLang backend engine.")
        return self.engine.shutdown()
