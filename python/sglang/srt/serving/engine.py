"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import fields
from typing import Dict, List, Optional, Union

from sglang.srt.managers.controller_multi import (
    start_controller_process as start_controller_process_multi,
)
from sglang.srt.managers.controller_single import launch_tp_servers
from sglang.srt.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.serving.engine_args import EngineArgs
from sglang.srt.utils import kill_child_process, prepare_model, prepare_tokenizer

logger = logging.getLogger(__name__)


class Engine:
    """
    The core LLM Engine
    """

    def __init__(self, engine_args: EngineArgs):
        self.engine_args = engine_args

        # Spin up the engine.
        self.startup()

    def startup(self):
        """
        Start the Engine, corresponding to the shutdown method.
        """
        # Prepare model and tokenizer
        self.engine_args.model_path = prepare_model(self.engine_args.model_path)
        self.engine_args.tokenizer_path = prepare_tokenizer(
            self.engine_args.tokenizer_path
        )

        # Launch processes for multi-node tensor parallelism
        self.tp_procs = None
        if self.engine_args.nnodes > 1 and self.engine_args.node_rank != 0:
            tp_size_local = self.engine_args.tp_size // self.engine_args.nnodes
            gpu_ids = [
                i for _ in range(self.engine_args.nnodes) for i in range(tp_size_local)
            ]
            tp_rank_range = list(
                range(
                    self.engine_args.node_rank * tp_size_local,
                    (self.engine_args.node_rank + 1) * tp_size_local,
                )
            )
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                self.engine_args.nccl_ports[0],
                self.engine_args,
            )
            try:
                for p in self.tp_procs:
                    p.join()
            finally:
                kill_child_process(os.getpid(), including_parent=False)
                return

        # Initialize TokenizerManager and other processes
        self.tokenizer_manager = TokenizerManager(self.engine_args)

        pipe_controller_reader, pipe_controller_writer = mp.Pipe(duplex=False)
        pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

        if self.engine_args.dp_size == 1:
            start_process = start_controller_process_single
        else:
            start_process = start_controller_process_multi

        self.proc_controller = mp.Process(
            target=start_process,
            args=(
                self.engine_args,
                pipe_controller_writer,
            ),
        )
        self.proc_controller.start()

        self.proc_detoken = mp.Process(
            target=start_detokenizer_process,
            args=(
                self.engine_args,
                pipe_detoken_writer,
            ),
        )
        self.proc_detoken.start()

        # Wait for the model to finish loading
        controller_init_state = pipe_controller_reader.recv()
        detoken_init_state = pipe_detoken_reader.recv()

        if controller_init_state != "init ok" or detoken_init_state != "init ok":
            self.proc_controller.kill()
            self.proc_detoken.kill()
            raise RuntimeError(
                "Initialization failed. "
                f"controller_init_state: {controller_init_state}, "
                f"detoken_init_state: {detoken_init_state}"
            )

        assert self.proc_controller.is_alive() and self.proc_detoken.is_alive()
        logger.info(f"Engine successfully started.")

    def shutdown(self):
        # Shutdown the tokenizer_manager first, to make sure no more requests come in.
        self.tokenizer_manager.shutdown()

        # Once tokenizer_manager is shut down, we can safely shutdown Engine

        # Terminate and join TP processes if they exist
        if self.tp_procs:
            for proc in self.tp_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()

        # Shutdown proc_controller(which processes requests from tokenizer_manager), and
        # proc_detoken(which precoesses response to final ouput)/
        for proc in [self.proc_controller, self.proc_detoken]:
            if proc.is_alive():
                proc.terminate()
                proc.join()


class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        seed: int = 0,
        context_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        engine_arg_fields = {field.name for field in fields(EngineArgs)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in engine_arg_fields}

        # Warn about any extra kwargs
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in engine_arg_fields}
        if extra_kwargs:
            logger.warn(f"Warning: Ignored unexpected kwargs: {extra_kwargs}")

        engine_args = EngineArgs(
            model_path=model,
            tokenizer_path=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tp_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            random_seed=seed,
            context_length=context_length,
            **filtered_kwargs,
        )
        self.llm_engine = Engine(engine_args)

    def generate(
        self,
        prompts: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[
            Union["SamplingParams", List["SamplingParams"]]
        ] = None,
        prompt_token_ids: Optional[Union[List[List[int]], List[int]]] = None,
    ):
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either 'prompts' or 'prompt_token_ids' must be provided.")

        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params_dicts = [{} for _ in prompts]
        elif isinstance(sampling_params, List):
            sampling_params_dicts = [sp.to_dict() for sp in sampling_params]
        else:
            sampling_params_dicts = [sampling_params.to_srt_kwargs() for _ in prompts]

        gen_req_input = GenerateReqInput(
            text=prompts,
            input_ids=prompt_token_ids,
            sampling_params=sampling_params_dicts,
        )

        try:
            request = None

            # Use a synchronous call to run the async helper
            results = asyncio.run(self._generate_async_helper(gen_req_input, request))

            # Shutdown the engine
            self.llm_engine.shutdown()

            return results

        except ValueError as e:
            raise e

    async def _generate_async_helper(self, gen_req_input, request):
        results = []
        async for response in self.llm_engine.tokenizer_manager.generate_request(
            gen_req_input, request
        ):
            if isinstance(response, list):
                # if gen_req_input is a list input, it is deemed a batched input, then the response is already a list
                results.extend(response)
            else:
                results.append(response)
        return results
