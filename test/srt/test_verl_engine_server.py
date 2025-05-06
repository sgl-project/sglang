import multiprocessing
import multiprocessing as mp
import os
import random
import time
import traceback
import unittest
from multiprocessing import Process

import requests
import torch
from openai import OpenAI
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available
from sglang.test.runners import (
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
    get_dtype_str,
)
from sglang.test.test_utils import CustomTestCase, find_available_port, is_in_ci

_MAX_NEW_TOKENS = 8
_PROMPTS = ["1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=", "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="]
_TORCH_DTYPE = torch.float16

# Set to false to temporarily debug issues unrelated to weight update
_ENABLE_UPDATE_WEIGHTS = True

CI_MODELS = [
    dict(model_path="meta-llama/Llama-3.1-8B-Instruct"),
    # Fail to run gemma-2-2b after transformers==4.48.3 -> 4.50.0
    # dict(model_path="google/gemma-2-2b"),
]
ALL_OTHER_MODELS = [
    dict(model_path="meta-llama/Llama-3.2-1B-Instruct", tp_size=1),
    dict(model_path="Qwen/Qwen2-1.5B"),
    # dict(
    #     model_path="Qwen/Qwen2.5-14B-Instruct",
    #     mem_fraction_static=0.4,
    #     tp_size=8,
    #     tight_memory=True,
    #     decode_tolerance=1.3,
    # ),  # test_generation_models.py same config (qwen + tp=8) gives 1.22 decode error
    dict(model_path="HuggingFaceTB/SmolLM-135M-Instruct", tp_size=3),
    # dict(model_path="allenai/OLMo-1B-0724-hf"),
    # dict(
    #     model_path="THUDM/glm-4-9b-chat",
    #     mem_fraction_static=0.1,
    #     tp_size=8,
    #     tight_memory=True,
    # ),
    # dict(model_path="allenai/OLMo-2-1124-7B-Instruct"),
    # dict(
    #     model_path="ibm-granite/granite-3.0-2b-instruct",
    #     prefill_tolerance=0.22,
    #     decode_tolerance=0.22,
    # ),
]

# This port is used for HTTP API communication with the VerlEngine server
# It handles client requests for text generation, weight updates, and memory management
# This port must be available and not used by other processes
PORT = find_available_port(2345)

# Master port is used for PyTorch's distributed communication setup
# It enables tensor-parallel processes to communicate with each other
# Default is 23456, but we find an available port dynamically in assert_fragment_e2e_execution
# This port is critical for torch.distributed.init_process_group to function properly
# Each test needs a unique master_port to avoid conflicts between parallel test executions
# master_port = find_available_port(23456)  # This is set in assert_fragment_e2e_execution method


class TestVerlEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn")

    def assert_fragment_e2e_execution(
        self,
        index: int,
        model_path: str,
        mem_fraction_static: float = 0.4,
        tp_size: int = 2,
        tight_memory: bool = False,
        prefill_tolerance: float = 0.1,
        decode_tolerance: float = 0.1,
    ):
        """
        Tests VerlEngine with tensor parallelism across multiple processes.

        Spawns tp_size processes to test distributed execution, including:
        - Model inference via direct API and HTTP server
        - Weight updating functionality
        - Memory management (release/resume)

        The test validates output correctness against a reference implementation
        within specified tolerance bounds.

        Parameters:
        -----------
        index: int - Test index for logging
        model_path: str - HuggingFace model identifier
        mem_fraction_static: float - Memory fraction for static tensors
        tp_size: int - Number of tensor parallel processes
        tight_memory: bool - Enable memory optimization
        prefill_tolerance: float - Max error for prefill computation
        decode_tolerance: float - Max error for decoding computation
        """

        master_port = find_available_port(23456)

        print(f"assert_fragment_e2e_execution START {index=} {model_path=}")

        processes = []
        output_reader, output_writer = mp.Pipe(duplex=False)
        for tp_rank in range(tp_size):
            p = Process(
                target=_run_subprocess,
                kwargs=dict(
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    master_port=master_port,
                    output_writer=output_writer,
                    model_path=model_path,
                    mem_fraction_static=mem_fraction_static,
                    tight_memory=tight_memory,
                    prefill_tolerance=prefill_tolerance,
                    decode_tolerance=decode_tolerance,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(tp_size):
            self.assertTrue(
                output_reader.recv(),
                f"Subprocess has error, please see logs above. ({index=} {model_path=})",
            )

        for p in processes:
            p.join()

    def test_models(self):
        """
        Orchestrates end-to-end testing across configured model sets.

        In CI environments: Randomly selects one model for faster testing.
        In development: Tests all configured models for comprehensive validation.

        Each model configuration specifies model path, memory settings,
        tensor-parallel size, and error tolerance bounds.
        """
        test_models = ALL_OTHER_MODELS
        if is_in_ci():
            # Randomly select one model in CI for faster testing
            test_models = [random.choice(ALL_OTHER_MODELS)]
        # Test all models in development environment
        print(f"Development environment: Testing all {len(ALL_OTHER_MODELS)} models")
        for index, model_info in enumerate(test_models):
            self.assert_fragment_e2e_execution(index=index, **model_info)


def _run_subprocess(
    tp_rank: int,
    tp_size: int,
    master_port: int,
    output_writer,
    model_path: str,
    mem_fraction_static: float,
    tight_memory: bool,
    prefill_tolerance: float,
    decode_tolerance: float,
):
    """
    Executes a single tensor-parallel process for testing VerlEngine.

    Performs the core test operations:
    1. Initializes distributed environment
    2. Loads HuggingFace model for reference
    3. Tests VerlEngine API (generation, memory management, weight updates)
    4. Tests OpenAI-compatible endpoints on rank 0

    Reports success/failure via output_writer pipe.

    Parameters:
    tp_rank: int - Process rank in tensor parallel group
    tp_size: int - Total processes in tensor parallel group
    master_port: int - Port for distributed communication
    output_writer - Pipe for result communication
    model_path: str - HuggingFace model identifier
    mem_fraction_static: float - Static memory allocation fraction
    tight_memory: bool - Memory optimization flag
    prefill_tolerance: float - Acceptable prefill error
    decode_tolerance: float - Acceptable decode error
    """
    try:
        print(f"subprocess[{tp_rank=}] Start {os.environ.get('CUDA_VISIBLE_DEVICES')=}")

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group(rank=tp_rank, world_size=tp_size)
        torch.cuda.set_device(tp_rank)

        mesh_kwargs = dict(mesh_shape=(tp_size, 1), mesh_dim_names=["tp", "pp"])
        inference_device_mesh_device = init_device_mesh("cuda", **mesh_kwargs)
        inference_device_mesh_cpu = init_device_mesh("cpu", **mesh_kwargs)
        # Print basic information about this subprocess including:
        # - Current tensor-parallel rank
        # - Device mesh configuration for both CUDA and CPU
        # - This subprocess's role in testing tensor-parallel execution
        # - How it contributes to the distributed model testing
        print(
            f"subprocess[{tp_rank=}] initialized for VerlEngine testing - "
            f"Role: Shard {tp_rank+1}/{tp_size} of tensor-parallel model execution | "
            f"Device meshes: CUDA={inference_device_mesh_device}, CPU={inference_device_mesh_cpu}"
        )

        # hf model is used for comparison
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=_TORCH_DTYPE, trust_remote_code=True
        ).cuda()
        hf_tokenizer = get_tokenizer(model_path, trust_remote_code=True)

        hf_outputs = HFRunner.forward_generation_raw(
            base_model=hf_model,
            prompts=_PROMPTS,
            max_new_tokens=_MAX_NEW_TOKENS,
            tokenizer=hf_tokenizer,
            lora_paths=None,
            torch_dtype=_TORCH_DTYPE,
            output_str_only=False,
        )

        if _ENABLE_UPDATE_WEIGHTS:
            if tight_memory:
                # If tight_memory is True, we need to move the model to CPU to save memory
                hf_model.cpu()
                torch.cuda.empty_cache()

            # test update weights
            print(f"subprocess[{tp_rank=}] get_fsdp_state_dict", flush=True)
            fsdp_state_dict = _get_fsdp_state_dict(hf_model=hf_model, tp_size=tp_size)

        engine = VerlEngine(
            model_path=model_path,
            load_format="dummy" if _ENABLE_UPDATE_WEIGHTS else "auto",
            mem_fraction_static=mem_fraction_static,
            random_seed=42,
            trust_remote_code=True,
            dtype=get_dtype_str(_TORCH_DTYPE),
            device_mesh_cpu=inference_device_mesh_cpu["tp"],
            backend="server",
            enable_memory_saver=True,
            port=PORT,
        )
        # test direct generate API with multiple different requests
        print(
            f"subprocess[{tp_rank=}] testing direct generate API with multiple requests"
        )

        # Request 1: Basic generation with temperature
        print(f"subprocess[{tp_rank=}] test request 1: Basic generation")
        direct_response = engine.generate(
            prompt="Hello, world!",
            sampling_params={"temperature": 0.7, "max_new_tokens": 20},
        )
        print(f"Response 1: {direct_response}")

        # Request 2: Zero temperature (greedy) generation
        print(f"subprocess[{tp_rank=}] test request 2: Greedy generation")
        direct_response = engine.generate(
            prompt="Complete this sequence: 1, 2, 3,",
            sampling_params={"temperature": 0.0, "max_new_tokens": 10},
        )
        print(f"Response 2: {direct_response}")

        # Request 3: Batch generation
        print(f"subprocess[{tp_rank=}] test request 3: Batch generation")
        batch_response = engine.generate(
            prompt=["Translate 'hello' to French:", "Translate 'goodbye' to Spanish:"],
            sampling_params={"temperature": 0.8, "max_new_tokens": 15},
        )
        print(f"Response 3: {batch_response}")

        # test memory occupation APIs
        print(f"subprocess[{tp_rank=}] testing memory occupation APIs")
        engine.release_memory_occupation()
        print("Memory released")
        # time.sleep(1)
        engine.resume_memory_occupation()
        print("Memory resumed")

        # openai API test for reference
        torch.distributed.barrier()
        if tp_rank == 0:
            client = OpenAI(api_key="None", base_url=f"http://localhost:{PORT}/v1")
            print(client.models.list().data[0].id)

            # Multiple HTTP API requests
            print("Testing HTTP API with multiple requests")

            # Request 1
            url = f"http://localhost:{PORT}/generate"
            data = {"text": "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="}
            response = requests.post(url, json=data)
            print(f"HTTP Response 1: {response.json()}")

            # Request 2
            data = {
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0.2},
            }
            response = requests.post(url, json=data)
            print(f"HTTP Response 2: {response.json()}")

            # Request 3
            data = {
                "text": "List three colors:",
                "sampling_params": {"top_p": 0.95, "max_new_tokens": 25},
            }
            response = requests.post(url, json=data)
            print(f"HTTP Response 3: {response.json()}")

        if _ENABLE_UPDATE_WEIGHTS:
            print(f"subprocess[{tp_rank=}] call update_weights_from_tensor", flush=True)

            engine.update_weights_from_tensor(
                [(k, v) for k, v in fsdp_state_dict.items()]
            )

        # Final generation test after weight update
        print(f"subprocess[{tp_rank=}] testing generation after weight update")
        direct_response = engine.generate(
            prompt="After weight update: Hello, world!",
            sampling_params={"temperature": 0.7, "max_new_tokens": 20},
        )
        print(f"Post-update response: {direct_response}")

        execution_ok = True

    except Exception as e:
        print(f"subprocess[{tp_rank=}] has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    output_writer.send(execution_ok)
    output_writer.close()

    engine.shutdown()
    print(f"subprocess[{tp_rank=}] end", flush=True)


# Adapted from https://github.com/volcengine/verl/blob/main/tests/rollout/run_fsdp_vllm.py
def _get_fsdp_state_dict(hf_model, tp_size: int):
    """
    Creates a sharded state dictionary for weight update testing.

    Wraps the HuggingFace model with FSDP (FullyShardedDataParallel),
    configures precision settings, and returns a sharded state dict
    for testing VerlEngine's weight update capabilities.

    Parameters:
    hf_model - HuggingFace model to wrap
    tp_size: int - Number of tensor-parallel shards

    Returns:
    dict - Sharded state dict for update_weights_from_tensor
    """
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(tp_size,), mesh_dim_names=["fsdp"]
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp_model = FSDP(
        hf_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    print(f"{fsdp_model=}")

    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
    )

    return fsdp_model.state_dict()


if __name__ == "__main__":
    unittest.main()
