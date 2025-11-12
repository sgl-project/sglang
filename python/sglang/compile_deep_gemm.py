"""
Compile DeepGEMM Kernels for a model with specify server arguments

This script launches a server for capturing DeepGEMM calls and then compiles the kernels.
It accepts server arguments (the same as launch_server.py).

Usage:
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

"""

import argparse
import dataclasses
import multiprocessing
import os
import time

import requests

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.warmup import warmup

multiprocessing.set_start_method("spawn", force=True)

# Reduce warning
envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.set(True)
# Force enable deep gemm
envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(True)
# Force enable mha chunked kv for DeepSeek V3 to avoid missing kv_b_proj DeepGEMM case
os.environ["SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"] = "0"


@dataclasses.dataclass
class CompileArgs:
    timeout: int = 3600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--timeout", type=int, default=CompileArgs.timeout)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


@warmup("compile-deep-gemm")
async def warm_up_compile(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    print("\nGenerate warm up request for compiling DeepGEMM...\n")
    generate_req_input = GenerateReqInput(
        input_ids=[0, 1, 2, 3],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
    )
    if disaggregation_mode != "null":
        generate_req_input.bootstrap_room = 0
        generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

    await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


def launch_server_internal(server_args):
    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process_and_send_one_request(
    server_args: ServerArgs, compile_args: CompileArgs
):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = compile_args.timeout

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            if server_args.node_rank == 0:
                response = requests.get(f"{base_url}/v1/models", headers=headers)
            else:
                # This http api is created by launch_dummy_health_check_server for none-rank0 node.
                response = requests.get(f"{base_url}/health", headers=headers)
            if response.status_code == 200:
                # Rank-0 node send a request to sync with other node and then return.
                if server_args.node_rank == 0:
                    payload = {
                        "input_ids": [0, 1, 2, 3],
                        "sampling_params": {
                            "max_new_tokens": 8,
                            "temperature": 0,
                        },
                    }
                    # In PD mode, include fake bootstrap fields so workers don't assert
                    if server_args.disaggregation_mode != "null":
                        payload["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
                        payload["bootstrap_room"] = 0

                    response = requests.post(
                        f"{base_url}/generate",
                        json=payload,
                        timeout=600,
                    )
                    if response.status_code != 200:
                        error = response.json()
                        raise RuntimeError(f"Sync request failed: {error}")
                # Other nodes should wait for the exit signal from Rank-0 node.
                else:
                    start_time_waiting = time.perf_counter()
                    while proc.is_alive():
                        if time.perf_counter() - start_time_waiting < timeout:
                            time.sleep(10)
                        else:
                            raise TimeoutError("Waiting for main node timeout!")
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError(
        "DeepGEMM Kernels compilation timeout."
        "\n\nFeel free and please restart the command."
    )


def refine_server_args(server_args: ServerArgs, compile_args: CompileArgs):
    # Disable cuda graph and torch compile to save time
    server_args.disable_cuda_graph = True
    server_args.enable_torch_compile = False
    print(f"Disable CUDA Graph and Torch Compile to save time...")

    server_args.load_format = "dummy"
    print(f"Set load format to dummy to save time...")

    # Set watchdog timeout to compile_args.timeout because compilation will take a long time
    server_args.watchdog_timeout = compile_args.timeout
    server_args.warmups = "compile-deep-gemm"


def run_compile(server_args: ServerArgs, compile_args: CompileArgs):
    print(
        "Begin DeepGEMM Kernels compilation...\n"
        "It may take a long time and timeout maybe raised "
        "while the compilation is still in progress.\n"
        "Just feel free to restart the command "
        "until the compilation is fully finished.\n"
    )

    proc = launch_server_process_and_send_one_request(server_args, compile_args)

    print("\nDeepGEMM Kernels compilation finished successfully.")

    # Sleep for safety
    time.sleep(10)
    if proc.is_alive():
        # This is the rank0 node.
        kill_process_tree(proc.pid)
    else:
        try:
            kill_process_tree(proc.pid)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    CompileArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    compile_args = CompileArgs.from_cli_args(args)

    refine_server_args(server_args, compile_args)

    run_compile(server_args, compile_args)
