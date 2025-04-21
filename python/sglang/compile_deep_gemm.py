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

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree

multiprocessing.set_start_method("spawn", force=True)

# Reduce warning
os.environ["SGL_IN_DEEP_GEMM_PRE_COMPILE_STAGE"] = "1"


@dataclasses.dataclass
class CompileArgs:
    time_out: int = 600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--time-out", type=int, default=CompileArgs.time_out)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


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
    timeout = compile_args.time_out

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                # Resend a request to comfirm warmup is finished
                response = requests.post(
                    f"{base_url}/generate",
                    json={
                        "input_ids": [0, 1, 2, 3],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 2,
                            "ignore_eos": True,
                        },
                    },
                )
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def refine_server_args(server_args: ServerArgs):
    # Disbale cuda graph and torch compile to save time
    server_args.disable_cuda_graph = True
    server_args.enable_torch_compile = False
    print(f"Disable CUDA Graph and Torch Compile to save time...")

    # Set watchdog timeout to 1 hours because compilation will take a long time
    server_args.watchdog_timeout = 3600 * 1


def run_compile(server_args: ServerArgs, compile_args: CompileArgs):
    print(
        "Begin DeepGEMM Kernels compilation...\n"
        "It may take a long time and timeout maybe raised "
        "while the compilation is still in progress.\n"
        "Just feel free to restart the command "
        "until the compilation is fully finished.\n"
    )

    proc = launch_server_process_and_send_one_request(server_args, compile_args)

    kill_process_tree(proc.pid)

    print("\nDeepGEMM Kernels compilation finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    CompileArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    compile_args = CompileArgs.from_cli_args(args)

    refine_server_args(server_args)

    run_compile(server_args, compile_args)
