"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args, ServerArgs
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    #server_args = prepare_server_args(sys.argv[1:])
    server_args =  ServerArgs(model_path="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    #server_args =  ServerArgs(model_path="meta-llama/Llama-3.1-8B-Instruct",
                    tp_size=2,
                    base_gpu_id=4,
                    disable_cuda_graph=True,
                    trust_remote_code=True)      

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
