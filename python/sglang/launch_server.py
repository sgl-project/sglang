"""Launch the inference server."""

import argparse
import os

import torch.multiprocessing as multiprocessing

from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_child_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    multiprocessing.set_start_method("forkserver", force=True)

    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_child_process(os.getpid(), including_parent=False)
