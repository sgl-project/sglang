"""Launch the rpyc service server."""

import argparse
import logging

from sglang.srt.utils import start_rpyc_service_process
from sglang.srt.managers.controller.tp_worker import ModelTpService


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(message)s",
    )

    proc = start_rpyc_service_process(ModelTpService, args.port)
    print("Listen for connections...")
    while True:
        pass