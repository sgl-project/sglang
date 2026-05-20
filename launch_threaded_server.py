"""Launch HTTP server backed by ThreadedEngine."""
import os
import sys

os.environ["PYTHON_GIL"] = "0"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.engine_threaded import ThreadedEngine
from sglang.srt.entrypoints.http_server import _setup_and_run_http_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(raw_args)
    engine = ThreadedEngine(server_args=server_args)

    _setup_and_run_http_server(
        server_args=server_args,
        tokenizer_manager=engine.tokenizer_manager,
        template_manager=engine.template_manager,
        port_args=engine.port_args,
        scheduler_infos=engine._scheduler_init_result.scheduler_infos,
        subprocess_watchdog=None,
    )
