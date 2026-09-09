"""Test-only lifecycle for the standalone renderer and its native engine."""

import os
import shutil
import subprocess
import time
from contextlib import ExitStack, contextmanager
from urllib.parse import urlsplit

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_free_port
from sglang.test.test_utils import popen_launch_server


@contextmanager
def launch_rust_renderer(model, base_url, *, timeout, engine_args=(), renderer_args=()):
    binary = shutil.which(os.environ.get("SGLANG_RENDERER_BIN", "sglang-renderer"))
    if binary is None:
        raise FileNotFoundError(
            "build sglang-renderer and add it to PATH or set SGLANG_RENDERER_BIN"
        )
    public = urlsplit(base_url)
    engine_port = get_free_port()
    while engine_port == public.port:
        engine_port = get_free_port()
    engine_url = f"http://127.0.0.1:{engine_port}"
    with ExitStack() as stack:
        engine = popen_launch_server(
            model,
            engine_url,
            timeout=timeout,
            # Python's default warmup sends text to the token-ID-only engine.
            # popen_launch_server waits for its pre-tokenized health probe instead.
            other_args=["--skip-server-warmup", *engine_args],
            env={"SGLANG_RUST_SERVER": "1"},
        )
        stack.callback(kill_process_tree, engine.pid)
        renderer = subprocess.Popen(
            [
                binary,
                model,
                "--engine-url",
                engine_url,
                "--host",
                public.hostname,
                "--port",
                str(public.port),
                "--proxy-unhandled-routes",
                "--sampling-defaults",
                "openai",
                *renderer_args,
            ]
        )
        stack.callback(kill_process_tree, renderer.pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if engine.poll() is not None or renderer.poll() is not None:
                raise RuntimeError("renderer or native engine exited during startup")
            try:
                response = requests.get(base_url + "/_sglang_renderer/ready", timeout=1)
                if (
                    response.status_code == 204
                    and response.headers.get("x-sglang-renderer") == "ready"
                ):
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("standalone renderer did not become ready")
        yield
