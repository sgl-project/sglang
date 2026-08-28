"""Embedded Rust renderer sidecar lifecycle."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import socket
import time
from collections.abc import Sequence

from sglang.srt.utils.common import kill_itself_when_parent_died, kill_process_tree
from sglang.srt.utils.network import NetworkAddress, get_free_port
from sglang.srt.utils.watchdog import SubprocessWatchdog

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0
SHUTDOWN_TIMEOUT = 10.0


def find_renderer_binary() -> str:
    binary = os.environ.get("SGLANG_RENDERER_BIN") or shutil.which("sglang-renderer")
    if binary is None:
        raise RuntimeError(
            "sglang-renderer executable was not found; install a wheel built "
            "with the Rust renderer or set SGLANG_RENDERER_BIN"
        )
    return binary


def build_renderer_args(
    server_args,
    model_config,
    public_addr: NetworkAddress,
    internal_server_url: str,
    num_reserved_tokens: int,
) -> list[str]:
    """Translate resolved server state into the renderer-owned CLI subset."""
    args = [
        str(server_args.model_path),
        "--engine-url",
        internal_server_url,
        "--fallback-url",
        internal_server_url,
        "--tokenizer-path",
        str(server_args.tokenizer_path),
        "--served-model-name",
        server_args.served_model_name,
        "--host",
        public_addr.host,
        "--port",
        str(public_addr.port),
        "--tokenizer-workers",
        str(server_args.tokenizer_worker_num),
        "--sampling-defaults",
        server_args.sampling_defaults,
        "--context-length",
        str(model_config.context_len),
        "--vocab-size",
        str(model_config.vocab_size),
        "--num-reserved-tokens",
        str(num_reserved_tokens),
    ]
    for flag, value in [
        ("--revision", server_args.revision),
        ("--chat-template", server_args.chat_template),
        ("--tool-call-parser", server_args.tool_call_parser),
        ("--reasoning-parser", server_args.reasoning_parser),
    ]:
        if value is not None:
            args.extend((flag, str(value)))
    if server_args.default_chat_template_kwargs:
        args.extend(
            (
                "--default-chat-template-kwargs",
                json.dumps(
                    server_args.default_chat_template_kwargs,
                    separators=(",", ":"),
                ),
            )
        )
    for enabled, flag in [
        (server_args.allow_auto_truncate, "--allow-auto-truncate"),
        (server_args.enable_return_hidden_states, "--enable-return-hidden-states"),
        (
            server_args.stream_response_default_include_usage,
            "--stream-response-default-include-usage",
        ),
    ]:
        if enabled:
            args.append(flag)
    return args


def validate_embedded_renderer(server_args, model_config) -> None:
    if server_args.skip_tokenizer_init:
        raise ValueError("SGLANG_RUST_RENDERER requires a tokenizer")
    if model_config.is_multimodal:
        raise ValueError(
            "SGLANG_RUST_RENDERER currently supports text-only models; "
            "multimodal OpenAI preprocessing is not implemented"
        )
    if server_args.ssl_keyfile or server_args.ssl_certfile:
        raise ValueError("SGLANG_RUST_RENDERER does not yet implement TLS")
    if server_args.enable_http2:
        raise ValueError("SGLANG_RUST_RENDERER does not yet implement HTTP/2")
    if server_args.hf_chat_template_name:
        raise ValueError(
            "SGLANG_RUST_RENDERER does not yet apply --hf-chat-template-name"
        )
    if server_args.completion_template:
        raise ValueError(
            "SGLANG_RUST_RENDERER does not yet apply --completion-template"
        )
    if server_args.enable_cache_report:
        raise ValueError(
            "SGLANG_RUST_RENDERER does not yet implement --enable-cache-report"
        )


def renderer_process(
    binary: str, args: Sequence[str], cores: Sequence[int] | None
) -> None:
    kill_itself_when_parent_died()
    if cores is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cores))
        except OSError as error:
            logger.warning("Rust renderer could not set CPU affinity: %s", error)
    os.execv(binary, [binary, *args])


def connect_host(host: str) -> str:
    if not host or host == "0.0.0.0":
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "::1"
    return host


class RustRendererHost:
    """Own the optional renderer in front of one embedded Rust server."""

    def __init__(
        self,
        server_args,
        model_config,
        public_addr: str,
        num_reserved_tokens: int,
    ):
        validate_embedded_renderer(server_args, model_config)
        self.public_addr = NetworkAddress.parse(public_addr)
        self.internal_server_addr = NetworkAddress("127.0.0.1", get_free_port())
        self.args = build_renderer_args(
            server_args=server_args,
            model_config=model_config,
            public_addr=self.public_addr,
            internal_server_url=self.internal_server_addr.to_url(),
            num_reserved_tokens=num_reserved_tokens,
        )
        self._sidecar: RustRendererSidecar | None = None

    def start(self, cores: Sequence[int] | None) -> None:
        if self._sidecar is not None:
            raise RuntimeError("Rust renderer is already running")
        self._sidecar = RustRendererSidecar.launch(
            args=self.args,
            public_addr=self.public_addr,
            internal_server_url=self.internal_server_addr.to_url(),
            cores=cores,
        )

    def stop(self) -> None:
        if self._sidecar is not None:
            self._sidecar.stop()
            self._sidecar = None


class RustRendererSidecar:
    def __init__(self, process, public_addr: NetworkAddress):
        self.process = process
        self.public_addr = public_addr
        self._watchdog = SubprocessWatchdog(
            processes=[process], process_names=["sglang-renderer"]
        )

    @classmethod
    def launch(
        cls,
        args: Sequence[str],
        public_addr: NetworkAddress,
        internal_server_url: str,
        cores: Sequence[int] | None = None,
    ) -> RustRendererSidecar:
        binary = find_renderer_binary()
        process = mp.get_context("spawn").Process(
            name="sglang_renderer",
            target=renderer_process,
            args=(binary, args, cores),
        )
        sidecar = cls(process, public_addr)
        try:
            process.start()
            sidecar._wait_until_listening()
            sidecar._watchdog.start()
        except BaseException:
            sidecar.stop()
            raise
        logger.info(
            "Rust renderer listening on %s and forwarding Rust-server routes to %s",
            public_addr.to_host_port_str(),
            internal_server_url,
        )
        return sidecar

    def _wait_until_listening(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        address = (connect_host(self.public_addr.host), self.public_addr.port)
        while time.monotonic() < deadline:
            if not self.process.is_alive():
                self.process.join(timeout=0)
                raise RuntimeError(
                    "sglang-renderer exited during startup with code "
                    f"{self.process.exitcode}"
                )
            try:
                with socket.create_connection(address, timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError(
            "sglang-renderer did not bind "
            f"{self.public_addr.to_host_port_str()} within {STARTUP_TIMEOUT:g} seconds"
        )

    def stop(self) -> None:
        self._watchdog.stop()
        if self.process.pid is not None and self.process.is_alive():
            os.kill(self.process.pid, signal.SIGINT)
            self.process.join(timeout=SHUTDOWN_TIMEOUT)
        elif self.process.pid is not None:
            self.process.join(timeout=0)
        if self.process.pid is not None and self.process.is_alive():
            logger.warning("Rust renderer did not stop after SIGINT; terminating it")
            self.process.terminate()
            self.process.join(timeout=SHUTDOWN_TIMEOUT)
        if self.process.pid is not None and self.process.is_alive():
            logger.warning("Rust renderer did not terminate; killing its process tree")
            kill_process_tree(self.process.pid, wait_timeout=SHUTDOWN_TIMEOUT)
