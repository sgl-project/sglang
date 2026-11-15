"""Rust-mode ``--warmups``, which the Python HTTP server runs in its lifespan.

Without a Python HTTP server, the listener-hosting scheduler runs the
``--warmups`` registry through its own listener (each listener warms its own
DP rank) once its scheduler loop serves requests, and holds health at 503 until
they finish; the Rust startup warmup runs after them, as Python's does. A
failed warmup takes the server down, as a failed lifespan startup does in
Python mode; a drain that begins meanwhile (SIGTERM) only stops them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import threading
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

import requests

from sglang.srt.entrypoints.warmup import _rust_server_warmups, _warmup_registry
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.rust_extensions._server import Server

logger = logging.getLogger(__name__)

# The GenerateReqInput fields a registered warmup may set; anything else would
# reach the listener as a field it might refuse or ignore.
_FORWARDED_FIELDS = (
    "text",
    "input_ids",
    "sampling_params",
    "image_data",
    "audio_data",
    "modalities",
    "bootstrap_host",
    "bootstrap_room",
)


def rust_server_warmup_names(warmups: Optional[str]) -> List[str]:
    """``--warmups`` as Rust mode runs it: the comma-separated names, less the
    blank ones an empty value or a trailing comma leaves (Python warns about
    those and runs nothing for them)."""
    if warmups is None:
        return []
    return [name for name in warmups.split(",") if name.strip()]


def check_rust_server_warmups(names: List[str]) -> None:
    """Refuse a warmup Rust mode cannot run: one not marked Rust-capable, or
    one not registered at all (Python mode only warns about that).

    The launcher checks before it spawns the schedulers (``Engine``) and the
    scheduler that runs them checks again. The scheduler cannot tell a
    misspelled name from a warmup an entry point registered only in the
    launcher process, and skipping such a warmup would leave it silently
    unrun, so both refuse it."""
    unknown = sorted(name for name in names if name not in _warmup_registry)
    unsupported = sorted(
        name
        for name in names
        if name in _warmup_registry and name not in _rust_server_warmups
    )
    errors = []
    if unsupported:
        errors.append(
            f"{', '.join(unsupported)}: Rust mode runs only warmups registered "
            "with warmup(name, rust_server=True), which send plain generate "
            "requests to the listener"
        )
    if unknown:
        errors.append(
            f"{', '.join(unknown)}: not registered in this process (Rust-capable "
            "warmups must register when their module is imported)"
        )
    if errors:
        raise ValueError(
            "Rust-server mode (SGLANG_RUST_SERVER=1) cannot run --warmups "
            + "; ".join(errors)
            + ". Remove them or unset SGLANG_RUST_SERVER."
        )


def _warmup_timeout(disaggregation_mode: str) -> float:
    # The Rust startup warmup's bound (`warmup_timeout` in http/app.rs).
    timeout = envs.SGLANG_WARMUP_TIMEOUT.get()
    if timeout > 0 and math.isfinite(timeout):
        return timeout
    return 1800 if disaggregation_mode != "null" else 600


class _WarmupsStopped(Exception):
    """The server began draining: the warmups stop without failing."""


class _ListenerClient:
    """The slice of ``TokenizerManager`` a Rust-capable warmup uses: one
    ``generate_request`` per request, sent to the listener's ``/generate``.

    With ``--api-key`` the listener requires it on ``/generate`` (a ``NORMAL``
    route, which ``--admin-api-key`` does not unlock), so the client sends it;
    Python's warmups run in process and never cross that check."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        api_key: Optional[str],
        is_draining: Callable[[], bool],
    ):
        self._url = f"{base_url}/generate"
        # Checked before each request: a warmup may send hundreds, and a
        # draining listener still admits them.
        self._is_draining = is_draining
        # Each request's bound: `/generate` sends nothing until it finishes, so
        # a request the listener never answers fails after this long.
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # The listener is this process's own; a proxy from the environment
        # (CI's http_proxy, whose no_proxy may not list the bind address) would
        # refuse or misroute it.
        self._session = requests.Session()
        self._session.trust_env = False

    @staticmethod
    def _payload(req: GenerateReqInput) -> Dict[str, Any]:
        # The listener answers a batch 200 even when an item fails.
        if isinstance(req.text, list) or (
            req.input_ids and isinstance(req.input_ids[0], list)
        ):
            raise ValueError("a Rust-mode warmup sent a batch request")
        payload = {}
        for field in dataclasses.fields(req):
            value = getattr(req, field.name)
            default = (
                field.default_factory()
                if field.default_factory is not dataclasses.MISSING
                else field.default
            )
            if value == default:
                continue
            if field.name not in _FORWARDED_FIELDS:
                raise ValueError(
                    f"a Rust-mode warmup set GenerateReqInput.{field.name}, "
                    "which it does not forward"
                )
            payload[field.name] = value
        return payload

    async def generate_request(
        self, req: GenerateReqInput, request: Optional[Any] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        if self._is_draining():
            raise _WarmupsStopped()
        response = await asyncio.to_thread(
            self._session.post,
            self._url,
            json=self._payload(req),
            headers=self._headers,
            timeout=self._timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"warmup request failed: {response.status_code} {response.text[:500]}"
            )
        yield response.json()


def _run_warmups(
    server: Server,
    base_url: str,
    disaggregation_mode: str,
    warmup_names: List[str],
    api_key: Optional[str],
    scheduler_loop_started: threading.Event,
    is_draining: Callable[[], bool],
    on_failure: Callable[[str], None],
) -> None:
    # What a failure names.
    stage = "startup"
    try:
        client = _ListenerClient(
            base_url, _warmup_timeout(disaggregation_mode), api_key, is_draining
        )
        # Python runs them once the schedulers are up; each request's timeout
        # must not count the rest of scheduler initialization.
        while not scheduler_loop_started.wait(timeout=1.0):
            if is_draining():
                raise _WarmupsStopped()
        for name in warmup_names:
            stage = name
            logger.info(f"Running warmup {name}")
            asyncio.run(_warmup_registry[name](disaggregation_mode, client))
        stage = "health release"
        # A drain that began during the last request: releasing would start
        # the Rust startup warmup under the drain.
        if is_draining():
            raise _WarmupsStopped()
        server.finish_custom_warmups()
        logger.info("Warmup ended")
    except BaseException as e:
        if is_draining():
            # Stopped by the drain, or the drain closed the listener under it;
            # health stays 503 through the drain.
            logger.info(f"Warmup stopped: the server is draining ({e!r})")
            return
        logger.exception(f"Custom warmup {stage} failed")
        on_failure(f"custom warmup {stage} failed: {e!r}")


def start_rust_server_warmups(
    server: Server,
    base_url: str,
    disaggregation_mode: str,
    warmup_names: List[str],
    api_key: Optional[str],
    scheduler_loop_started: threading.Event,
    is_draining: Callable[[], bool],
    on_failure: Callable[[str], None],
) -> Optional[threading.Thread]:
    """Run the warmups off the scheduler thread, once ``scheduler_loop_started``
    is set; release health once they all finish, and call ``on_failure`` with
    the error if one fails. Once ``is_draining``, the rest are skipped and a
    failure is not one."""
    if not warmup_names:
        return None
    thread = threading.Thread(
        target=_run_warmups,
        args=(
            server,
            base_url,
            disaggregation_mode,
            warmup_names,
            api_key,
            scheduler_loop_started,
            is_draining,
            on_failure,
        ),
        name="rust-server-warmups",
        daemon=True,
    )
    thread.start()
    return thread
