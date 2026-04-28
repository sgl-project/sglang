"""End-to-end tests for disaggregated diffusion.

Launches encoder / denoiser / decoder role instances plus a DiffusionServer
head, sends a generation request through the HTTP front-end, and verifies
that a non-empty output comes back.

Two configurations are covered:

1. :class:`TestDisaggZImage1Rank` — 1 rank per role (baseline disagg path).
2. :class:`TestDisaggZImage2RankDenoiser` — denoiser with
   ``--denoiser-sp 2`` across 2 GPUs. Exercises the multi-rank receive path
   where only rank 0 owns the RDMA TransferManager and must broadcast
   prompt/image tensors to non-rank-0 ranks before
   ``execute_forward`` — without that broadcast the denoising stage fails
   ``verify_input`` on an empty ``prompt_embeds``.

Run directly:

    pytest -v python/sglang/multimodal_gen/test/server/test_disagg_server.py
    pytest -v ... -k ZImage1Rank              # one class
    pytest -v ... -k test_generates_image     # one test
"""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import time
import unittest
from pathlib import Path

import requests
import torch

from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    find_free_port,
    wait_for_server_health,
)
from sglang.test.test_utils import CustomTestCase

HOST = "127.0.0.1"
_LOG_DIR = Path(os.environ.get("SGLANG_TEST_LOG_DIR", "/tmp"))

# Env knob: bump if a cold HF download is needed on a fresh CI runner.
_STARTUP_TIMEOUT_S = float(os.environ.get("SGLANG_DISAGG_STARTUP_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def _kill_tree(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_for_log(path: Path, message: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            try:
                if message in path.read_text(errors="ignore"):
                    return True
            except OSError:
                pass
        time.sleep(2)
    return False


def _tail_log(path: Path, n: int = 50) -> str:
    if not path.exists():
        return f"<no log at {path}>"
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError as e:
        return f"<log read failed: {e}>"
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Disagg cluster helper
# ---------------------------------------------------------------------------


class DisaggCluster:
    """Launch encoder / denoiser / decoder / server as separate processes.

    ``gpu_layout`` is a mapping role → list of physical GPU ids. The length
    of each list determines ``--num-gpus`` for that role, and the first id is
    passed as ``--base-gpu-id``. For a multi-rank role the GPUs must be
    contiguous starting from ``base-gpu-id`` (sglang derives local_rank from
    ``base-gpu-id + rank``).
    """

    def __init__(
        self,
        model: str,
        name: str,
        gpu_layout: dict[str, list[int]],
        extra_role_args: dict[str, list[str]] | None = None,
        startup_timeout: float = _STARTUP_TIMEOUT_S,
    ) -> None:
        self.model = model
        self.name = name
        self.gpu_layout = gpu_layout
        self.extra_role_args = extra_role_args or {}
        self.startup_timeout = startup_timeout
        self._procs: list[subprocess.Popen] = []
        self._fhs: list = []
        self._logs: dict[str, Path] = {}
        self._alloc_ports()

    def _alloc_ports(self) -> None:
        self.base_port = find_free_port(HOST)
        self.api_port = find_free_port(HOST)
        self._role_ports = {
            "encoder": find_free_port(HOST),
            "denoiser": find_free_port(HOST),
            "decoder": find_free_port(HOST),
        }

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "DisaggCluster":
        for attempt in range(3):
            try:
                self._launch_roles()
                self._launch_server_head()
                self._warmup()
                return self
            except Exception as e:
                print(
                    f"[disagg-test] Cluster {self.name} attempt {attempt + 1} "
                    f"failed: {e}",
                    flush=True,
                )
                self.stop()
                self._alloc_ports()
                if attempt == 2:
                    raise
        return self  # unreachable

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- internals -----------------------------------------------------------

    def _start_proc(self, cmd: list[str], log_path: Path) -> subprocess.Popen:
        fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=os.environ.copy(),
        )
        self._procs.append(proc)
        self._fhs.append(fh)
        return proc

    def _launch_roles(self) -> None:
        for role in ("encoder", "denoiser", "decoder"):
            port = self._role_ports[role]
            gpus = self.gpu_layout[role]
            log = _LOG_DIR / f"disagg_{self.name}_{role}.log"
            self._logs[role] = log

            cmd = [
                "sglang",
                "serve",
                "--model-path",
                self.model,
                "--disagg-role",
                role,
                "--disagg-server-addr",
                f"tcp://{HOST}:{self.base_port}",
                "--scheduler-port",
                str(port),
                "--num-gpus",
                str(len(gpus)),
                "--base-gpu-id",
                str(gpus[0]),
                "--log-level",
                "info",
                *self.extra_role_args.get(role, []),
            ]
            self._start_proc(cmd, log)

            ready_msg = f"Role {role.upper()} ready"
            if not _wait_for_log(log, ready_msg, self.startup_timeout):
                raise RuntimeError(
                    f"{role} failed to start for {self.name}. Log tail:\n"
                    f"{_tail_log(log)}"
                )

    def _launch_server_head(self) -> None:
        log = _LOG_DIR / f"disagg_{self.name}_server.log"
        self._logs["server"] = log
        # Role processes register their transfer work_endpoint with the
        # derived value ``tcp://0.0.0.0:<port>`` (see disagg_args.py). The
        # server head must advertise the same literal so ``_handle_register``'s
        # endpoint_to_idx exact-string match succeeds.
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--disagg-role",
            "server",
            "--encoder-urls",
            f"tcp://0.0.0.0:{self._role_ports['encoder']}",
            "--denoiser-urls",
            f"tcp://0.0.0.0:{self._role_ports['denoiser']}",
            "--decoder-urls",
            f"tcp://0.0.0.0:{self._role_ports['decoder']}",
            "--scheduler-port",
            str(self.base_port),
            "--port",
            str(self.api_port),
            "--host",
            HOST,
            "--disagg-timeout",
            "120",
            "--log-level",
            "info",
            *self.extra_role_args.get("server", []),
        ]
        self._start_proc(cmd, log)
        try:
            wait_for_server_health(
                f"http://{HOST}:{self.api_port}",
                path="/v1/models",
                timeout=self.startup_timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"server head failed to become healthy for {self.name}: {e}\n"
                f"Server log tail:\n{_tail_log(log)}"
            ) from e

    def _warmup(self) -> None:
        """Send a warmup request to establish RDMA connections."""
        try:
            _generate_image(self.api_port, self.model)
        except Exception as e:
            raise RuntimeError(
                f"Warmup request failed for {self.name}: {e}\n"
                f"Server log tail:\n{_tail_log(self._logs.get('server', Path('/dev/null')))}"
            ) from e

    def stop(self) -> None:
        for proc in self._procs:
            _kill_tree(proc.pid)
        for fh in self._fhs:
            try:
                fh.close()
            except OSError:
                pass
        # Give OS a moment to release ports before the next test.
        time.sleep(3)
        self._procs.clear()
        self._fhs.clear()


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _generate_image(api_port: int, model: str) -> bytes:
    # Use raw requests (openai SDK pulls in a lot and complicates CI deps).
    resp = requests.post(
        f"http://{HOST}:{api_port}/v1/images/generations",
        json={
            "model": model,
            "prompt": "A sunset over mountains",
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
        },
        timeout=600,
    )
    if resp.status_code != 200:
        print(
            f"[disagg-test] Server returned {resp.status_code}: {resp.text[:2000]}",
            flush=True,
        )
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["data"][0]["b64_json"])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


def _require_gpus(n: int) -> None:
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available < n:
        raise unittest.SkipTest(f"need {n} GPUs, have {available}")


class _DisaggTestBase(CustomTestCase):
    """Shared setup: launch cluster once per class, tear down at the end."""

    model: str = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    required_gpus: int = 2
    cluster_name: str = ""
    gpu_layout: dict[str, list[int]] = {}
    extra_role_args: dict[str, list[str]] = {}

    cluster: DisaggCluster | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        _require_gpus(cls.required_gpus)
        cls.cluster = DisaggCluster(
            model=cls.model,
            name=cls.cluster_name,
            gpu_layout=cls.gpu_layout,
            extra_role_args=cls.extra_role_args,
        )
        cls.cluster.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.cluster is not None:
            # Dump log tails for debugging CI failures
            for role_name, log_path in cls.cluster._logs.items():
                print(
                    f"\n=== [{cls.cluster_name}] {role_name} log tail ===",
                    flush=True,
                )
                print(_tail_log(log_path, n=80), flush=True)
            cls.cluster.stop()
            cls.cluster = None
        super().tearDownClass()


class TestDisaggZImage1Rank(_DisaggTestBase):
    """Baseline: 1 rank per role, 2 physical GPUs."""

    cluster_name = "zimage_1rank"
    required_gpus = 2
    gpu_layout = {
        "encoder": [0],
        "denoiser": [1],
        "decoder": [0],
    }

    def test_generates_image(self) -> None:
        assert self.cluster is not None
        img = _generate_image(self.cluster.api_port, self.model)
        # A real PNG is well above 1 KB; catches empty / error responses.
        self.assertGreater(len(img), 1_000, f"image too small: {len(img)} bytes")


class TestDisaggZImage2RankDenoiser(_DisaggTestBase):
    """Multi-rank denoiser (``--denoiser-sp 2``) on 2 GPUs.

    Regression guard for the bug where non-rank-0 denoiser ranks entered
    ``execute_forward`` with an empty Req because ``ParallelExecutor``'s
    REPLICATED stage does not broadcast the batch. With the fix, rank 0
    broadcasts both scalar and tensor fields over NCCL before compute.
    """

    cluster_name = "zimage_sp2"
    required_gpus = 2
    gpu_layout = {
        "encoder": [0],
        "denoiser": [0, 1],
        "decoder": [0],
    }
    extra_role_args = {
        "denoiser": ["--denoiser-sp", "2"],
    }

    def test_generates_image_with_sp2_denoiser(self) -> None:
        assert self.cluster is not None
        img = _generate_image(self.cluster.api_port, self.model)
        self.assertGreater(len(img), 1_000, f"image too small: {len(img)} bytes")


# ---------------------------------------------------------------------------
# Disagg + OTel tracing
# ---------------------------------------------------------------------------


def _generate_image_with_traceparent(
    api_port: int, model: str, trace_id_hex: str, span_id_hex: str
) -> tuple[int, bytes]:
    """Same as :func:`_generate_image` but seeds a known W3C traceparent.

    Returns ``(status_code, image_bytes)``. Kept separate so the tracing test
    can tolerate non-200 responses while still reporting useful diagnostics.
    """
    traceparent = f"00-{trace_id_hex}-{span_id_hex}-01"
    resp = requests.post(
        f"http://{HOST}:{api_port}/v1/images/generations",
        headers={"traceparent": traceparent},
        json={
            "model": model,
            "prompt": "A sunset over mountains",
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json",
        },
        timeout=600,
    )
    if resp.status_code != 200:
        return resp.status_code, b""
    return resp.status_code, base64.b64decode(resp.json()["data"][0]["b64_json"])


def _as_hex(v) -> str:
    """OTLP span trace_id/span_id/parent_span_id come back as raw bytes over
    gRPC and as hex strings over HTTP; normalize to lowercase hex."""
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, str):
        return v.lower()
    return ""


class TestDisaggZImageTracing(_DisaggTestBase):
    """End-to-end verification of OTel trace propagation across disagg roles.

    Spins up the same 1-rank cluster as :class:`TestDisaggZImage1Rank` with
    ``--enable-trace`` wired to an in-process OTLP collector on every role and
    the server head, sends one image-generation request with a controlled
    ``traceparent``, and asserts the server head plus all three role worker
    processes emit per-role ``scheduler_dispatch``/``gpu_forward`` spans under
    the same trace_id. This is the regression guard for trace-context
    propagation over the encoder→denoiser→decoder JSON hops.
    """

    cluster_name = "zimage_trace"
    required_gpus = 2
    gpu_layout = {
        "encoder": [0],
        "denoiser": [1],
        "decoder": [0],
    }

    # Populated in setUpClass so the collector port is known before
    # DisaggCluster launches.
    collector = None
    collector_port: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        # Fast batch-span-processor flush so the test doesn't wait for the
        # default 5s schedule. Must be set before sglang imports OTel.
        os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
        os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

        from sglang.test.otel_collector import LightweightOtlpCollector

        cls.collector_port = find_free_port(HOST)
        cls.collector = LightweightOtlpCollector(port=cls.collector_port)
        cls.collector.start()

        trace_args = [
            "--enable-trace",
            "--otlp-traces-endpoint",
            f"127.0.0.1:{cls.collector_port}",
        ]
        cls.extra_role_args = {
            "encoder": list(trace_args),
            "denoiser": list(trace_args),
            "decoder": list(trace_args),
            "server": list(trace_args),
        }

        # If super().setUpClass() raises, CustomTestCase's safe-setUpClass
        # wrapper will invoke tearDownClass, which stops the collector.
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            if cls.collector is not None:
                cls.collector.stop()
                cls.collector = None

    def test_disagg_spans_share_trace_id(self) -> None:
        assert self.cluster is not None
        assert self.collector is not None

        trace_id = os.urandom(16).hex()
        span_id = os.urandom(8).hex()

        # Warmup was sent (without traceparent) by DisaggCluster.__enter__;
        # clear those spans so the assertions only consider this request.
        self.collector.clear()

        status, img = _generate_image_with_traceparent(
            self.cluster.api_port, self.model, trace_id, span_id
        )
        self.assertEqual(status, 200, "request did not complete cleanly")
        self.assertGreater(len(img), 1_000, f"image too small: {len(img)} bytes")

        # Spans flush asynchronously from each role's BatchSpanProcessor. Poll
        # briefly until we see the expected shape.
        deadline = time.time() + 30
        spans = []
        while time.time() < deadline:
            spans = [
                s for s in self.collector.get_spans() if _as_hex(s.trace_id) == trace_id
            ]
            # Expect: root Req span + >=3 scheduler_dispatch + >=3 gpu_forward
            n_dispatch = sum(1 for s in spans if s.name == "scheduler_dispatch")
            n_forward = sum(1 for s in spans if s.name == "gpu_forward")
            if n_dispatch >= 3 and n_forward >= 3:
                break
            time.sleep(1)

        names = [s.name for s in spans]
        self.assertGreaterEqual(
            sum(1 for n in names if n == "scheduler_dispatch"),
            3,
            f"expected >=3 scheduler_dispatch spans (one per disagg role), "
            f"got names={names!r}",
        )
        self.assertGreaterEqual(
            sum(1 for n in names if n == "gpu_forward"),
            3,
            f"expected >=3 gpu_forward spans (one per disagg role), "
            f"got names={names!r}",
        )

        # All spans we saw must share the propagated trace_id. This is the
        # actual regression guard for this PR: it proves the W3C carrier
        # survives encoder→denoiser→decoder JSON hops (via ``_trace_state``).
        # The HTTP-level carrier extraction (root Req parented under the
        # client's span_id) is already covered by ``test_tracing.py`` in
        # monolithic mode and asserting it here is flaky — the server head's
        # BatchSpanProcessor may not flush the Req span before role spans
        # reach the collector, since the role spans close first.
        trace_ids = {_as_hex(s.trace_id) for s in spans}
        self.assertEqual(
            trace_ids,
            {trace_id},
            f"spans split across multiple traces: {trace_ids}",
        )


if __name__ == "__main__":
    unittest.main()
