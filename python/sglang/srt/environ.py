# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
import base64
import functools
import json
import os
import warnings
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Dict, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


_NON_UTF8_PREFIX = "base64:"


def _default_cache_subdir(name: str) -> str:
    """A directory under SGLANG_CACHE_DIR, for env defaults that track it.

    Pass as a callable default: SGLANG_CACHE_DIR is declared further down the
    Envs body, and resolving late also lets tests override it.
    """
    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), name)


def _default_on_in_ci() -> bool:
    """Enable an expensive self-check by default in CI."""
    return envs.SGLANG_IS_IN_CI.get()


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any, secret: bool = False):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False
        self.secret = secret

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        try:
            yield
        finally:
            if backup_present:
                os.environ[self.name] = backup_value
            else:
                os.environ.pop(self.name, None)
            self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvJSON(EnvField):
    def parse(self, value: str | None) -> list | dict | None:
        if not value:
            return None
        if os.path.exists(value):
            with open(value) as f:
                return json.load(f)
        return json.loads(value)


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_NEW_NAME = EnvBoolWithAlias(True, deprecated_name="SGLANG_OLD_NAME")
    """

    def __init__(self, default: Any, deprecated_name: str, secret: bool = False):
        super().__init__(default, secret=secret)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvStrWithAlias(_DeprecatedEnvFallback, EnvStr):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class GateGemvMode(IntEnum):
    """Small-batch Inkling gate linear implementation.

    OFF: always the cublas GEMM
    PAIR: PDL-chained GEMV and gate JIT kernels
    FUSED: single-launch GEMV + gate epilogue (last-block ticket)
    """

    OFF = 0
    PAIR = 1
    FUSED = 2


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class InvariantCheckLevel(IntEnum):
    """Signal level for value/index validity checks (see invariants.py).

    OFF: data layer only (sanitize/containment); no detection, no signal.
    WARN: detect + throttled log/count; degrade, never crash (prod on-demand).
    STRICT: detect + crash on GUARD/FATAL violations (CI default).

    The data layer is unconditional and independent of this level; only the
    detection + signal layer is gated here.
    """

    OFF = 0
    WARN = 1
    STRICT = 2


def _k31_gb_best(best: Any, default: Any) -> Callable[[], Any]:
    """A non-bool child's default: ``best`` under the K3.1 GB switch."""
    return lambda: (
        best
        if envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get()
        else default
    )


class Envs:
    # Organization principles for this registry:
    # - Put every field in exactly one topical section. Prefer an existing
    #   section; add a new one only when no current section is a clear fit.
    # - Group by the behavior and owning call sites, not by name similarity
    #   alone. Keep closely related lifecycle or feature knobs adjacent.
    # - Keep each section focused and below 30 fields. Split growing sections
    #   by subsystem or lifecycle instead of creating catch-all groups.
    # - Order broad runtime subsystems before shared storage and backends; keep
    #   platform- and model-specific integrations in dedicated later sections.
    # - Use the same three-line section header everywhere; do not add ad hoc
    #   one-line headings or append unrelated fields at the end of a section.
    # - Keep vendor-specific aliases with their owning integration, and keep
    #   test/debug knobs with the feature or test workflow they exercise.
    # - Keep explanatory comments attached to their field when moving it.
    # - For organization-only changes, AST-check that field names, descriptor
    #   types, and defaults are unchanged and that only field order moved.

    # ===================================================================
    # Runtime configuration and process identity
    # ===================================================================
    # Per-role config-namespace bookkeeping: off / record / enforce (value is
    # validated fail-loud in runtime_context, which resolves it once at import
    # so the read stays dynamo-prunable).
    SGLANG_ROLE_NAMESPACES = EnvStr("off")
    # Record mode: append each newly observed (role, namespace) pair to this
    # file so the audit survives signal-killed workers.
    SGLANG_ROLE_NAMESPACES_OUT = EnvStr(None)
    IS_H200 = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    # Explicit child values override this default for megakernel runtime optimizations.
    SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION = EnvBool(False)
    # Kimi K3.1 GB300 runtime optimizations; explicitly set children win.
    SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION = EnvBool(False)

    # ===================================================================
    # Model configuration, discovery, and weight loading
    # ===================================================================
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    # Shard the Qwen4-Exp PLE n-gram embedding within each attention-TP group
    # instead of gathering DP tokens for a global-TP lookup.
    SGLANG_USE_ATTN_TP_NGRAM = EnvBool(False)
    # Bitwise-exact, shape-guarded Qwen4 PLE decode fusion. Unsupported inputs
    # and phases fall back to the original implementation.
    SGLANG_ENABLE_QWEN4_PLE_FUSION = EnvBool(True)
    # --ple-offload-backend file: where the sparse, file-backed PLE table lives
    # (deterministic name, reused across restarts), whether prefill-sized
    # gathers hint the page cache first, and an escape hatch for the device
    # attribute check (pageable host memory reachable through host page tables).
    SGLANG_QWEN4_PLE_FILE_DIR = EnvStr(lambda: _default_cache_subdir("ple"))
    SGLANG_QWEN4_PLE_FILE_PREFETCH = EnvBool(True)
    SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK = EnvBool(False)
    # Faulting rows in maps whole page-cache folios, so the mapping creeps
    # towards full residency (~45 KB/token) and eats the free memory that
    # sizes the KV pool. Cap its resident set; 0 disables the trim.
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB = EnvFloat(8.0)
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S = EnvFloat(30.0)
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
    SGLANG_ENABLE_WEIGHT_LOADER_V2 = EnvBool(False)
    # Copy rank-local MoE slices into independent CPU storage before H2D when
    # they reference a larger mmap-backed checkpoint storage.
    SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D = EnvBool(False)
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    # In seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging.
    # Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1)
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # ===================================================================
    # HTTP server and health
    # ===================================================================
    # Decompress request bodies tagged with `x-body-compressed`.
    SGLANG_ENABLE_REQUEST_DECOMPRESSION = EnvBool(False)
    # Override parsed request fields from headers.
    SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES = EnvBool(False)
    # Name prefix of the usage accounting response headers on the kimi_k3 chat
    # surface (<prefix>-Prompt-Tokens, <prefix>-Cached-Tokens). Empty: not sent.
    SGLANG_USAGE_HEADER_PREFIX = EnvStr("")
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)
    # ASGI import string that multi-worker Uvicorn and Granian processes load;
    # point it at an app that extends `http_server.app` with extra routes.
    SGLANG_HTTP_SERVER_APP_PATH = EnvStr("sglang.srt.entrypoints.http_server:app")
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)
    SGLANG_EXPOSE_OWN_ENV_VARS = EnvBool(False)
    SGLANG_DIAG_BYPASS_HEALTH_GENERATE = EnvBool(False)

    # ===================================================================
    # Logging
    # ===================================================================
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_DECODE_GRAPH_KEY = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)
    # Read by JIT kernel headers (sgl_kernel/logging.h): DEBUG | INFO | WARNING | ERROR.
    SGLANG_JIT_LOG_LEVEL = EnvStr("WARNING")
    SGLANG_ENABLE_RANK_CONSENSUS_CHECKER = EnvBool(False)

    # ===================================================================
    # IPC, broadcasters, and ports
    # ===================================================================
    SGLANG_USE_PICKLE_IPC = EnvBool(True)
    # Log top-level PickleWrapper frames unwrapped on msgpack IPC decode.
    SGLANG_LOG_PICKLE_IPC_OBJECTS = EnvBool(False)
    # Comma-separated peer IPs admitted by TCP-bound IPC sockets; empty admits all.
    SGLANG_ZMQ_ALLOWED_PEERS = EnvTuple(tuple())
    # Source IP for outgoing TCP IPC connects, so an allowlisting peer sees it.
    SGLANG_ZMQ_SOURCE_IP = EnvStr(None)
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)
    # Base port hint for ephemeral sockets (ZMQ, SHM broadcaster, etc.).
    # When set, get_open_port() and shm_broadcast search upwards from this
    # value instead of asking the OS for a random port.  Useful to keep all
    # SGLang ports in a predictable range behind a firewall.
    SGLANG_PORT = EnvInt(None)
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # ===================================================================
    # CI and test execution
    # ===================================================================
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)
    SGLANG_TEST_TI2I_INPUT_IMAGE = EnvStr(None)
    # Expand jit_kernel test grids to their full parameter ranges (nightly).
    SGLANG_JIT_KERNEL_RUN_FULL_TESTS = EnvBool(False)
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # ===================================================================
    # Crash diagnostics and shutdown
    # ===================================================================
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # ===================================================================
    # Constrained decoding and grammar
    # ===================================================================
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # ===================================================================
    # Fault injection and regression tests
    # ===================================================================
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)
    SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB = EnvFloat(0.0)
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2**31)
    # Scheduler: report the mamba state pool full to lazy tracking-slot alloc, so each boundary evicts first
    SGLANG_TEST_MAMBA_LAZY_POOL_FULL = EnvBool(False)
    # KL tests: skip the cache-hit count assertion (e.g. when eviction reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)
    # HiCache: GPU sleep cycles before each layer's host-to-device load, so a forward
    # that skips the per-layer load wait reads unloaded layers
    SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES = EnvInt(0)

    # ===================================================================
    # CI reporting: per-model metrics jsonl for nightly XPU dashboard
    # ===================================================================
    # When set, XPU nightly tests append one JSON record per model to this file
    # so xpu-ci-job-monitor.yml can render per-model ref/actual/status/duration
    # tables. Unset (the default) is a full no-op — pre-existing CI unaffected.
    SGLANG_TEST_METRICS_FILE = EnvStr(None)

    # ===================================================================
    # PD and scripted-runtime tests
    # ===================================================================
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)
    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # ===================================================================
    # Profiling, tracing, and metrics
    # ===================================================================
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    # profile_by_stage: do not start the decode-stage capture until a decode batch
    # reaches this many requests (0 = first decode batch). Lets a batch-size bench
    # capture steady-state full-admission decode steps instead of the ramp-up.
    SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS = EnvInt(0)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBool(False)
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    # Opt-in: emit one CUDA-graph capture trace per captured batch size (per-bs).
    # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE (single combined trace) takes
    # precedence when both are set.
    SGLANG_GRAPH_BATCH_CAPTURE = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    # Allocator-history buffer for /start_profile activities=["MEM"]; the
    # default truncates long windows (each entry is one alloc/free event).
    SGLANG_MEM_PROFILE_MAX_ENTRIES = EnvInt(100000)
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_TRACE_ASYNC = EnvBool(False)
    SGLANG_TRACE_ASYNC_FLUSH_THRESHOLD = EnvInt(100)
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E = EnvBool(False)
    SGLANG_TRACE_LOGITS_E2E_SYNC = EnvBool(False)
    SGLANG_TRACE_SAMPLER_E2E = EnvBool(False)
    SGLANG_TRACE_QWEN_MOE_DEEPEP_E2E = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_CONTIG = EnvBool(False)
    SGLANG_DEEPEP_V2_TRACE_MASKED = EnvBool(False)

    # ===================================================================
    # Debugging and invariant checks
    # ===================================================================
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_VALIDATE_MAMBA_REPLAY_STATE_INDICES = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS = EnvBool(False)
    SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS = EnvBool(False)
    # NaN-fill the unified memory pool at boot (debug repro switch).
    SGLANG_DEBUG_POISON_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    # Scheduler state checks. BUSY runs the busy pool check (its value also sets
    # the log verbosity). IDLE runs no check of its own: it picks whether a leak
    # found by the idle pool checks or the KV page check raises or only warns.
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # The explicit environment variable still takes precedence over this CI
    # default, so production remains opt-in and CI remains opt-out if needed.
    SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK = EnvBool(_default_on_in_ci)
    # Check each schedule that the running requests equal the ones batches hold.
    SGLANG_ENABLE_RUNNING_REQS_CHECK = EnvBool(_default_on_in_ci)
    # Physical KV-page checks: committed<=allocated + no page alias. Runs them at
    # every busy iteration, with or without the BUSY flag.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)
    SGLANG_TBO_DEBUG = EnvBool(False)
    # Timing probe: run the swap-in fully but skip the host->device KV bytes,
    # measuring the "IO is free" floor. GARBAGE OUTPUT -- benchmarking only.
    SGLANG_DEBUG_HISPARSE_SKIP_IO = EnvBool(False)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    SGLANG_ENABLE_NAN_LOGITS_CHECK = EnvBool(False)
    # Signal level for value/index validity checks (nan/inf/oob/...); see
    # invariants.py. OFF (prod default) runs only the free data layer, WARN
    # adds throttled logging, STRICT (CI default) crashes on violations.
    # Supersedes SGLANG_ENABLE_ASYNC_ASSERT, which is bridged as STRICT until
    # every callsite migrates.
    SGLANG_INVARIANT_CHECK = EnvInt(InvariantCheckLevel.OFF)

    # ===================================================================
    # Runtime simulations
    # ===================================================================
    # Benchmark mode: changes only the accepted length; verify still runs every op real
    # serving runs, at real cost (SGLANG_SIMULATE_ACC_GREEDY is the opt-out).
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_ACC_TOKEN_MODE = EnvStr("fixed")
    # DSpark opt-in: with SGLANG_SIMULATE_ACC_LEN set, temperature-only sampling requests
    # run the greedy draft/accept path and only the bonus token is temperature-sampled.
    SGLANG_SIMULATE_ACC_GREEDY = EnvBool(False)
    # DSpark opt-in: apply the simulated correct_len / bonus inside the folded verify epilogue
    # (CUDA/HIP graph) instead of falling back to the eager accept + commit path.
    SGLANG_DSPARK_SIMULATE_FOLD = EnvBool(False)
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)

    # ===================================================================
    # DSpark speculative decoding
    # ===================================================================
    SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER = EnvBool(False)
    SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS = EnvBool(False)
    SGLANG_DSPARK_DEBUG_DUMP = EnvTuple(tuple())
    SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_STS_COLLECT_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH = EnvStr("")
    SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL = EnvInt(0)
    SGLANG_DSPARK_ENABLE_SPS_RECORD = EnvBool(False)
    SGLANG_DSPARK_FP32_LM_HEAD = EnvBool(False)
    # Argmax every draft row, sampling requests included, accepted as a one-hot
    # draft: no step logits kept, lower acceptance at high temperatures.
    SGLANG_DSPARK_USE_GREEDY_DRAFT = EnvBool(False)
    SGLANG_DSPARK_FOLDED_PROPOSAL = EnvBool(True)
    SGLANG_DSPARK_STACKED_CTX_KV = EnvBool(True)
    SGLANG_DSPARK_EMBED_IN_GRAPH = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_BF16 = EnvBool(True)
    SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD = EnvBool(True)
    # With the TP-sharded markov_w2, run the draft's vocab all-gathers (each
    # step's per-shard token candidates, and the step logits sampling keeps) over
    # the NVLink push collective (CustomAllReduceV2's multicast plane) instead of
    # the NCCL ring. Only taken when the group's communicator has a multicast
    # plane; off, or no such plane, keeps the NCCL all-gather.
    SGLANG_DSPARK_NVLINK_VOCAB_GATHER = EnvBool(True)
    SGLANG_DSPARK_ENABLE_MULTI_STREAM = EnvBool(True)
    SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS = EnvInt(2)
    # Static verify graphs with bs in [MIN_BS, MAX_BS] (MAX_BS 0: unbounded) run the
    # greedy accept and its bookkeeping as one kernel.
    SGLANG_DSPARK_FUSED_ACCEPT_TAIL = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    SGLANG_DSPARK_ACCEPT_TAIL_MIN_BS = EnvInt(1)
    SGLANG_DSPARK_ACCEPT_TAIL_MAX_BS = EnvInt(0)
    # Static verify graphs stage the draft context-KV commit; the next draft graph
    # writes it on a side stream forked at its head, joined at its first KV store.
    SGLANG_DSPARK_DEFER_CONTEXT_KV = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark draft graph: its in-graph attention-metadata prep runs on a side stream
    # and joins at the first draft KV store, overlapping the embedding.
    SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark static verify: the verify graph scatters new seq lens and the bonus into
    # the FutureMap's slots; the host skips those scatters for folded accepts.
    SGLANG_OPT_DSV41_DSPARK_POST_FOLD = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark target verify: a pre-planned replay skips re-copying input_ids / positions
    # the plan-time load already copied from the same tensors.
    SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark: draw the simulated acceptance length (and the sim bonus seed) before the
    # draft launch instead of between the draft and the target verify.
    SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD = EnvBool(False)
    # Sampled DSpark verify accepts draw with block-wise rejection (blockwise_rejection.py:
    # exact in law, counter-based draws agree across TP ranks) instead of the classic kernel.
    SGLANG_OPT_DSV41_BLOCKWISE_REJECTION = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # The deferred commit's staging (hidden, write locations, positions, commit lengths)
    # in one Triton launch instead of four copies and a fill.
    SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )

    # ===================================================================
    # Memory pools and KV-cache sizing
    # ===================================================================
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    # Disable lazy compaction in the unified memory pool allocator and
    # fall back to the per-free eager compaction. Used for production
    # A/B and quick rollback. Default False (lazy compaction on).
    SGLANG_DISABLE_LAZY_COMPACTION = EnvBool(False)
    # Periodically log lazy-compaction stats per sub-pool (observability only).
    SGLANG_LOG_LAZY_COMPACTION_STATS = EnvBool(False)
    SGLANG_LOG_LAZY_COMPACTION_STATS_INTERVAL_SEC = EnvInt(30)
    # Per-call move cap on a non-urgent lazy-compaction flush, so a large
    # backlog cannot stall the scheduler loop; urgent flushes are uncapped.
    SGLANG_LAZY_COMPACTION_MAX_MOVES_PER_CALL = EnvInt(4096)
    # Max pages one compaction move-kernel launch copies (eager or lazy, urgent
    # or not); a larger move is issued as consecutive launches of this size
    # within the same compaction call.
    SGLANG_COMPACTION_MAX_PAGES_PER_MOVE = EnvInt(4096)
    # HND KV layout folds (page, head) into one paged index for per-kv-head sparse
    # page tables (DP attn); paged backends like trtllm_mha consume it directly.
    SGLANG_USE_HND_KVCACHE = EnvBool(False)
    # --enable-kv-cache-sharding: register single-node prefix gather scratch
    # as NCCL symmetric memory and request copy engines on its dedicated
    # communicator. Leave NCCL_CTA_POLICY unset to preserve per-comm policies.
    # The engine defaults NCCL cuMem on for eligible single-node deployments.
    SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER = EnvBool(True)

    # Attention (aiter, ROCm): route NEXTN spec draft_extend (EAGLE-v2 KV
    # catch-up) through aiter unified_attention (GQA-packed + split-KV) instead
    # of the occupancy-starved mha_batch_prefill FMHA. Independent kill-switch
    # for the new path; pairs with SGLANG_AITER_UNIFIED_VERIFY. Default on.
    SGLANG_AITER_UNIFIED_DRAFT_EXTEND = EnvBool(True)
    # Use ASM prefill for gfx950 HD128 FP8 KV, including cached prefixes.
    SGLANG_AITER_ASM_PREFILL_HD128 = EnvBool(True)
    # Attention (aiter, ROCm): hand chunked prefill the page-level KV view so
    # gfx950 fp8 hd256 takes aiter's paged-varlen asm kernel. That kernel is
    # compiled for 4D LINEAR [N, 64, H, D], so it serves --page-size 64 and
    # nothing else; at any other page size the view is never built and this
    # flag does nothing whichever way it is set. Where it does apply the win
    # grows with context (~1-2% throughput and 3-7% TTFT at 60k and above) and
    # is flat to marginally negative at moderate ISL, so this is a per-workload
    # switch rather than a pure kill-switch. Off falls back to the contiguous
    # gather path. Default on.
    SGLANG_AITER_PAGED_PREFILL_ASM = EnvBool(True)
    # Bucket aiter paged_attention_ragged decode partitions by sequence length.
    SGLANG_AITER_DECODE_PARTITION_BUCKETS = EnvBool(True)
    # size the KV pool after CUDA-graph capture
    SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)

    # ===================================================================
    # Scheduler token budgeting and admission
    # ===================================================================
    SGLANG_ENABLE_WAITING_PREFIX_REFRESH = EnvBool(True)
    # PD prefill: bootstrap-queued requests each refresh also touches; 0 skips them.
    SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX = EnvInt(512)
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)
    SGLANG_MAX_NEW_TOKENS_LIMIT = EnvInt(None)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    # Window for the token-weighted recent cache-hit rate used to estimate
    # waiting-queue prefill load.
    SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS = EnvFloat(15.0)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    # Compact extend-attention scheduler tile-budget admission (AMD/HIP-only).
    # Budget <= 0 disables; >0 sets the max prefix-extend tiles per batch.
    SGLANG_PREFILL_TILE_BUDGET = EnvInt(0)
    # Tile-budget mode: "compact" (default, counts actual per-request tiles) or
    # "legacy" (rectangular grid, max_extend_len-shaped).
    # Internal/testing only - users should not need to change this.
    SGLANG_PREFILL_TILE_BUDGET_MODE = EnvStr("compact")
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)
    # shortest-prefill-first: order the in-batch shared-prefix dedupe shortest
    # first, so the shortest request of a shared-prefix group runs and the rest
    # wait for it. Off: queue order.
    SGLANG_SPF_SHORTEST_FIRST_DEDUPE = EnvBool(False)
    # Opt-in, shortest-prefill-first: hold a truncated chunked request's uncomputed
    # tail in the prefill budget, so requests admitted beside it (whose KV stays
    # locked until sent) cannot leave its later chunks without KV.
    SGLANG_K31_PF_SPF_RESERVE_CHUNKED_TAIL = EnvBool(False)

    # ===================================================================
    # Scheduler polling, timeouts, and output
    # ===================================================================
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)
    # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    # Non-streaming requests flush intermediate outputs every N decoded tokens.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)
    # Most per-request states the detokenizer keeps; past it, the oldest is
    # evicted. See https://github.com/sgl-project/sglang/issues/2812
    SGLANG_DETOKENIZER_MAX_STATES = EnvInt(1 << 16)

    # ===================================================================
    # Overlap scheduler and pipeline parallelism
    # ===================================================================
    # Process a prefill step's result before scheduling the next step, for TTFT.
    SGLANG_DISABLE_OVERLAP_AFTER_PREFILL = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP"
    )
    # Force delay_sample_func for all overlap decode (not just grammar mode),
    # allowing CPU result processing to overlap with subsequent forward computation
    # and reducing the impact of sampling overhead on the critical path.
    SGLANG_ENABLE_DELAY_SAMPLE = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # Force the WAR barrier to wait for the whole forward instead of the
    # read-done fastpath event.
    SGLANG_FORCE_COARSE_WAR_BARRIER = EnvBool(False)
    # Enable prefill read-done publication after compliant metadata initialization.
    SGLANG_ENABLE_PREFILL_WAR_READ_DONE = EnvBool(True)
    # Write the forward's small metadata (DP token counts, extend lengths, the
    # non-padded token count) with one kernel launch per array instead of pinned
    # host-to-device copies. Where other host-to-device DMA can be queued ahead of
    # these small copies on the copy engine, they wait behind it and stall the
    # forward; typically KV-cache transfers that use DMA instead of a 2-block SM
    # copy kernel, so as not to disturb persistent kernels.
    SGLANG_ENABLE_MLP_SYNC_METADATA_FILL = EnvBool(False)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    # Run PP tensor communication on a dedicated stream so asynchronous sends
    # do not fence the next forward through the scheduler stream.
    SGLANG_PP_COMM_OVERLAP = EnvBool(False)
    # PP rank 0 posts its output recv only after its own forward. Enable it if PP
    # hangs or stalls on kernels that need every SM (e.g. flashinfer MLA).
    SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD = EnvBool(False)
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    # Opt-in: keep receiving prefill requests while forward results are pending.
    SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING = EnvBool(False)
    # Opt-in: retire finished prefill KV transfers while a forward result is pending.
    SGLANG_DISAGG_PREFILL_POLL_TRANSFERS_WHILE_WAITING = EnvBool(False)
    # Opt-in: drain HiCache L3 events while a prefill forward result is pending.
    SGLANG_DISAGG_PREFILL_POLL_HICACHE_WHILE_WAITING = EnvBool(False)
    # Opt-in, PD prefill overlap loop: when the newest forward ran eagerly (no CUDA
    # graph) and the oldest result copy is complete on every TP rank, process it
    # before the next launch instead of one forward later. Unfinished forwards keep
    # the overlap; ignored with continuous input polling.
    SGLANG_DISAGG_PREFILL_EAGER_RESULT_FIRST = EnvBool(False)
    # Override NVML Confidential Computing autodetection. None = autodetect.
    SGLANG_FORCE_CONFIDENTIAL_COMPUTE = EnvBool(None)

    # ===================================================================
    # Radix and sparse KV caches
    # ===================================================================
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)
    # Kill-switch for the shared-index (IndexShare) swap-in prefetch
    # (auto-enabled for GLM-5.2-style DSA); set True to A/B synchronous swap-in.
    SGLANG_DISABLE_HISPARSE_PREFETCH = EnvBool(False)
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(True)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    # The tree-core registry falls back to Python for:
    # - Session-aware caching.
    # - C128 or other unsupported components.
    # - Custom component overrides.
    # - Non-Linux platforms.
    # - Unsupported PyTorch versions.
    # - Devices other than CPU or CUDA.
    # - Installs with neither the Rust extension nor its sources.
    # - Source builds with a missing or unusable Rust toolchain.
    # This also applies when Rust is explicitly selected.
    SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("rust")
    # Once decode passes the sliding window, drop the SWA part of the prefill's
    # tree lock; its SWA KV becomes evictable rather than freed.
    SGLANG_OPT_RELEASE_PREFILL_SWA = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW"
    )

    # ===================================================================
    # PD disaggregation runtime
    # ===================================================================
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES = EnvInt(0)
    # Enable on both P and D with the same --sampling-mask-max-tokens value.
    SGLANG_ENABLE_DISAGG_SAMPLING_MASK = EnvBool(False)
    # Retained only to reject the removed setting during startup.
    SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS = EnvInt(None)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT = EnvInt(1)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    # A wedged RDMA stack fails startup here instead of at the scheduler watchdog.
    SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT = EnvInt(60)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS = EnvInt(16384)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    # Deferred decode-side KV release: on abort, hold an in-flight request's KV
    # pages/slot until the prefill acks the transfer drained, or the timeout
    # below fires. Only applies to backends that ack the drain
    # (supports_deferred_decode_kv_release).
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE = EnvBool(True)
    SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT = EnvFloat(30.0)
    # Under overlap, publish decode KV destinations one iteration later instead of
    # reading them back inline; False restores the inline publish.
    SGLANG_ENABLE_DISAGG_DECODE_DEFERRED_PUBLISH = EnvBool(True)

    # ===================================================================
    # Distributed and model-parallel runtime
    # ===================================================================
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBoolWithAlias(
        False, deprecated_name="SYNC_TOKEN_IDS_ACROSS_TP"
    )
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)
    # Gated background transfers (distributed/device_communicators/
    # gated_transfer.py): force open a transfer gate whose pause has been held
    # this long, logging an error. A pause that never resumes means work the
    # resume depends on is stuck behind a parked chunk. 0 disables it.
    SGLANG_GATED_TRANSFER_WATCHDOG_MS = EnvInt(1000)

    # ===================================================================
    # Tool calling and native web search
    # ===================================================================
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)
    # Native web search (Exa). EXA_API_KEY is the vendor BYOK credential
    # (kept as-is, not renamed to SGLANG_*); the SGLANG_EXA_* knobs tune the
    # request defaults for the built-in GPT-OSS web_search tool.
    EXA_API_KEY = EnvStr(None, secret=True)
    SGLANG_EXA_NUM_RESULTS = EnvInt(10)
    SGLANG_EXA_SEARCH_TYPE = EnvStr("auto")
    SGLANG_EXA_INCLUDE_HIGHLIGHTS = EnvBool(True)
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)
    SGLANG_ENABLE_RESPONSE_TEMPLATE_TURN_GRAMMAR = EnvBool(True)

    # ===================================================================
    # HiCache storage backends and mmap allocation
    # ===================================================================
    # Per-call cudaHostRegister limit in GB.
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB = EnvInt(256)
    # Base token count for each MLA/DSA dedup broadcast chunk.
    SGLANG_MLA_DEDUP_CHUNK_TOKENS = EnvInt(2048)
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_SKIP_HOST_DUPLICATE_RECLAIM = EnvBool(False)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    # Enable client-side metadata caching to optimize filesystem checks (e.g. for Lustre/NFS/FUSE)
    SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE = EnvBool(False)
    # Positive cache TTL for filesystem metadata lookups (-1 disables positive expiration)
    SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL = EnvFloat(5.0)
    # Buffer mode: staged prefetches pin their device anchor from IO commit
    # to consumption so eviction cannot waste the fetch. Cap = fraction of
    # the pool the pins may hold; 0 disables pinning.
    SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP = EnvFloat(0.5)
    # Buffer mode: store only the SWA window ending at each node (what a
    # prefetch ending there restores) instead of every SWA page.
    SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP = EnvBool(False)
    # Only affects buffer mode: omit intermediate Mamba storage checkpoints.
    # Keep complete request endpoints; unknown/manual backup sources stay eligible.
    SGLANG_HICACHE_MAMBA_REQUEST_END_ONLY = EnvBool(False)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    # >0: prefault host pools with a private THP mapping using this many threads; 0 disables.
    SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS = EnvInt(0)
    # Opt-in: size the Mamba pool's HiCache host share by its conv/temporal
    # checkpoint only, not by the device-only spec-verify scratch and ReplaySSM
    # rings it also holds.
    SGLANG_K31_PF_HICACHE_MAMBA_STATE_BYTES = EnvBool(False)
    # Opt-in, explicit --hicache-size: size every host pool from one per-rank
    # host-memory budget, sampled on all ranks before any pool is built
    # (host-local minimum), instead of re-sampling host memory per pool.
    SGLANG_K31_HICACHE_BUDGET = EnvBoolWithAlias(False, deprecated_name="K31P_HCBUDGET")
    # With SGLANG_K31_HICACHE_BUDGET: malloc_trim(0) before the snapshot, so freed
    # weight-loading heap pages count as available host memory.
    SGLANG_K31_HICACHE_BUDGET_TRIM = EnvBoolWithAlias(
        False, deprecated_name="K31P_HCBUDGET_TRIM"
    )
    SGLANG_TEST_HICACHE_BACKEND_RANDOM_DELAY = EnvFloat(None)
    SGLANG_TEST_HICACHE_BACKEND_RANDOM_FAILURE = EnvFloat(None)

    # ===================================================================
    # KV-transfer staging and Mooncake transport
    # ===================================================================
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # Queue incoming KV scatters between forwards; disabled by default.
    SGLANG_PLB_SCATTER_DEFER = EnvBool(False)
    # Decode stream priority relative to background KV scatters.
    SGLANG_PLB_STREAM_PRIORITY = EnvInt(0)
    # 0: original kernel; >0: persistent CTAs; -1: one CTA per page.
    SGLANG_PLB_SCATTER_CTAS = EnvInt(0)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    # Opt-in limit for the number of KV cache indices represented by one
    # synchronous all-layer Mooncake batch. Set to a positive value to split
    # larger transfers; 0 preserves the legacy single-batch behavior.
    SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES = EnvInt(0)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(True)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)
    SGLANG_FAILED_SESSION_PROBE_MAX_INTERVAL_S = EnvFloat(300.0)
    SGLANG_FAILED_SESSION_IDLE_TIMEOUT_S = EnvFloat(600.0)
    # Opt-in: shard Mooncake prefill transfers across the transfer queues by a
    # SplitMix64 hash of the bootstrap room instead of by the destination
    # sessions' port sum (cherry-pick of sgl-project/sglang#42163).
    SGLANG_K31_PF_MOONCAKE_ROOM_SHARD = EnvBool(False)

    # ===================================================================
    # Mooncake store
    # ===================================================================
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)
    MOONCAKE_TENANT_ID = EnvStr("default")

    # ===================================================================
    # Ascend MemCache (HiCache L3); see https://gitcode.com/Ascend/memcache
    # ===================================================================
    SGLANG_HICACHE_MEMCACHE_CONFIG_PATH = EnvStr(None)
    SGLANG_NPU_MEMCACHE_ENABLE_WARMUP = EnvBool(False)

    # ===================================================================
    # MoRI transport and expert dispatch
    # ===================================================================
    SGLANG_DEEPEP_V2_FORCE_MAX_LEN = EnvBool(False)
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)
    # Unset: EPv2 where moe_hook finds it supported (DeepSeek-V4 only), else EPv1.
    # true forces EPv2; false forces EPv1.
    SGLANG_MORI_EP_V2 = EnvBool(True)
    SGLANG_MORI_DISPATCH_DTYPE = EnvStr("auto")
    SGLANG_MORI_COMBINE_DTYPE = EnvStr("auto")
    SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS = EnvInt(0)
    # Hand AITER only ceil32(sum of per-rank DP tokens) rows of the MORI EP receive
    # buffer; unproved layouts or metadata keep the full view. Forces DP-synced buckets.
    SGLANG_MORI_RECV_BOUND = EnvBool(False)
    # MXFP4 trtllm-gen MoE: deferred-finalize calls above 32 rows pass the finalized
    # output shape, so they hit FlashInfer's tuned tactics (off: only 97-384 rows do).
    SGLANG_OPT_MXFP4_DEFERRED_FINALIZED_SHAPE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Manual MoE input row limit (EPv1 and EPv2); 0 leaves it to SGLANG_MORI_RECV_BOUND.
    SGLANG_MORI_MOE_MAX_INPUT_TOKENS = EnvInt(0)

    # ===================================================================
    # AMD, ROCm, and AITER
    # ===================================================================
    SGLANG_USE_AITER = EnvBool(False)
    # Set by someone bisecting a numerical or performance regression to an
    # online-FP8 linear on gfx1250, to take the Triton tile out of the picture
    # without rebuilding. The tile is on by default because hipBLASLt ships no
    # tuned gfx1250 kernel, so the library GEMM this falls back to is slower
    # than the BF16 path it replaces and, under torch.compile, fails to lower
    # torch._scaled_mm at all. That is the part the code cannot tell you: this
    # is a diagnostic switch, not a supported off-ramp -- to actually stop
    # serving FP8, drop --quantization fp8.
    SGLANG_DISABLE_GFX1250_FP8_GEMM = EnvBool(False)
    # Kill-switch for the gfx115x kernels (Quark int4 W4A16 GEMMs and row pad, target-verify
    # attention, GDN prefill); each also falls back to the generic path where it does not apply.
    SGLANG_DISABLE_GFX115_KERNELS = EnvBool(False)
    # DeepSeek-V4 in-tree copies of open aiter tuning PRs, each applied only while the
    # pinned aiter still ships the pre-PR config (see
    # models/deepseek_common/amd/aiter_intree_tune.py). 0 = use aiter as is.
    # TODO(aiter#6004): remove once ROCm/aiter#6004 is in the aiter pin
    # TODO(aiter#6029): remove once ROCm/aiter#6029 is in the aiter pin
    SGLANG_AITER_INTREE_GEMM_TUNE = EnvBool(True)
    # TODO(aiter#6030): remove once ROCm/aiter#6030 is in the aiter pin
    SGLANG_AITER_INTREE_MHC_TILE_M16 = EnvBool(True)
    # Fuse MiniMax-M3 main/index QK norm + RoPE with main KV and index-K cache
    # insertion. Requires an AITER build with the fp8_e4m3_unit cache contract.
    # Enabled by default for supported configurations; set to 0 to disable.
    SGLANG_M3_USE_AITER_FUSED_QKNORM = EnvBool(True)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Under CUDA-graph capture, stage the aiter custom all-gather / reduce-scatter
    # inputs through the pre-registered IPC pool instead of registering the
    # graph-pool tensors, which lets peers read stale data on replay (NaN in
    # DeepSeek-V4.1 DP attention + TP MoE decode on MI355X). Off by default; the
    # DeepSeek-V4.1 MI355X DP-attention recipe sets it.
    SGLANG_AITER_CAPTURE_COPY_IN = EnvBool(False)
    # The gfx950 TP4 JIT quick all-reduce (quick_all_reduce_hip) runs an INT4 request below this
    # many MiB in INT8, which is as fast at decode sizes with about 1/16 of INT4's error; 0 runs
    # INT4 at every size.
    SGLANG_ROCM_QUICK_REDUCE_INT8_BELOW_MB = EnvInt(64)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    # Quantize the variable-length DP-MoE gather payload (SGLANG_DP_USE_GATHERV
    # path, prefill/extend only) to fp8-e4m3 with per-token-group-128 scales:
    # halves the gathered hidden-state bytes over NCCL; the combine
    # (reduce_scatterv) leg stays bf16 (NCCL SUM cannot run on fp8).  Lossy on
    # the wire — same group quantization the MoE expert GEMMs apply to their
    # input anyway, but router/shared-expert reads see rounded values, so this
    # stays accuracy-gated and default OFF.
    SGLANG_ENABLE_DP_GATHER_FP8 = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # gfx1250 GPT-OSS W4A8 MoE (aiter moe_gemm_a8w4): backend of both expert
    # GEMMs. "" lets aiter choose (gluon on gfx1250); "triton" or "gluon" forces one.
    SGLANG_GFX1250_W4A8_BACKEND = EnvStr("")
    # gfx1250 GPT-OSS W4A8 MoE: GEMM 1 gathers its fp8 activation rows in-kernel
    # instead of a torch gather of num_tokens * topk bf16 rows. Needs
    # SGLANG_GFX1250_W4A8_BACKEND=triton; expert parallelism keeps the torch gather.
    SGLANG_GFX1250_W4A8_KERNEL_GATHER = EnvBool(False)
    # aiter opus moe_sorting dispatch policy (0 auto, 1 oneshot, 2 multi-phase). Auto picks
    # oneshot below ~24 tokens, which on gfx950 costs 11-16 us vs 6-7 us for multi-phase at
    # E=385 / 129; outputs are identical and multi-phase is never slower up to 16384 tokens.
    SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY = EnvInt(2)
    # Fold `silu(gate) * up` into the triton MoE up-GEMM epilogue. W13 rows are
    # permuted in place at load so gate/up land in adjacent columns of the same
    # output tile, which removes intermediate_cache1 and the standalone
    # activation launch per MoE layer. Opt-in because the in-place permute is
    # not compatible with runtime weight updates or EPLB expert rearrangement,
    # both of which assume the checkpoint's halves layout.
    SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED = EnvBool(False)
    # Opt in to Beam's SM103 BF16 TP4/EP1 single-token MoE GEMV.
    # Requires SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.
    SGLANG_OPT_BEAM_DECODE_GEMV = EnvBool(False)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    # Send Qwen3.8's 512-expert softmax router back to aiter `fused_topk`
    # instead of `topk_gating`.
    SGLANG_ROCM_DISABLE_AITER_TOPK_GATING = EnvBool(False)
    # Set this to 1 to stop sglang from clearing an inherited
    # ROCPROFILER_REGISTER_LIBRARY.
    SGLANG_ROCM_KEEP_INHERITED_ROCPROFILER_REGISTER = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    USE_ROCM_AITER_ROPE_BACKEND = EnvStr("0")
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_ROCM_SMALLM_MOE = EnvBool(True)
    SGLANG_ROCM_SMALLM_FP8_BS = EnvBool(True)
    # Quark MXFP4 dense linear on gfx95: use AITER's preshuffled ASM/CK FP4 GEMM
    # (shared a4w4 tuned CSVs) instead of the Triton gemm_afp4wfp4 path.
    SGLANG_ROCM_USE_AITER_FP4_ASM_GEMM = EnvBool(False)
    # Fold the KDA [f_a|b] tail into the wide [q,k,v,g] projection so the whole
    # in-proj is one GEMM. Decode is bandwidth bound there, so the 144 extra
    # output columns ride along nearly free.
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ = EnvBool(True)
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ_MAX_TOKENS = EnvInt(256)
    # Kimi-K3 fused KDA decode on ROCm: "gfx950" (in-tree kernel) or "aiter";
    # any other value keeps the unfused decode.
    SGLANG_ROCM_K3_KDA_FUSED_BACKEND = EnvStr("gfx950")
    # Kimi-K3 FlyDSL operators: "auto" (in-tree, then aiter), "sglang" or "aiter".
    SGLANG_K3_FLYDSL_SOURCE = EnvStr("auto")
    # On gfx95x with SGLANG_USE_AITER=1, quantize the BF16 Qwen3.5/3.8 GDN
    # in_proj_qkvz and out_proj to MXFP4 after loading. 0 keeps them BF16.
    SGLANG_ROCM_GDN_ONLINE_MXFP4 = EnvBool(True)
    # ROCm decode attention kernel: auto (aiter_sparse on gfx950, tilelang elsewhere) |
    # aiter_sparse | tilelang | triton | torch | comparison | unified_kv_triton
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("auto")
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR = EnvInt(8192)
    # Smaller MegaMoEV2 capacity for decode steps; a step with an extend on any
    # DP rank keeps SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR. Must hold one rank's
    # largest decode forward (checked at startup). 0 disables it; otherwise a
    # power of two no larger than SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR.
    SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR = EnvInt(0)
    SGLANG_AMD_FLYDSL_MEGA_QUANT = EnvStr("")
    SGLANG_AITER_MEGA_RANK_SYNC = EnvBool(False)
    SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY = EnvBool(False)
    SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD = EnvBool(False)
    # Above 8192 tokens of context, aiter's non-static workspace is large enough
    # that mem_fraction_static is scaled by 0.85 to leave room for it. Set this to
    # honor an explicitly passed --mem-fraction-static instead. Off by default:
    # the reserve is load-bearing, and skipping it OOMs long-context aiter serving
    # that fits comfortably with it (67.32 GiB request against 47.40 GiB free on a
    # 288 GB MI355 in nightly-4-gpu-mi35x-minimax-m3). Worth setting only when the
    # scaled fraction is itself too small to hold the model weights.
    SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION = EnvBool(False)
    # Route Kimi-K3-style h12 + fp8 MLA decode through aiter Triton Gluon when
    # import and Triton cga_layout prerequisites hold. Set to 0 to force the
    # zero-pad mla_decode_fwd fallback (benchmarking / emergency disable).
    SGLANG_AITER_MLA_GLUON = EnvBool(True)
    # Select the AITER MLA kernel for decode, "asm" or "gluon".
    SGLANG_AITER_MLA_DECODE_BACKEND = EnvStrWithAlias(
        "asm", deprecated_name="SGLANG_AITER_MLA_DCP_DECODE_BACKEND"
    )
    # Select the AITER MLA kernel for target verify, "asm" or "gluon".
    SGLANG_AITER_MLA_VERIFY_BACKEND = EnvStr("asm")
    # Let aiter plan the KV splits for the asm persistent MLA decode.
    SGLANG_AITER_MLA_AUTO_KV_SPLITS = EnvBool(False)
    # Run gfx942 block-FP8 linears with the AITER CK blockscale GEMM instead of Triton.
    SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK = EnvBool(True)

    # DSV4 Aiter flags
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    # Unified KV wired the fused qk-norm-rope kernel to decode only, so MTP
    # target-verify kept running the norm+RoPE as separate kernels. Set to 0 to
    # go back to the unfused chain on the verify path.
    SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY = EnvBool(True)
    # DSPARK: run the target lm_head projection (verify logits) in fp8 via
    # torch._scaled_mm (per-token activation scale + per-output-channel weight
    # scale, weight quantized once and cached). Halves the per-step head GEMM;
    # off by default. Only engages for a floating-point (non-quantized) head.
    SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD = EnvBool(False)
    # Pooled prefill-logits budget of the HIP FP4 indexer, in MiB.
    SGLANG_DSV4_FP4_LOGITS_BUDGET_MB = EnvInt(2048)
    # Split-K for HCA streams in the HIP unified-KV paged decode; 0 keeps the heuristic.
    SGLANG_MLA_HCA_KV_SPLITS = EnvInt(4)

    # ===================================================================
    # Apple Silicon and MLX
    # ===================================================================
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)
    # MLX buffer-cache cap in GB.
    SGLANG_MLX_CACHE_LIMIT_GB = EnvFloat(None)

    # ===================================================================
    # Ascend NPU
    # ===================================================================
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    # Requantize plain block-FP8 (fp32-scale) checkpoints to MXFP8 at load
    # time for arch35 (Ascend 950) Linear and MoE MXFP8 GEMMs. Disabled keeps
    # supported 128x128 FP32-scale checkpoints on native block-FP8 GEMMs.
    # Requantization is lossy (128x128 FP32 scales -> 1x32 UE8M0): enable
    # explicitly for performance after validating workload accuracy.
    SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8 = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    # Kimi-K3 attention-TP shared experts: overlap AG / MLP / RS with the
    # routed front / DeepEP dispatch / routed GEMMs, respectively.
    SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Fuse grouped Kimi-K3 SiTU with valid-row MXFP8 quantization before GMM2.
    # Set to 0 to restore the separate SiTU + npu_dynamic_mx_quant path.
    SGLANG_NPU_MOE_SITU_MXFP8_FUSED = EnvBool(True)
    SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD = EnvBool(False)
    # Use FIAS V2 for DSpark MLA target verify and MHA draft paths. Graph
    # replay requires torch_npu's V2 handler to update actual_seq_kvlen.
    SGLANG_NPU_USE_FIAS_V2_BSND = EnvBool(False)
    # BF16 wo_a: use F.linear for single-local-group decode (Flash TP8),
    # retaining the original weight layout. Opt-in for A/B.
    SGLANG_OPT_NPU_BF16_WO_A_GEMM = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Enable int4x2 weights loading
    SGLANG_NPU_W4A4_NEW_PACKING = EnvBool(False)
    # Opt in to fused MHA KV writes. Disabled by default because ScatterPa
    # can increase memory usage during NPU compiled graph capture.
    SGLANG_NPU_USE_SCATTER_PA_KV_CACHE = EnvBool(False)
    # Use the graph-safe Triton-Ascend kernel for masked speculative KV commits.
    SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE = EnvBool(False)
    # Quantize x to int8 in the dispatch operator (vendor alias consumed by the
    # Ascend DeepEP library; the MTP draft-build scopes override it to False).
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False)
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")
    # Set to 0 to allow AscendAttnBackend to set needs_cpu_seq_lens to False for dsa models.
    SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS = EnvBool(True)

    # ===================================================================
    # MUSA
    # ===================================================================
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # ===================================================================
    # Quantization
    # ===================================================================
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    # Cache BF16 expansions of Hopper group32 FP8 weights for larger GEMMs.
    # Disable to save the additional weight memory or allow online updates.
    SGLANG_OPT_HOPPER_BLOCK_FP8_BF16 = EnvBool(True)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    # Kill switch for fusing dense-MLP SiLU*mul with the NVFP4 down_proj input quant.
    SGLANG_DISABLE_SILU_FP4_QUANT_FUSION = EnvBool(False)
    # GLM NextN (MTP): cast the draft layer's bf16 fused MoE to per-channel FP8
    # on load. Unrelated to the NVFP4 block-FP8 NextN path above.
    SGLANG_GLM_NEXTN_MOE_PTPC = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    # HIP: convert only MXFP8 dense linears to block-fp8; fused MoE stays MX 1x32,
    # since the block-scale MoE kernel lacks SwiGLU-OAI
    SGLANG_FORCE_MXFP8_BLOCK_CONVERT_DENSE = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")

    # ===================================================================
    # Humming quantization
    # ===================================================================
    SGLANG_HUMMING_ONLINE_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_INPUT_QUANT_CONFIG = EnvJSON(None)
    SGLANG_HUMMING_USE_F16_ACCUM = EnvBool(False)
    SGLANG_HUMMING_MOE_GEMM_TYPE = EnvStr("")

    # ===================================================================
    # FlashInfer, FlashMLA, and TRT-LLM
    # ===================================================================
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Force FlashInfer decode on (1) or off (0) tensor cores; unset picks from
    # the KV cache dtype and the GQA group size.
    SGLANG_FLASHINFER_USE_TENSOR_CORE = EnvBool(None)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Per-rank dispatch capacity of the FlashInfer MoE A2A dispatcher. Unset
    # sizes it from the per-rank prefill chunk, with a 4096 floor.
    SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(None)
    # FlashInfer MegaMOE (generic moe_ep.MoEEpMegaLayer backend). Sizes the
    # per-rank symmetric workspace; must be >= the largest padded per-rank batch
    # (derived from cuda_graph_max_bs / chunked_prefill_size when unset).
    SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK = EnvInt(0)
    # Opt-in in-kernel FC2 top-k reduce (cross-rank REDG atomic-add) for the
    # cutedsl mega kernels (NVFP4 W4A4 / W4A16 and MXFP8). Permits in-kernel
    # reduction; FlashInfer's tuning profile may still select external reduction.
    # BF16 atomic accumulation is nondeterministic -- keep off for
    # bit-reproducibility.
    # No effect on the DeepGEMM (block-FP8) mega path, which lacks the knob.
    SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE = EnvBool(False)
    # Cross-rank combine wire format for the FlashInfer NVFP4 cutedsl MegaMOE
    # kernel. "bf16" is exact/default; "mxfp8" and "nvfp4" reduce combine
    # traffic with a small accuracy tradeoff and require FC2 reduce outside the
    # kernel.
    SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE = EnvStr("bf16")
    # Enable per-token FP32 activation scaling for serialized ModelOpt FP4 with
    # FlashInfer TRT-LLM or CuTe DSL v2 MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # Use BF16 activations with FlashInfer CuTe DSL NVFP4 dense and MoE weights,
    # including the flashinfer_megamoe runner (BF16 combine).
    SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16 = EnvBool(False)
    # Launch the TRT-LLM MoE grouped GEMMs with PDL only at or below this
    # token count.
    SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS = EnvInt(8192)
    # TRT-LLM MLA workspace in MB.
    SGLANG_TRTLLM_MLA_WORKSPACE_MB = EnvInt(_k31_gb_best(256, 150))
    # Use FlashInfer's fused atomic CUTLASS/CuTe DSL MoE finalize.
    SGLANG_FLASHINFER_MOE_FUSED_FINALIZE = EnvBool(False)
    # Experimental Nemotron small-M NVFP4 path; validated only on SM120.
    SGLANG_OPT_SM120_NVFP4_PHASE_LAYOUT = EnvBool(False)
    SGLANG_OPT_SM120_MTP_ROUTER_TOP22 = EnvBool(False)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # Small-batch CSGMV split-K changes rounding. Disabled in deterministic mode.
    SGLANG_ENABLE_LORA_CSGMV_SPLIT_K = EnvBool(False)
    # Opt in to the dual decode graph for triton_v2 LoRA: a base-only graph for
    # adapter-free batches next to the adapter graph (two graphs per batch
    # size); off by default.
    SGLANG_ENABLE_LORA_DUAL_GRAPH = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # Split TRTLLM-GEN decode attention into sorted, equal-size request groups.
    # One preserves the default single-call path; values above one are useful
    # for batches whose KV sequence lengths have a large spread.
    SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS = EnvInt(1)
    # Split the KV of each (request, KV head) of a full-attention layer across at
    # least this many CTAs in TRTLLM-GEN fixed-q-len decode/verify (not ragged
    # verify), at batch sizes >= MIN_BS. FlashInfer stops splitting once
    # batch * kv_heads exceeds half the SMs. 0 keeps FlashInfer's choice.
    SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS = EnvInt(0)
    # Measured with 4 KV heads per rank on one GPU type; re-tune elsewhere.
    SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS_MIN_BS = EnvInt(12)
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
    # Store DeepSeek-V4 SWA KV directly in FlashInfer's 64-token SM120 page
    # layout. The scheduler continues to allocate 256-token logical pages.
    SGLANG_OPT_SM120_DIRECT_SWA_KV = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    # Also autotune one EXTEND-shaped dummy at max_prefill_tokens during
    # warmup. Opt-in: the extra forward needs transient activation headroom
    # that small-VRAM or tightly-packed configs may not have.
    SGLANG_FLASHINFER_AUTOTUNE_EXTEND = EnvBool(False)
    SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE = EnvBool(False)
    SGLANG_OPT_FLUX3_PRIMS_TS_TUNING_BUCKETS = EnvTuple(tuple())
    SGLANG_USE_SAGE_ATTENTION_IMPL = EnvStr("")

    # ===================================================================
    # Triton and Torch compilation
    # ===================================================================
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)
    # A-B kill-switch for Work-Centric (Lean) Attention. When True, forces the
    # standard Triton decode kernel even if --enable-lean-attention or the auto-gate
    # would select Lean. Used to isolate the Lean kernel in benchmarks.
    SGLANG_DISABLE_LEAN_ATTENTION = EnvBool(False)
    # Persistent-grid size multiplier for the Lean decode kernel:
    # total_programs = round(device_CU_count * this). Default 1.0 (one CTA per CU), which
    # maximizes KV work-tiles per CTA and minimizes the cross-CTA combine/atomic reduction.
    # Kernel + E2E A/B sweeps found 1.0 beats 2.0 across uniform and ragged configs on both
    # MI300X (gfx942) and MI355X (gfx950) — 2.0 oversubscribed the CUs and regressed high-batch
    # decode. Exposed as a knob (e.g. set 2.0) for grid A/B tuning without a rebuild.
    SGLANG_FORCE_LEAN_GRID_CU_MULT = EnvFloat(1.0)

    # Torch Compile
    # Compact extend-attention query-tile grid (parity with flash-attn's
    # ragged-aware launch). HIP only (ignored elsewhere), and used only when it
    # launches fewer tiles than the legacy grid; 0 turns it off.
    SGLANG_TRITON_COMPACT_EXTEND_ATTENTION = EnvBool(True)
    # Raise if Triton loads a kernel after the engine starts serving. This
    # verifies that startup warmup covers every kernel specialization used at
    # serving time.
    SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY = EnvBool(False)
    SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS = EnvFloat(1.0)
    SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB = EnvFloat(1.0)
    # gfx950 MLA decode stage-1: pick the launch geometry and split count per batch.
    # Reorders the fp32 accumulation, so off by default.
    SGLANG_MLA_DECODE_TUNE = EnvBool(False)
    # Native FP8 prefill for exact gfx950 Kimi-K3 zero-prefix and absorbed
    # cached-prefix shapes. Validated at 98% GSM8K accuracy.
    SGLANG_TRITON_FP8_PREFILL_ATTN = EnvBool(True)
    # CUDA absorbed-MLA extend (12 heads, 576/512): value-column programs per head,
    # and leading prefix tiles scored in parallel; both exact, 1 / 0 = stock launch.
    SGLANG_TRITON_ABSORBED_MLA_DV_SPLIT = EnvInt(1)
    SGLANG_TRITON_ABSORBED_MLA_PREFIX_TILES = EnvInt(0)
    # Triton target-verify metadata: a length copy instead of a bs = 1 cumsum,
    # and no mask indptr without a custom mask; the kernels read the same values.
    SGLANG_OPT_TRITON_VERIFY_METADATA_TRIM = EnvBool(False)
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)
    SGLANG_TORCH_DYNAMIC_SHAPE = EnvBool(False)
    # torch.compile backend for CPU graphs (defaults to "inductor"; "eager" or "aot_eager" for debugging)
    SGLANG_CPU_GRAPH_COMPILE_BACKEND = EnvStr("inductor")
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # ===================================================================
    # Expert parallel load balancing
    # ===================================================================
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    # Chunk size for the rebalance expert-weight P2P exchange; set
    # >= num_physical_experts to submit a single batch_isend_irecv.
    SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE = EnvIntWithAlias(
        32, deprecated_name="SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE"
    )

    # ===================================================================
    # DeepGEMM
    # ===================================================================
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT = EnvBool(False)
    SGLANG_DEEPGEMM_STANDARD_LAYOUT = EnvStr("auto")
    SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION = EnvFloat(0.25)
    # Cap the DeepGEMM masked grouped-GEMM per-expert padded capacity at
    # round_up(max(masked_m), 256) instead of round_up(rank_tokens, 256):
    # shrinks the [num_local_experts, m, *] MoE intermediates ~4x under
    # load imbalance (they otherwise OOM saturated --moe-runner-backend
    # deep_gemm serving).  Costs one D2H sync per MoE layer.
    SGLANG_OPT_DG_MASKED_M_CAP = EnvBool(False)
    # Wide-DP eager prefill uses compact routing storage; masked storage scales
    # with num_local_experts and can OOM on skewed batches.
    SGLANG_OPT_DG_COMPACT_EAGER = EnvBool(False)
    # Drop dp-attention MAX_LEN pad rows from MoE dispatch (StandardDispatcher
    # post-translation topk_ids -> -1): pad rows otherwise run the router on
    # stale hidden values and burn expert compute whose outputs are discarded;
    # colliding pad top-ks also inflate the DeepGEMM masked-GEMM workspace to
    # OOM at saturation.  Capture-safe (reads only global_num_tokens_gpu).
    SGLANG_OPT_MASK_DP_PAD_MOE = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    # Resolved lazily so it tracks SGLANG_CACHE_DIR, which is defined below.
    SGLANG_DG_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_ENABLE_FP8_BLOCK_GEMV = EnvBool(False)
    SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT = EnvBool(False)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # ===================================================================
    # Cache directories
    # ===================================================================
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    # Persistent CuTe DSL AOT objects. Resolved lazily so it tracks
    # SGLANG_CACHE_DIR; set to an empty string to keep compilation
    # process-local. Must be trusted: cached objects are loaded into the process.
    SGLANG_CUTE_AOT_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("cute_aot"))

    # ===================================================================
    # Kernel development: JIT build cache, diagnostics and benchmarks
    # ===================================================================
    # Everything here is a developer knob for working ON kernels -- building
    # them, inspecting what the compiler produced, and benchmarking them. Flags
    # that select a kernel in production live with their own feature instead.
    # JIT kernel build cache. None = unset, resolving to ~/.cache/sglang/jit;
    # point it at a persistent mount to share builds across CI jobs.
    SGLANG_JIT_CACHE_DIR = EnvStr(None)
    # Log, at INFO, which dependency changed whenever a module is rebuilt.
    SGLANG_JIT_CACHE_DEBUG = EnvBool(False)
    # How many builds to keep per module variant. None = unset = keep all, which
    # is what makes reverting an edit an instant hit instead of a rebuild; set
    # it to trade that away for disk (1 keeps only the most recent build).
    SGLANG_JIT_CACHE_KEEP = EnvInt(None)
    # Skip the cache lookup and run the compiler for every module this process
    # loads. The result is still published, so the cost is one rebuild per
    # module, not one per load.
    SGLANG_JIT_FORCE_RECOMPILE = EnvBool(False)
    # Raise instead of compiling when a module misses the cache, so a
    # deployment that expects a pre-seeded cache fails loudly at startup
    # rather than silently eating a cold compile.
    SGLANG_CRASH_ON_JIT_COMPILE = EnvBool(False)
    # Ask the device compiler for per-kernel resource usage (registers, spills,
    # shared memory) and log it at INFO. Changes the build flags, so it compiles
    # into its own cache entry and leaves the normal one alone -- but that entry
    # is a hit on the second run, and a cache hit has nothing to report, so pair
    # this with SGLANG_JIT_FORCE_RECOMPILE to see the report every time.
    SGLANG_JIT_LOG_RESOURCE_USAGE = EnvBool(False)
    # Drop the GB/s and TFLOPS columns from the benchmark marker's table.
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_BANDWIDTH = EnvBool(False)
    SGLANG_JIT_BENCHMARK_DISABLE_LOG_FLOPS = EnvBool(False)
    # The hgcc --gpu-architecture target used by JIT builds on SAIL (hgcc
    # cannot read CUDA sources, so they are converted by sailify first).
    # One of: native, all, all-ppu,
    # all-vm, ppu_10, ppu_15, vm_10, vm_15. "native" builds for the local
    # device only, matching how the nvcc path pins -gencode to this GPU.
    SGLANG_SAIL_GPU_ARCH = EnvStr("native")

    # Per-file overrides for packaged MoE LoRA plans, tiles, and base GEMM configs.
    SGLANG_LORA_MOE_CONFIG_DIR = EnvStr(None)
    # Per-file overrides for dense LoRA plan tables.
    SGLANG_LORA_DENSE_CONFIG_DIR = EnvStr(None)
    # Root of the csgmv_configs tuning tables; defaults to the packaged directory.
    SGLANG_LORA_CONFIG_DIR = EnvStr(None)

    # ===================================================================
    # Expert-parallel dispatch and MoE execution
    # ===================================================================
    # Deprecated in favor of '--deepep-dispatcher-output-dtype bf16' but still
    # read by several call sites; do not use in new code.
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # Per-rank buffer capacity, not a model token limit.
    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    # 0 lets ElasticBuffer select its theoretical communication SM/QP counts.
    SGLANG_DEEPEP_V2_NUM_SMS = EnvInt(0)
    SGLANG_DEEPEP_V2_ENABLE_PREFILL_EXPAND = EnvBool(None)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    SGLANG_ENABLE_QWEN_DEEPEP_SHARED_OVERLAP = EnvBool(True)
    # Force dynamic Waterfill with runtime EP all-reduce instead of the default
    # static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(True)
    # The [M*top_k, hidden] HBM round trip pays only at small M; 0 disables.
    # GLM-5.2 (H=6144): neutral at 192 on B300 TP8, crossover 192-223 on GB300 TP4.
    SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS = EnvInt(192)
    # Let the CuTe DSL fused MoE finalize consume expert-parallel handoffs.
    # Unset follows the NVIDIA runtime master; an explicit value wins.
    SGLANG_OPT_CUTEDSL_EP_DEFERRED_FINALIZE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DeepSeek/GLM MoE (deepseek_v2.py): quantize the (dp-gathered) MoE input
    # to per-token-group-128 fp8 ONCE and feed both the fused shared-expert
    # GEMM (cutlass w8a8 linear) and the routed experts' triton fused runner,
    # instead of quantizing the same [T, hidden] tensor twice with different
    # scale layouts. Only engages on CUDA with fp8 block-128 weights, the
    # standard dispatcher, and the triton MoE runner; falls back silently
    # otherwise.
    SGLANG_OPT_MOE_QUANT_ONCE = EnvBool(False)

    # ===================================================================
    # DeepGEMM Mega MoE
    # ===================================================================
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(8192)
    # Blackwell MegaMoE uses a whole-grid software barrier. Keep a small
    # residency margin so every cluster can launch beside other streams.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS = EnvInt(2)
    SGLANG_OPT_DEEPGEMM_MEGA_GATE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_LOCALIZE = EnvBool(False)
    # Opt-in: raise DeepGEMM's BLOCK_M (16/32 -> 64, 64 -> 96) for FP8xFP4 calls in
    # decode and target-verify CUDA graphs.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_RAISE_DECODE_BLOCK_M = EnvBool(False)

    # ===================================================================
    # Top-k kernels
    # ===================================================================
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    # Opt-in: route DeepSeek-V3 grouped topk through the unified Triton router
    # instead of the flashinfer/AOT grouped kernels. Off by default (flashinfer is
    # the tuned production path); the Triton path is bit-exact on DeepSeek-V3.2 e2e
    # and benchmarks at parity, so this is a consolidation escape hatch, not a perf flip.
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK = EnvBool(False)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)
    # Exactness of the top-k v2 kernel, compiled in once per process: 0 fast (an
    # overflowing threshold bin keeps an arbitrary subset; opt-in), 1 exact (values
    # equal torch.topk's; the default on every platform), 2 exact with the same
    # selected set on every run.
    SGLANG_TOPK_V2_MODE = EnvInt(1)

    # ===================================================================
    # Kernel selection and fused backends
    # ===================================================================
    # MiniCPM sparse attention developer switches
    SGLANG_MINICPM_FUSE_TOPK = EnvBool(False)
    SGLANG_MINICPM_DENSE_AS_SPARSE = EnvBool(False)
    SGLANG_MINICPM_FORCE_DENSE = EnvBool(False)

    # Force every sglang.kernels BaseFusedOp onto one backend (a KernelBackend
    # value, e.g. "torch" / "torch_compile" / "triton" / "helion" / "aot"); unset =
    # auto-select by priority. "torch" flips all fused ops to their pure-torch
    # reference implementations for numerical-bug bisection.
    SGLANG_FORCE_FUSED_OP_BACKEND = EnvStr(None)
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)

    # ===================================================================
    # Logits and log-probability processing
    # ===================================================================
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)
    SGLANG_ENABLE_LOGPROB_CHUNK = EnvBool(True)
    SGLANG_LOGPROB_CHUNK_SIZE = EnvInt(2048)
    # Compute input logprobs from logits via per-row logsumexp instead of
    # materializing the full-vocab log-softmax. Escape hatch only; the two
    # paths are mathematically identical.
    SGLANG_ENABLE_FAST_INPUT_LOGPROBS = EnvBool(True)

    # ===================================================================
    # Deterministic inference and all-reduce
    # ===================================================================
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    # NCCL channel count pinned on CUDA so the all-reduce reduces a token the
    # same way whatever else shares its batch. Raise it to buy back bandwidth
    # on links that can drive more channels.
    SGLANG_DETERMINISTIC_NCCL_NCHANNELS = EnvInt(8)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    # Default per-direction workspace cap for CustomAllReduceV2; explicit
    # constructor sizes take precedence over this.
    SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB = EnvInt(_k31_gb_best(32 * 1024, 16 * 1024))
    # Largest M the startup bf16 GEMM warmup covers (the prefill chunk; about
    # 13 s of startup on K3.1); 0 disables it.
    SGLANG_WARMUP_BF16_GEMM_MAX_M = EnvInt(16384)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
    SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)
    # See docs/references/environment_variables.
    SGLANG_ENABLE_PCIE_IPC_ALLREDUCE = EnvBool(False)
    SGLANG_PCIE_IPC_MAX_NUMEL = EnvInt(0)

    # ===================================================================
    # RoPE cache
    # ===================================================================
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_FP32 = EnvBool(False)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # ===================================================================
    # Speculative decoding
    # ===================================================================
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
    # Experimental: allow pipeline parallelism x speculative decoding
    # (EAGLE/MTP). Off by default; see the PP+spec RFC for constraints
    # (non-overlap schedule, no DP attention).
    SGLANG_ENABLE_PP_SPEC = EnvBool(False)
    SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION = EnvBool(False)
    # Capture the per-replay attention-metadata prep (init_forward_metadata_out_graph)
    # into a small CUDA graph, collapsing its host dispatch cost to one launch.
    # Experimental; auto-falls back to eager if the backend's prep is not capturable.
    SGLANG_ENABLE_METADATA_GLUE_GRAPH = EnvBool(False)
    # Experimental: capture the EAGLE verify's sampling, acceptance and state commit
    # in CUDA graphs (speculative/eagle_post_verify_graphs.py).
    SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Let an unchanged top-k-1 EAGLE round read the previous round's top-1 and hidden
    # rows where the draft-extend graph left them instead of relaying copies.
    SGLANG_OPT_SPEC_RELAY_IN_PLACE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Runtime optimizations that follow SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION. Each
    # still accepts its former name (deprecated_name) until the alias is removed.
    # The fused small-field copy gives each field its own blocks and moves aligned fields in 16 B units.
    SGLANG_OPT_SMALL_FIELD_COPY_VECTORIZED = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_BK_VEC_COPY",
    )
    # Verify input staging, bit-identical: reuse last round's mamba track indices while their
    # inputs are unchanged, and skip the replay-time input_ids / positions copies that the
    # same-stream pre-planned load_batch already made.
    SGLANG_OPT_SPEC_VERIFY_REUSE_STAGED_INPUTS = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_MKH_N3B",
    )
    # The post-verify graphs' per-round input load as one small_field_copy launch that also
    # reads the injected acceptance length from its pinned host slot (bit-identical); needs
    # SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.
    SGLANG_OPT_SPEC_POST_VERIFY_ONE_LAUNCH_LOAD = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_MKH_N3C",
    )
    # A topk-1 chain against TRT-LLM MHA, which masks the draft window itself, is built
    # with build_chain_tree instead of a full-context tree mask.
    SGLANG_OPT_TRTLLM_MHA_CHAIN_VERIFY = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_CHAIN_TREE",
    )
    # A verify whose requests all sample with a finite top-k within the compact limit
    # samples on that support (eagle_sample's compact branch), graphed or eager.
    SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING",
    )
    # A sampled verify result (predict, accept_index, num_correct_drafts) is one int32
    # buffer, broadcast to the TP group once.
    SGLANG_OPT_SPEC_PACKED_VERIFY_BCAST = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST",
    )
    # The scheduler's per-pass TP syncs (request broadcast size, HiCache ready counts)
    # through a shared-memory table when the group shares a host (shm_tag_exchange.py).
    SGLANG_OPT_SCHEDULER_SHM_SYNC = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_SHM_SYNC",
    )
    # With SGLANG_OPT_SCHEDULER_SHM_SYNC, a group across hosts exchanges per host and only
    # the host leaders meet (HierTagExchange); with two hosts, optionally over one TCP link.
    SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_SHM_SYNC_MULTIHOST",
    )
    SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP = EnvBoolWithAlias(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get,
        deprecated_name="SGLANG_Q35MF_EDGE_MULTIHOST_TCP",
    )
    # PD decode optimizations follow the shared switch unless explicitly disabled.
    SGLANG_ENABLE_NV_MEGA_PDDEC_RUNTIME_OPTIMIZATION = EnvBool(True)
    # Reuse the resolved CPU sequence lengths for FlashInfer draft planning,
    # avoiding a second blocking readback after generating the same indptr.
    SGLANG_OPT_FLASHINFER_DRAFT_HOST_INDPTR = EnvBool(False)
    SGLANG_OPT_FLASHINFER_VERIFY_HOST_PLAN = EnvBool(False)
    SGLANG_OPT_FUSED_KDA_VERIFY = EnvBool(False)
    # A/B: keep the DFLASH draft greedy head eager (not folded in-graph).
    SGLANG_DFLASH_EAGER_DRAFT_SAMPLER = EnvBool(False)
    # Disable the compact-slot SWA window lookup for compact MLA drafts; off by default.
    SGLANG_DISABLE_DFLASH_COMPACT_SWA_WINDOW = EnvBool(False)
    # Keep an all-sliding-window MLA DFLASH draft's KV in a per-request ring of
    # window + verify-block rows instead of one row per target slot. On by default;
    # resolved at startup to off for drafts or backends it does not fit.
    SGLANG_DFLASH_DRAFT_KV_RING = EnvBool(True)
    # Gather DFLASH candidate top-k in one multicast all-gather.
    SGLANG_OPT_DFLASH_FUSED_CANDIDATE_AG = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Compute a draft block's ring locs in one Triton launch; off by default.
    SGLANG_OPT_DFLASH_RING_LOCS_FUSED = EnvBool(False)
    # Capture the DFLASH aux append (draft KV) as CUDA graphs.
    SGLANG_DFLASH_AUX_GRAPH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Build the DFLASH Mamba commit indices in one kernel on preallocated buffers.
    SGLANG_DFLASH_COMMIT_METADATA = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Replay the DFLASH aux graph on a side stream over the draft prep.
    SGLANG_OPT_DFLASH_AUX_OVERLAP = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Skip the SWA draft's LSE and zero-KV fixup when DCP is off.
    SGLANG_OPT_DFLASH_SWA_SKIP_LSE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Defer the TP gather of an unsoftcapped target-verify vocab shard; off by default.
    SGLANG_DFLASH_DEFER_PLAIN_GATHER = EnvBool(False)
    # Capture the megakernel post-verify tail as CUDA graphs; off by default.
    SGLANG_OPT_DFLASH_MEGAKERNEL_TAIL_GRAPH = EnvBool(False)
    # Split the DFLASH draft's context fc across TP by output columns.
    SGLANG_OPT_DFLASH_TP_AUX_FC = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Prefetch the DFLASH draft's small tensors into L2 in its graph.
    SGLANG_OPT_DFLASH_DRAFT_L2_PREFETCH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # Batch DFLASH decode's small device copies into one SM kernel.
    SGLANG_OPT_DFLASH_BATCHED_SMALL_COPY = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get
    )
    # With AUX_OVERLAP: create the aux side stream with the highest stream priority (its CTAs are dispatched ahead of pending
    # lower-priority work, e.g. the side-stream KDA commit kernel). Scheduling only: same kernels, same values.
    SGLANG_OPT_DFLASH_AUX_STREAM_HIGH_PRIORITY = EnvBool(False)
    # Opt-in: the DFLASH verify's post-section (acceptance, target mamba state
    # commit on a side stream, draft-KV append) in per-batch-size CUDA graphs
    # (speculative/dflash_post_verify_graphs.py).
    SGLANG_DFLASH_GRAPH_POST_VERIFY = EnvBool(False)
    # With SGLANG_DFLASH_GRAPH_POST_VERIFY: shard the draft context fc across TP.
    SGLANG_DFLASH_SHARD_DRAFT_FC = EnvBool(False)

    # DFLASH draft, context-stage and commit switches (NVIDIA K3.1 host); all default off.
    # DFLASH decode: the post-verify Mamba/KDA commit runs on a side stream,
    # beside the draft-KV append; the step joins it before returning.
    SGLANG_OPT_DFLASH_COMMIT_FORK = EnvBool(False)
    # With COMMIT_FORK: the step does not join the forked commit; the next target verify (or the next extend / idle
    # forward) joins it, so the draft prep and the draft run beside it. The commit reads a copy of its state indices
    # made before the fork, and the result read-back (copy_done) waits for it. Engages only on the overlap scheduler with
    # the KDA ReplaySSM exact-fold commit and the extra_buffer radix strategy, without unified memory, HiCache, PD or PP.
    SGLANG_OPT_DFLASH_COMMIT_JOIN_LATE = EnvBool(False)
    # DFLASH decode aux stage split in three graphs (dflash_shadow_ctx.py): the fc shard GEMM, gather, hidden norm and
    # context kv_a GEMM read only the verify's rows, so they run before the accept; only the context KV store waits for
    # it. 1 = the pre-accept part after the accept's TP sync; 2 = the fc GEMM at verify end, the rest after that sync.
    # Engages only with the served context fusion (fc shard, context GEMM, store norm fold) on the aux side stream.
    SGLANG_OPT_DFLASH_SHADOW_CTX = EnvInt(0)
    # SHADOW_CTX's largest aux batch (rows) split in three graphs; 0 = 64 (the original decode cap). Never above the fc
    # shard's own cap (FC_SHARD_MAX_ROWS, else CTX_FUSED_MAX_ROWS): A2 needs the shard's gather. Same kernels, same bits.
    SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS = EnvInt(0)
    # Test hook for SHADOW_CTX. CHECK: the first N split rounds compare the stored draft KV bytes with the eager
    # stock stage on the same inputs (committed rows) and with a snapshot (rejected rows).
    SGLANG_TEST_DFLASH_SHADOW_CTX_CHECK = EnvInt(0)
    # DFLASH MLA draft: skip the per-forward zeroed fp8-scale buffer;
    # unquantized drafts never take one, and a request for one raises.
    SGLANG_OPT_DFLASH_MLA_DRAFT_SKIP_ZERO_ALLOC = EnvBool(False)
    # DFLASH draft at one request per step: the layers' all-reduces and the selector's
    # top-k gather use one-shot epoch-tagged mailboxes (dflash_draft_collectives.py).
    SGLANG_OPT_DFLASH_DRAFT_PK3_ALL_REDUCE = EnvBool(False)
    SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER = EnvBool(False)
    # With the packed gather: also take it at every draft batch size up to this many requests (one mailbox per
    # batch size); 1 keeps it at one request, other batch sizes use the group's all-gather.
    SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER_MAX_BS = EnvInt(1)
    # DFLASH context stage, each bitwise to stock (speculative/dflash_context_fusion.py): column-sharded
    # draft fc + multimem gather; all MLA layers' context kv_a in one GEMM; RoPE + FP8 store in one kernel.
    SGLANG_OPT_DFLASH_FC_SHARD = EnvBool(False)
    SGLANG_OPT_DFLASH_CTX_KV_GEMM = EnvBool(False)
    # Row cap of the fc shard and the context GEMM (default 64 = decode B8). Wide draft batches (graphs to 16-64 requests, 128-512
    # rows) raise it to their largest graph rows; each row count still pins only an algorithm bitwise equal to F.linear.
    SGLANG_OPT_DFLASH_CTX_FUSED_MAX_ROWS = EnvInt(64)
    # Separate, lower row cap for the fc shard alone (0 = the cap above). Above ~96 rows the sharded fc + multimem gather loses
    # to the replicated fc while the context GEMM keeps winning (TP4 microbenchmark), so wide batches cap the fc lower.
    SGLANG_OPT_DFLASH_FC_SHARD_MAX_ROWS = EnvInt(0)
    SGLANG_OPT_DFLASH_CTX_KV_STORE = EnvBool(False)
    # With CTX_KV_STORE: the layers' kv_a RMSNorms folded into the store kernel (flashinfer's H = 512 BF16 order); accepted
    # only after a bitwise probe against the served norm modules. Test hook: CHECK = compare the stored k_nope FP8 bytes with
    # the served norm + cast after each of the first N aux replays.
    SGLANG_OPT_DFLASH_CTX_KV_STORE_NORM = EnvBool(False)
    SGLANG_TEST_DFLASH_CTX_KV_STORE_NORM_CHECK = EnvInt(0)
    # DFLASH draft: the draft grouped convs and the draft layers' input and post-attention RMSNorms trigger their PDL dependents
    # before their own wait, so the GEMMs after them launch and stage weights earlier (q_a / kv_a norms: routed, but their inputs fail
    # the wrapper's 2-D contiguous input check, so they run the served kernel; the final norm is not routed). One
    # int32 device flag, fixed at 1, enables the trigger; arithmetic unchanged. The convs need
    # SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE.
    SGLANG_OPT_DFLASH_DRAFT_EARLY_TRIGGER = EnvBool(False)
    SGLANG_OPT_DFLASH_COMPACT_WINDOW_FUSED = EnvBool(False)
    # DFLASH MLA draft: draft-layer GEMMs of <= 8 rows on the megakernel plugin's
    # split-K projection (an FP32 accumulation reorder of the stock GEMMs).
    SGLANG_OPT_DFLASH_DRAFT_SPLITK = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: the MLP's SiLU-mul in the gate_up split-K epilogue (bitwise to the
    # split-K gate_up + stock act_and_mul pair; one kernel fewer per draft layer).
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: also the 9-64-row draft blocks (batches of 2-8 requests) whose row count has a
    # tactic in dflash_draft_splitk.WIDE_TACTICS, instead of the stock GEMMs (and, with _ACT, the fused SiLU-mul).
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: the 96-row draft blocks (12 requests x block 8) of the K=7168 linears (conv kernel
    # projections, fused qkv_a, unfused gate_up) on SK3B, a streaming cluster split-K (weights through their own shared-memory
    # ring from kernel entry, x after the PDL wait; speculative/dflash_sk3b.py, dflash_draft_splitk.SK3B_TACTICS), bitwise to the
    # stock cuBLAS GEMM at that row count.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT: the fused gate_up + SiLU-mul at the dflash_draft_splitk.SK3B_ACT_TACTICS row counts (96)
    # on SK3B with an act epilogue (speculative/dflash_sk3b_act.py):
    # bitwise to the same SK3B gate_up + the stock act, so bitwise to the stock cuBLAS gate_up + act at 96 rows; saves the act kernel.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_ACT = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B (and _SK3B_ACT): the SK3B rows of the fused qkv_a and gate_up projections also prefetch
    # the next GEMM's weight (o_proj, down) into L2 after their own weight stream (dflash_draft_splitk.PF_FLAG). A cache hint: outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B: the conv kernel projections of the draft's grouped convs at their
    # SK3B row counts run with the SIDE-0 grouped conv in the projection's epilogue (speculative/dflash_sk3b_conv.py): the
    # projection output is SK3B's and the conv output bitwise to the stock Triton grouped conv of it; saves one conv kernel per sublayer.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: the 24-88-row draft blocks (3-11 requests x block 8) of the K=7168 linears on SK3B too
    # (dflash_draft_splitk.SK3B_LOW_TACTICS; with _SK3B_ACT also the fused gate_up + SiLU-mul, with _SK3B_CONV the conv epilogue): bitwise to the
    # stock cuBLAS GEMM at these rows; these rows take precedence over SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE's and
    # SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS' rows.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_LOW = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: the 104-128-row draft blocks (13-16 requests x block 8) of the K=7168 linears on SK3B with split 4
    # (dflash_draft_splitk.SK3B_B4_TACTICS; with _SK3B_ACT also the fused gate_up + SiLU-mul, with _SK3B_CONV the conv epilogue): the stock cuBLAS
    # GEMM's four K slices and rank-order FP32 sum at these rows -> bitwise to it.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_B4 = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK (16 rows also _OV_WIDE_ROWS with 16): the conv kernel projections of the draft's grouped convs at
    # 8 rows (split-K (128, 8, 7, 5)) and 16 rows (OV-wide (64, 16, 4, 11)) run with the SIDE-0 grouped conv in the projection's epilogue
    # (speculative/dflash_splitk_conv.py): same tactic, same reduction -> the projection output is the served one
    # bit for bit and the conv output bitwise to the stock Triton grouped conv of it; saves one conv kernel per sublayer. SK3B rows keep SK3B.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK_WIDE: at these draft-row counts (comma list, e.g. "16,24,32" = 2 / 3 / 4 requests at
    # 7 drafted; each needs an entry in dflash_draft_splitk.OV_WIDE_TACTICS) the K = 7168 wide projections (conv kernel
    # projections, fused qkv_a, gate_up and the fused gate_up + SiLU-mul) keep their wide tactic's tiles, split and trigger
    # but hold more pipeline stages (fewer than their K tiles) in the plugin's extended shared-memory mode. "" = off. Outputs
    # unchanged (same partial sums, same reduction order).
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV_WIDE_ROWS = EnvTuple(())
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK: the <= 8-row (1792, 7168) and (2112, 7168) draft GEMMs (conv kernel projections,
    # gate_up, fused qkv_a) take a tactic with as many stages as K tiles per split, in the extended shared-memory mode, so every weight
    # tile is issued before the PDL wait (dflash_draft_splitk.OV_TACTICS). Same split, same sum: bitwise to off. The conv
    # projections trigger after their wait (SplitKGemm fenced mode 2), as the draft AR site's early trigger requires.
    SGLANG_OPT_DFLASH_DRAFT_SPLITK_OV = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITK and the draft AR sites, <= 8 rows: each draft sublayer's grouped conv (side 0) runs
    # inside the next GEMM (fused qkv_a, gate_up or the fused gate_up + SiLU-mul): the GEMM loads the raw rows and convolves
    # its B stages in shared memory before the MMA (speculative/dflash_draft_conv_prologue.py). Same arithmetic as the
    # served conv kernel then the served projection: bitwise to off.
    SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE: only the MLP conv (side 0) runs inside gate_up (or the fused gate_up +
    # SiLU-mul); the attention conv and fused qkv_a keep the served conv kernel and projection, where the folded form
    # costs more than the conv kernel it removes. Bitwise to off.
    SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE_MLP_ONLY = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_MLAQ and SGLANG_OPT_DFLASH_DRAFT_SPLITKV_ATTN on the Triton draft backend: each DFlash MLA draft
    # layer's split-KV extend attention (partial + combine) and its w_vc projection run as ONE clustered launch
    # (speculative/dflash_draft_attnblock.py) for 1-4 requests of 8 rows; other calls keep the served kernels. Not bitwise:
    # F16 attention operands (served: FP8 q / p on the rail), the projection before the combine, one BF16 rounding.
    SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK = EnvBool(False)
    # MLA DFLASH draft at TP > 1, <= 64 rows: each sublayer's all-reduce + grouped conv finish + add-RMSNorm run as one
    # kernel (speculative/dflash_draft_ar_site.py; multicast Lamport mailbox on the draft TP group); the reduction
    # rounds each rank's convolved partial before the sum, so outputs differ from the stock chain by rounding.
    SGLANG_OPT_DFLASH_DRAFT_AR_SITE = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_SITE: a site whose next kernel is the split-K conv projection releases it (PDL)
    # right after its own griddepcontrol.wait, so the projection stages its weights during the exchange; the other
    # sites keep the release after the norm stores (DraftArSites._early_for). Outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_AR_SITE_EARLY_TRIGGER = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_SITE: each draft layer's o_proj / down_proj (<= 8 rows, split-K) multicasts its
    # convolved partial into the site's slots from its epilogue and the site only polls (speculative/dflash_draft_ar_ring.py).
    # The site's outputs are bitwise those without it.
    SGLANG_OPT_DFLASH_DRAFT_AR_RING = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_RING: the ring o_proj / down_proj epilogue threads wait for the PDL predecessor and load
    # the conv coefficients while the main loop runs, instead of after the tile is staged. Same arithmetic: outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_RING_EARLY_COEF = EnvBool(False)
    # DFLASH MLA draft on the Triton backend (<= 64 rows): q_a / kv_a RMSNorm, q_b, W_UK, RoPE, the FP8 KV-row store and the extend
    # K / V in one CuTe kernel (speculative/dflash_draft_mlaq.py DraftMlaQuery) instead of five launches. Numerics: the served
    # rounding points, another accumulation order (ulp-level differences).
    SGLANG_OPT_DFLASH_DRAFT_MLAQ = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_MLAQ: the largest draft row count it serves (multiple of 8, 8..128; default 64 = draft B8);
    # larger row counts keep the served chain. Draft batches of 9-16 requests are 72-128 rows.
    SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS = EnvInt(64)
    # With SGLANG_OPT_DFLASH_DRAFT_MLAQ: draft row counts from this value to 128 (a multiple of 8 in [16, 128]; 0 = off) run
    # DraftMlaQueryWide (speculative/dflash_draft_mlaq_wide.py: RMSNorms + RoPE + FP8 KV row, cuBLAS Q_B, W_UK + RoPE; the two Triton
    # launches programmatically dependent) instead of DraftMlaQuery / the served chain. Numerics: the served rounding points, another
    # accumulation order (ulp-level). Draft batches of 13-16 requests = 104-128 rows.
    SGLANG_OPT_DFLASH_DRAFT_MLAQ_WIDE_MIN_ROWS = EnvInt(0)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_RING: a polled site of at most 8 rows (one draft block) first has lane s of warp 0
    # wait for one fragment of source s before the CTA polls (the GATED_AR form), so fewer polls are in
    # flight while the peers' stores land. Outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_AR_GATE = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_RING: the ring o_proj / down_proj release their PDL dependents only after their own
    # griddepcontrol.wait (fenced), and the polled site after each (<= 8 rows) runs everything before its own wait and waits
    # at the end (the SITE_DEFERRED form); it writes its normed rows into a static buffer per site. Outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_AR_DEFERRED = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_RING (Rubin): the polled site of <= 8 rows lands each CTA's columns of the TP slots in
    # shared memory with validity-reporting bulk copies, re-issued until no copied 16 B chunk is armed, instead of every
    # thread polling them (the VALID_AR form). Replaces SGLANG_OPT_DFLASH_DRAFT_AR_GATE there. Outputs unchanged.
    SGLANG_OPT_DFLASH_DRAFT_AR_VALID = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_AR_SITE: unpolled sites (> 8 rows, or no ring) of at least this many rows exchange over
    # the plugin's two-shot Lamport box (reduce-scatter to column owners, owner multicast of the BF16 sums; the
    # ArNormTwoShot form) instead of every rank receiving every rank's row. 0 = never. Outputs unchanged (same sums, same order).
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_ROWS = EnvInt(0)
    # Two-shot draft AR sites: owners poll their TP slots in the plugin reduce's batched form (every slot re-loaded each pass, all
    # loads in flight; the TWOSHOT_BATCHED form) instead of re-loading only the pending ones. Same words, same order, same bits.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BATCHED = EnvBool(False)
    # Two-shot draft AR sites: the gather-side re-arm fence moves to one non-live thread after the norm's first CTA barrier
    # (overlaps the sum of squares / norm). Same bits.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_FENCE_OFFLOAD = EnvBool(False)
    # Two-shot draft AR sites inside one draft forward: site s leaves its gather row published and site s + 1's non-live
    # threads re-arm (and fence) it while the live threads push; the forward's last site re-arms its own. Same bits.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_DRR = EnvBool(False)
    # Draft AR sites (one-shot and two-shot): the norm's sum of squares is stored per thread in shared memory and the cluster
    # pushers rebuild warp_sum's butterfly tree from it (same adds, same order) instead of 5 warp shuffles. Same bits.
    SGLANG_OPT_DFLASH_DRAFT_AR_SMEMSUM = EnvBool(False)
    # Rows of the two-shot box, a multiple of 8 in [64, 128]: above 64 (the one-shot mailbox) the draft forward keeps its
    # AR sites up to this many rows (wide decode tier, 9-16 requests) on the two-shot path; needs TWOSHOT_ROWS <= 64.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_CAP = EnvInt(64)
    # With a two-shot box cap above 64: two-shot sites of more than this many rows run ceil(rows / N) rows per CTA, so their grid
    # stays at <= N 8-CTA clusters (clusters beyond what stays co-resident cost a second wave). Per row the same arithmetic
    # and exchange, so the same bits. 0 = one row per cluster.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_MAXCL = EnvInt(0)
    # DFLASH selector draft graph: its captured tail writes the verify tokens into a static buffer and the target verify
    # runner's static input_ids, the prefix seq-lens snapshot moves before the draft launch, and the verify runner skips
    # its input_ids / positions copies (speculative/dflash_post_stage.py). 0 = off, 1 = on, 2 = on + a per-round check
    # against the eager path (host syncs; identity arms only). Outputs unchanged.
    SGLANG_OPT_DFLASH_POST_IN_GRAPH = EnvInt(0)
    # Sparse TARGET_VERIFY graph (kimi_k31_verify_dsa_binding): the prologue's valid mask, per-request page-table de-expand
    # and score contexts in one Triton kernel (speculative/dflash_verify_prologue.py), and the stable-rows mask passed as a
    # callable (no kernel unless SGLANG_ENABLE_PP_SPEC uses it). Outputs unchanged (plain copies).
    SGLANG_OPT_DFLASH_VERIFY_PROLOGUE_FUSED = EnvBool(False)
    # DFLASH draft head (models/dflash.py _project_candidate_logits, dense BF16 head): at these drafted-row counts
    # (comma list, e.g. "21,28" = 3 / 4 requests at 7 drafted) project through the plugin split-K projection with one K
    # slice (M tile 64, N tile = rows rounded up to 8, 9 stages) instead of torch.matmul. "" = off. Numerics: one FP32
    # accumulation per output instead of cuBLAS's split-K order (draft logits only; target tokens unchanged).
    SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS = EnvTuple(())
    # One-row-per-cluster two-shot draft AR sites of more than this many rows compile with a launch bound of 6 CTAs per SM (<= 85
    # registers), so more of their CTAs stay resident instead of falling back to a second wave. 0 = never. Same bits.
    SGLANG_OPT_DFLASH_DRAFT_AR_TWOSHOT_BOUND_ROWS = EnvInt(0)
    # With SGLANG_OPT_DFLASH_SELECTOR_FUSED_ACCEPT and an all-reduce DFLASH_SELECTOR sync: the select kernel writes the
    # sync's fp32 all-reduce input and the commit kernel reads its output (three glue kernels fewer per round;
    # kernels/ops/speculative/selector_sampled_accept.py selector_sampled_accept_synced); same outputs.
    SGLANG_OPT_DFLASH_SELECTOR_SYNC_FOLD = EnvBool(False)
    # With SGLANG_OPT_DFLASH_SELECTOR_FUSED_ACCEPT, on rounds that take SGLANG_OPT_DFLASH_VERIFY_SOFTMAX_WIDE: the accept
    # evaluates p from the verify logits and per-row softmax stats where it reads it, instead of normalizing every
    # verify row over the vocabulary first (selector_sampled_accept.py selector_logit_accept); same outputs.
    SGLANG_OPT_DFLASH_SELECTOR_LOGIT_ACCEPT = EnvBool(False)
    # With SGLANG_OPT_DFLASH_SELECTOR_LOGIT_ACCEPT: the per-row softmax stats come from one read of the verify logits at
    # any row count (kernels/ops/speculative/verify_row_stats.py) instead of verify_softmax_wide's stats (<= 128 rows,
    # bitwise to the served p). The max and inv_temp keep the served bits; the denominator is summed in another order,
    # so p is not bit-identical to the served p: relative differences reach ~2e-6 at vocab 163840, the same order as
    # the served fp32 sum's own error against an FP64 sum.
    SGLANG_OPT_DFLASH_VERIFY_ROW_STATS = EnvBool(False)
    SGLANG_ENABLE_LILICORR_SAMPLING = EnvBool(False)
    SGLANG_LILICORR_REQUIRE_SAMPLING = EnvBool(False)
    SGLANG_RAGGED_VERIFY_MODE = EnvStr("static")
    SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE = EnvBool(False)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Which speculative decisions rank 0 broadcasts to its TP group. Comma
    # separated presets ("all", "rng", "init", "off"), or SpecTpSyncSite slugs
    # and numbers, each negatable with a leading "-": "all,-dspark-plan,-dspark-draft".
    # The default syncs only the startup memory probes: ranks compute every per-step
    # decision alike. If TP ranks still diverge or hang, set "all" and report a repro.
    # The K3.1 GB switch keeps its tuned site set.
    SGLANG_SPEC_TP_SYNC = EnvStr(_k31_gb_best("all,-dflash-accept-greedy", "init"))
    # The DFlash sync sites (token ids and accept lengths, all below 2**24) reach the TP group through one fp32
    # all-reduce of rank 0's values plus zeros (exact) instead of a ring broadcast, whose root leaves first.
    SGLANG_OPT_SPEC_TP_SYNC_ALLREDUCE = EnvBool(False)
    # Kill-switch for the draft-extend cuda graph. Draft extend then always runs
    # eager. Escape hatch for setups where the capture's memory pool costs more
    # than the graph saves (e.g. DeepEP MoE workspace captured at full dispatch
    # capacity).
    SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH = EnvBool(False)

    # K3.1 megakernel flags, one section per component; add flags under the
    # matching section, all default off.
    # --- K3.1: expert (MoE) ---

    # --- K3.1: attention ---

    # --- K3.1: KDA ---
    # KDA spec verify with --mamba-ssm-dtype bfloat16 and the ReplaySSM rings: run the fused CuTe verify
    # (kernels/ops/attention/kda_decode_mtp.py, state staged as BF16 and widened to FP32) instead of falling back to
    # the generic conv + Triton recurrence chain. Same math as FP32 on the same (BF16) state values.
    SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE = EnvBool(False)

    # --- K3.1: communication ---

    # --- K3.1: DFLASH draft ---
    # DFLASH selector draft, sampled verify: accept on the sparse selector q split
    # over the vocabulary, one rank-0 sync; the same draws as the dense-q path.
    SGLANG_OPT_DFLASH_SELECTOR_FUSED_ACCEPT = EnvBool(False)
    # With SGLANG_OPT_DFLASH_SELECTOR_FUSED_ACCEPT: draw the accept's uniforms on a side stream before the verify
    # (speculative/dflash_accept_rand.py); the same bits, off the critical path.
    SGLANG_OPT_DFLASH_ACCEPT_RAND_HOIST = EnvBool(False)
    # DFLASH MLA draft on the Triton backend: q / k concat and the FP8 KV-row store in one
    # kernel (kernels/ops/kvcache/mla_fused_kv_write.py), bitwise to stock.
    SGLANG_OPT_DFLASH_MLA_FUSED_KV_WRITE = EnvBool(False)
    # DFLASH MLA draft on the Triton backend: the draft block's extend attention as one split-KV pass + a fixed-order
    # combine (kernels/ops/attention/dflash_draft_splitkv.py). 0 = stock serial kernel; 1 = stock operand dtypes
    # (q / p cast to fp8 for the rail dots); 2 = bf16 rail operands (fp8 K / V upcast). Needs a bounded draft window.
    SGLANG_OPT_DFLASH_DRAFT_SPLITKV_ATTN = EnvInt(0)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITKV_ATTN = 1 at draft batch > 1: one CTA scores each rail tile for a group of query heads
    # (one KV head, MQA), so each K / V tile is read once per group instead of once per head. Bitwise equal outputs.
    SGLANG_OPT_DFLASH_DRAFT_P1_HPACK = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_SPLITKV_ATTN and SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE: the split-KV combine kernel also releases its PDL
    # dependents right after its own wait, so a PDL-launched successor (the SPLITK_WVC projection) issues its weight tiles during the
    # combine. The successor reads the combine's output only after its griddepcontrol.wait (whole combine grid). Bitwise equal outputs.
    SGLANG_OPT_DFLASH_DRAFT_P1_COMBINE_TRIGGER = EnvBool(False)
    # With SGLANG_OPT_DFLASH_DRAFT_P1_HPACK = 1: rail tiles per partial-pass slot (tps). "1" (default) = unchanged; an integer =
    # that tps at every draft batch; "auto" = the per-batch table tuned on the target GPU (dflash_draft_splitkv.P1_TPS_AUTO).
    # tps > 1 changes the fp32 reduction order of the draft attention (drafter numerics; not bitwise to tps = 1).
    SGLANG_OPT_DFLASH_DRAFT_P1_TPS = EnvStr("1")
    # With SGLANG_OPT_DFLASH_DRAFT_P1_HPACK = 1: the head-packed pass's launch at wider draft batches. "0" (default) = the served
    # launch (16 rows = 2 heads per CTA, 8 warps, 1 stage; tps from SGLANG_OPT_DFLASH_DRAFT_P1_TPS); "auto" = the per-batch table dflash_draft_splitkv.P1_WIDE_AUTO
    # (heads per CTA, V split, warps, stages, tps, combine chunk; tuned on the target GPU). Wider CTAs change the dot layouts and the fp32
    # reduction order (drafter numerics, not bitwise to "0"); batches outside the table keep the served launch. Any other value, or
    # "auto" without P1_HPACK, fails at startup.
    SGLANG_OPT_DFLASH_DRAFT_P1_WIDE = EnvStr("0")
    # DFlash2 selector head after the LM head in three launches (top-k + lattice operands, lattice GEMM, walk);
    # TP16 K3.1 draft shapes only, bitwise equal to the unfused head.
    SGLANG_OPT_DFLASH_FUSED_SELECTOR = EnvBool(False)
    # With SGLANG_OPT_DFLASH_FUSED_SELECTOR: the candidate top-k, the lattice GEMM and the walk after the gather in one
    # kernel (jit/csrc/speculative/dflash_selector_one_launch.cuh) instead of three launches. Bitwise to off.
    SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH = EnvBool(False)
    # With SGLANG_OPT_DFLASH_FUSED_SELECTOR, draft batches of 2+ requests: the draft head reads the drafted rows copied compact by a
    # PDL kernel (kernels/ops/speculative/dflash_draft_gather.py) instead of the plain reshape copy of the non-contiguous slice.
    # Pure copy: bitwise to off.
    SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER = EnvBool(False)
    # DFlash2 selector draft sampler: stage the per-request temperatures and greedy mask only when the batch's
    # sampling tensors, their contents (version counters) or bs change, instead of three launches every round.
    # Same buffer contents: bitwise to off.
    SGLANG_OPT_DFLASH_DRAFT_STAGE_ONCE = EnvBool(False)
    # DFLASH draft layer: the Triton glue kernels (grouped conv, the MLA fused KV write, the split-KV attention pair)
    # launch with PDL and wait before their first read, so the split-K projections that follow them stage their
    # weights early. Same kernels and math: bitwise to off.
    SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE = EnvBool(False)
    # DFLASH draft block prep as one launch (speculative/dflash_fused_draft_prep.py), CHECK only: for the first N
    # draft graph rounds the fused kernel writes shadow buffers that are compared bitwise with the eager prep's
    # outputs (a host sync per checked round). Served outputs unchanged. 0 = off.
    SGLANG_OPT_DFLASH_FUSED_DRAFT_PREP_CHECK = EnvInt(0)
    # DFLASH draft block prep as one launch, SERVED (ON form): the kernel writes the worker / runner / Triton backend
    # buffers the eager prep writes, the eager ops are skipped, and the draft graph runner skips fill_from and the
    # backend's out-graph prep for that replay. Bitwise to off (integer index work + an exact embedding row copy).
    # Needs the replicated draft embedding table (attention DP); exclusive with the CHECK knob above.
    SGLANG_OPT_DFLASH_FUSED_DRAFT_PREP = EnvBool(False)
    # ON-form re-check: for the first N fused replays, snapshot the runner's static buffers and the replay backend's tensors, run the
    # skipped fill_from + backend prep anyway, and compare bitwise (logs MISMATCH / OK). Test only (host syncs).
    SGLANG_TEST_DFLASH_FUSED_DRAFT_PREP_RECHECK = EnvInt(0)

    # --- K3.1: verify ---
    # DFLASH sampled verify: the target softmax as flashinfer's vocabulary-split kernels with the normalization
    # spread over (row, 4096-column) CTAs and the per-request temperature read in-kernel
    # (kernels/ops/speculative/verify_softmax_wide.py); bitwise equal to the flashinfer call it replaces.
    SGLANG_OPT_DFLASH_VERIFY_SOFTMAX_WIDE = EnvBool(False)

    # --- K3.1: host ---
    # Decode-stall diagnostics in every scheduler process (0 = off): log a GC pass of at least this many ms,
    # and the main thread's stack when it stays at one position this long (scheduler_components/stall_probe.py).
    SGLANG_LOG_GC_STALL_MS = EnvInt(0)
    SGLANG_LOG_STALL_STACK_MS = EnvInt(0)
    # DFlash decode: the target verify's input-slot copies, attention metadata and mamba track ids run on a
    # side stream that waits only for the draft replay's start (model_executor/runner/dflash_verify_prep.py).
    SGLANG_OPT_DFLASH_VERIFY_PREP_BESIDE_DRAFT = EnvBool(False)
    # DFlash decode seams: the round's small device-to-device copies (forced-AL token block, aux-graph input staging,
    # draft-token block, verify replay inputs) run as one batched SM copy kernel each instead of copy-engine memcpys.
    SGLANG_OPT_K31_SEAM_SMALL_COPY = EnvBool(False)
    # DFlash sampled verify (selector fused accept; no top-k / top-p / penalties / grammar / logprobs): no full-logit
    # all-gather. Each rank reduces its vocab shard to a per-row payload (argmax, softmax stats, Gumbel winner,
    # candidate logits), the TP group all-gathers the payloads and every rank combines them: the chain walk on the
    # served uniforms, the bonus by Gumbel-max (distribution-identical, not token-identical, to the served draw) and
    # the target argmax the forced-acceptance simulator reads (bitwise). kernels/ops/speculative/vocab_parallel_accept.py
    SGLANG_OPT_K31_VP_ACCEPT = EnvBool(False)
    # Forced-acceptance simulator only (the sampled accept keeps the full logits): the target argmax comes from the
    # vocab shard (softcap_row_argmax + an all-gather of [rows, 2]) instead of torch.argmax over the full logits; bitwise.
    SGLANG_OPT_K31_VP_SIM_ARGMAX = EnvBool(False)
    # SGLANG_OPT_K31_VP_ACCEPT cross-rank guard: every N vocab-parallel accepts, read the device counter of requests
    # whose seed / input hash differed between TP ranks and log it (one host sync per N rounds). 0 = never read.
    SGLANG_OPT_K31_VP_XRANK_LOG_EVERY = EnvInt(0)
    # SGLANG_OPT_K31_VP_ACCEPT payload all-gather: "multimem" = the served multimem all-gather kernel
    # (triton_symm_mem_ag, built once at worker init; NCCL if the group has no multicast), "nccl" = NCCL ring.
    SGLANG_OPT_K31_VP_GATHER = EnvStr("multimem")
    # SGLANG_OPT_K31_VP_ACCEPT column chunks per vocab shard in the local payload kernel: 0 = auto (~128 programs),
    # 1 = the whole shard per row (the first form), n = fixed. Changes only the FP32 summation order of the softmax
    # denominator; the target argmax stays bitwise.
    SGLANG_OPT_K31_VP_SPLIT = EnvInt(0)
    # Drop the multimem payload gather's entry barrier (the served logits gatherer's form). Default keeps it.
    SGLANG_OPT_K31_VP_GATHER_SKIP_ENTRY = EnvBool(False)

    # ===================================================================
    # Multimodal processing
    # ===================================================================
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # Supported by Kimi-K2.5 and native diffusion RGB input preprocessing.
    SGLANG_FORCE_CPU_IMAGE_PREPROCESSING = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)
    # Total seconds to download one media URL (redirects and body included).
    # Falls back to the legacy REQUEST_TIMEOUT when only that is set.
    SGLANG_MEDIA_URL_TIMEOUT = EnvFloat(
        lambda: float(os.getenv("REQUEST_TIMEOUT", "5"))
    )
    # Whether client media may name local files (paths or file:// URLs).
    SGLANG_MEDIA_ALLOW_LOCAL_FILES = EnvBool(True)

    # ===================================================================
    # Unlimited-OCR v2
    # ===================================================================
    # Send a split document's next window while the current one decodes.
    SGLANG_ENABLE_UOCR_DOC_PREFETCH = EnvBool(True)
    # GPU memory for early-encoded follow-up pages; past it they encode at prefill.
    SGLANG_UOCR_DOC_PREFETCH_MAX_MB = EnvInt(1024)
    # Stage Unlimited-OCR crops in pinned host memory ahead of the forward.
    SGLANG_ENABLE_UOCR_CROP_PREFETCH = EnvBool(True)
    # Pinned host memory for prefetched crops; past it crops stage inline.
    SGLANG_UOCR_CROP_PREFETCH_MAX_MB = EnvInt(1024)
    # Materialize multimodal SHM segments off the scheduler thread.
    SGLANG_ENABLE_UOCR_ASYNC_SHM_MATERIALIZATION = EnvBool(True)
    # SHM bytes materializing at once; past it a request materializes inline.
    SGLANG_UOCR_ASYNC_SHM_MAX_MB = EnvInt(4096)
    # Send processor-owned model_specific_data tensors (crops) through SHM.
    SGLANG_ENABLE_UOCR_SHM_WRAP_MODEL_DATA = EnvBool(True)

    # ===================================================================
    # Multimodal CUDA IPC transport
    # ===================================================================
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    # Reuse the mapping for the already-allocated bounded CUDA IPC pool. This
    # has no effect unless CUDA IPC feature transport is explicitly selected.
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(True)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # ===================================================================
    # Mamba state and cache
    # ===================================================================
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)
    # Kill-switch for the fused per-slot conv clear/copy kernel (MambaPool);
    # falls back to the per-conv-type Python loop.
    SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS = EnvBool(False)
    # Opt-in: on the unified radix tree, leave the matched-prefix mamba evictable
    # during decode (it is already COW'd to the request's own slot) and shrink the
    # mamba pool ratio accordingly. Frees one resident slot per running request,
    # raising max_running_requests. Off = original locking + ratio (escape hatch).
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK = EnvBool(False)
    # Opt-in: KDA commit-then-verify under --enable-linear-replayssm-spec. The
    # post-verify step only records each slot's accept length; the next
    # nv_cutedsl verify replays that window in-kernel and writes the committed
    # state back, so the state is read once per step instead of twice.
    # Track-crossing rows and every non-verify forward commit eagerly. BF16
    # state also needs SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE.
    SGLANG_OPT_KDA_SPEC_FOLD_VERIFY = EnvBool(False)
    # Fold mode: verify batches smaller than this flush first and verify without
    # the replay (the fold kernel costs more latency per CTA than it saves when
    # the whole batch is one wave). 0 = fold every chain verify.
    SGLANG_OPT_KDA_SPEC_FOLD_MIN_BS = EnvInt(0)
    # V-column tile of the ReplaySSM exact-fold commit kernel (and of the fold
    # mode's sparse commit on BF16 state, which follows it). On FP32 state any
    # value gives the same bits (it only partitions independent V columns); on
    # BF16 state a few elements move by 1-2 BF16 ulps with the tile. None keeps 8.
    SGLANG_OPT_KDA_REPLAYSSM_FOLD_BLOCK_V = EnvInt(None)
    # KDA ReplaySSM spec-verify records (rawv/rawk/d/k) in this dtype
    # (float32|bfloat16|float16). Unset: the activation dtype, as before.
    # g/beta stay fp32; GDN is unaffected.
    SGLANG_KDA_REPLAY_RECORD_DTYPE = EnvStr(None)
    # PD decode with --disaggregation-decode-enable-radix-cache on a hybrid-SSM
    # model: the radix tree keeps FULL KV only. Prefill always sends a
    # request's final recurrent state and decode never computes a prompt, so a
    # cached prefix only saves KV transfer and no tree-held state is ever read;
    # each request keeps its own one state slot. Set 0 to keep tree states.
    SGLANG_DECODE_STATE_FREE_RADIX = EnvBool(True)
    # Opt-in: store radix mamba checkpoints only in the hicache host pool.
    # Matched prefixes and donated checkpoints then occupy no GPU mamba slot,
    # so the pool ratio is the live slot plus the ping-pong slots extra-buffer
    # still writes. Requires --enable-hierarchical-cache. Off keeps the
    # donated GPU slot and the previous ratio.
    SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT = EnvBool(False)

    # ===================================================================
    # CUDA graphs and execution buffers
    # ===================================================================
    # Guards CUDA graph executable dedup via cudaGraphExecUpdate.
    SGLANG_ENABLE_CUDA_GRAPH_DEDUP = EnvBool(False)
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)
    # Graph-input fill fast paths: skip the padding reset of an unpadded replay and copy
    # one dense integer dtype pair with a single foreach launch.
    SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Reuse wholly-free graph-pool segments for step-local eager allocations.
    SGLANG_ENABLE_GRAPH_POOL_BORROW = EnvBool(False)
    # Mint capture's measured footprint as one span so the graph pool is carved
    # out of a single contiguous region instead of grown segment by segment.
    SGLANG_ENABLE_GRAPH_POOL_PRECARVE = EnvBool(False)
    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)
    # Breakable prefill CUDA graphs capture the Triton KDA extend instead of
    # breaking the graph at every KDA layer (layers/attention/linear/
    # kda_prefill_graph.py); this kill-switch restores the eager break.
    SGLANG_DISABLE_KDA_PREFILL_GRAPH_EXTEND = EnvBool(False)
    # Sequences a captured KDA extend bucket accepts; a larger prefill batch on
    # any dp rank runs the step eagerly. Bounds the padded grids and scratch.
    SGLANG_KDA_PREFILL_GRAPH_MAX_SEQS = EnvInt(128)
    # Experimental: breakable prefill CUDA graphs capture the Triton GDN extend
    # instead of breaking at every GDN layer (layers/attention/linear/
    # gdn_prefill_graph.py).
    SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Sequences a captured GDN extend bucket accepts (at most the request pool); a
    # larger prefill batch runs the step eagerly. Bounds the padded grids and scratch.
    SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS = EnvInt(128)

    # ===================================================================
    # Tokenizer, request state, embeddings, and reasoning controls
    # ===================================================================
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)
    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)
    SGLANG_PATCH_TOKENIZER = EnvBool(True)
    # Group replies by tokenizer worker; disable for per-request delivery.
    SGLANG_BATCH_TOKENIZER_OUTPUTS = EnvBool(True)
    # Encode long rendered chat prompts as chunks on the tokenizers thread pool.
    SGLANG_PARALLEL_PROMPT_ENCODE = EnvBool(True)
    # Shorter prompts use the single-call encode; below this the gain is eaten
    # by rayon dispatch and the id merge.
    SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS = EnvInt(32768)
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)
    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ===================================================================
    # Encoder pipeline and disaggregation
    # ===================================================================
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    SGLANG_ENCODER_MM_LOAD_WORKERS = EnvInt(4)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Seconds before permanently dropping an unhealthy encoder (0 = keep probing).
    SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL = EnvFloat(600.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # ===================================================================
    # Native gRPC server
    # ===================================================================
    # Native gRPC server. SGLANG_GRPC_PORT is the env fallback for the
    # --grpc-port CLI flag; setting either enables the native server alongside
    # HTTP. The worker-threads knob stays env-only (internal tuning, no CLI
    # surface).
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_GRPC_WORKER_THREADS = EnvInt(4)

    # ===================================================================
    # NUMA and CPU affinity
    # ===================================================================
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(True)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # ===================================================================
    # DeepSeek V4
    # ===================================================================

    # Model and Quantization
    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Set True to dequantize the FP4 experts to FP8 at runtime
    SGLANG_DSV4_FP4_DEQUANT = EnvBool(False)
    # DeepSeek-V4 default when a request carries no reasoning_effort: a
    # deepseek-recipe tier name; unset -> deepseek-recipe's default.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")
    # DeepSeek-V4.1 default when a request carries no reasoning_effort: one of
    # a deepseek-recipe tier name or an integer budget in [1, 100]; unset ->
    # deepseek-recipe's default.
    SGLANG_DSV41_REASONING_EFFORT = EnvStr(None)
    # Quantize the SWA fp8 KV cache from bf16-rounded values (matches
    # trainer-side QAT and the DSA-CP path) instead of fp32 registers.
    SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE = EnvBool(False)
    # Decoder SWA bounded replay: eager prefill steps with up to this many tail rows
    # run the tail layers from CUDA graphs (buckets of 128 rows); 0 keeps them eager.
    SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS = EnvInt(2048)
    # Prefill-graph shapes of at least this many tokens run the tail from the tail
    # graphs in a break (measured crossover 1536-2048 tokens on GB300 TP4).
    SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS = EnvInt(2048)
    # Paged KV layout of the DeepSeek-V4 family pools: "v4" (584 B/token, every
    # GPU), "v41" (the SM100 FlashMLA V4.1 formats: 528 B fp8 SWA cache, fp8 or
    # fp4 compressed caches), "v41_new" (the same bytes interleaved per token, for
    # FlashMLA's 2026.09.30 kernels) or "auto" (v41 on SM100, v4 elsewhere).
    SGLANG_DSV4_KV_LAYOUT = EnvStr("v4")
    # Compressed-cache layout under "v41": "auto" (fp4 for the fp4-rounded
    # ratio-1 / ratio-2 latents, fp8 for ratios 4 / 128), "fp8" or "fp4" for all.
    SGLANG_DSV4_COMPRESSED_KV_LAYOUT = EnvStr("auto")
    # unified_kv only: split the pool into an fp8 nope pool plus a parallel
    # bf16 rope pool, 640 B/token instead of 1024. The unified pool takes no
    # dtype, so --kv-cache-dtype has no effect there and this switch is the
    # only way to ask; on separate-KV it is the reverse -- --kv-cache-dtype
    # picks the buffer dtype and this switch is inert.
    SGLANG_DSV4_UNIFIED_KV_FP8 = EnvBool(False)
    # ROCm gfx950, DeepSeek-V4 128x128 block-fp8 dense linears: run the GEMM as
    # aiter's e8m0 mxscale bpreshuffle (flydsl) instead of the fp32-scale
    # CK/asm/triton bpreshuffle. The weight bytes are shared; only an e8m0 copy
    # of the (power-of-two) weight scale is added at load, and activations are
    # quantized with power-of-two (ue8m0) scales. Restricted to the (N, K)
    # shapes in SGLANG_DSV4_DENSE_MXSCALE_SHAPES ("N:K,N:K"; empty = built-in
    # whitelist of shapes that win in the microbench, "all" = every shape).
    SGLANG_DSV4_DENSE_MXSCALE = EnvBool(False)
    SGLANG_DSV4_DENSE_MXSCALE_SHAPES = EnvStr("")
    # fp8 unified_kv decode/verify on aiter's FlyDSL v4 nm decode when the
    # installed aiter ships it; 0 keeps the asm kernel.
    SGLANG_DSV4_FLYDSL_DECODE = EnvBool(True)
    # ... which copy: "auto" = aiter's when the installed aiter ships it, else
    # sglang's in-tree copy; "aiter" = aiter's only (asm without it); "intree" =
    # the in-tree copy only.
    # TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin
    SGLANG_DSV4_FLYDSL_DECODE_SOURCE = EnvStr("auto")
    # ... also with 128 local heads (DP attention). Off: there the asm path
    # below (split plan + HCA persistent, ROCm/aiter#6126) measured the same
    # DP8 DSpark decode step on MI355X without depending on FlyDSL.
    SGLANG_DSV4_FLYDSL_DECODE_H128 = EnvBool(False)
    # fp8 unified_kv asm decode (when FlyDSL is off or absent): pick the KV
    # split count from aiter's v4 nm latency model (ROCm/aiter#6126; ported
    # when aiter predates it) with a constant split_indptr, per stream kind.
    SGLANG_DSV4_FP8_DECODE_SPLIT_MODEL = EnvBool(True)
    # context the HCA split plan is sized for, capped at the server context
    # length (0: the server context length). Sizing for 1M over-splits HCA at
    # AgentX lengths (32k-200k).
    SGLANG_DSV4_FP8_DECODE_SPLIT_CTX = EnvInt(200_000)
    # fp8 unified_kv asm decode: HCA layers with 128 local heads (DP attention)
    # take aiter's persistent mla_decode_fwd_v4_nm_ps (in-kernel split + merge)
    # for MIN_ROWS..32768 rows; needs aiter >= ROCm/aiter#6126.
    SGLANG_DSV4_FP8_HCA_PERSIST = EnvBool(True)
    SGLANG_DSV4_FP8_HCA_PERSIST_MIN_ROWS = EnvInt(15)

    # Deprecated for --engram-config "placement". The deprecated engram variables
    # are read only by _resolve_engram_config.
    SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE = EnvBool(False)
    SGLANG_OPT_ENGRAM_EXTEND_HASH = EnvBool(False)
    SGLANG_OPT_DSV41_MEGA_MHC_PREFILL = EnvBool(False)
    # Deprecated for --engram-config "replication".
    SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT = EnvStr("per_rank")
    # Scheduler-side staging of shared host-table rows on GPUs without ATS
    # (engram_staging.py); pays off for large prefill batches.
    SGLANG_OPT_DSV41_ENGRAM_HOST_STAGING = EnvBool(False)
    # Deprecated for --engram-config "lookup".
    SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS = EnvBool(False)

    # Kernels and indexer
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_FLASHINFER_MHC = EnvBool(False)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    # Shard the ratio-1/2 index sources' dense prefill scores across the
    # attention-TP group and all-gather the selections, instead of every rank
    # scoring every row. Identical to the replicated path under fp32 logits;
    # bf16 logits vary with the tile's row count, so picks near the selection
    # floor can differ there.
    SGLANG_OPT_DSV4_INDEXER_TP_SHARD = EnvBool(False)
    SGLANG_OPT_DSV4_NONPAGED_INDEXER = EnvBool(True)
    # Per-rank local query rows (after DP-attention sharding when enabled),
    # not request ISL.
    SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS = EnvInt(8192)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    # DeepSeek-V4.1 ratio-1/2 index layers: the DeepGEMM fp4 logits kernels store bf16 instead
    # of fp32 (the reference ranks bf16 scores); every consumer of those logits must take bf16.
    # Unset: bf16 on SM100 (with bf16 head weights), fp32 elsewhere.
    SGLANG_OPT_DSV41_BF16_INDEX_LOGITS = EnvBool(None)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(True)
    # gfx950 DeepSeek-V4.1 prefill: attend with aiter's OPUS sparse kernel over a bf16
    # dequant of the chunk's SWA and compressed history, instead of the decode kernel.
    SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL = EnvBool(False)
    # DSpark draft block on the HIP radix backend: build the attention metadata inside the
    # draft CUDA graph from the raw inputs instead of eagerly before every replay.
    SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA = EnvBool(_default_hip)
    # gfx950 aiter MXFP8 dense route: keep the weight (16, 16)-preshuffled and run aiter's FlyDSL MXFP8
    # GEMM (needs aiter with ROCm/aiter#5896) instead of the Triton group32 GEMM on the plain weight.
    SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE = EnvBool(False)
    # V4.1 low-ratio decode / target-verify indexer: score with aiter's schedule-free row-group
    # FP4 MQA logits kernel, a request's verify rows sharing each key load.
    SGLANG_HIP_FP4_INDEXER_ROWGROUP = EnvBool(_default_hip)
    # V4.1 ratio-1 index-K pool in aiter's row-group page-8 layout (8 consecutive 8-slot pages per
    # 64-slot page), so a candidate consumer layer scores only its published blocks.
    SGLANG_HIP_DSV41_INDEX_K_PAGE8 = EnvBool(False)

    # cache, GEMM, and distributed
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    # CUDA: fuse inverse RoPE into the UE8M0 FP8 wo_a activation quantization.
    # Off by default; requires the FP8 wo_a path.
    SGLANG_OPT_FP8_WO_A_FUSED_INVROPE = EnvBool(False)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    # DSpark decode: build the verify window and the draft anchor column in one
    # launch instead of two adds, assign_extend_cache_locs and a copy.
    SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark folded draft: the draft graph's tail writes [anchor | draft tokens]
    # into the sampler's own buffer, so the step skips its eager concat.
    SGLANG_OPT_DSV41_DSPARK_MID_FOLD = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark static verify: run the target verify runner's metadata prep through the
    # metadata glue graph where its backend declares the prep host-free.
    SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark static verify: build the target verify metadata on a side stream beside the
    # draft graph, for requests up to the candidate span (needs the glue switch above).
    SGLANG_OPT_DSV41_TARGET_METADATA_EARLY = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # Decode / verify graph replays whose metadata glue graph is captured skip the
    # replay ForwardBatch view (the graph reads slots and buffers by address).
    SGLANG_OPT_SKIP_GLUED_REPLAY_VIEW = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark static verify: record the post-verify engram history commit in the
    # target-verify CUDA graph (epilogue) instead of launching it after the replay.
    SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    # DSpark static verify: an all-sampled step (no top-k / top-p) replays its
    # sampling accept and commits as a CUDA graph right after the verify graph.
    SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH = EnvBool(
        SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get
    )
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # ===================================================================
    # Inkling
    # ===================================================================
    SGLANG_OPT_USE_FUSED_GATE_TOPK = EnvBool(True)
    # Inside the fused gate: use the CUDA JIT top-k+renorm kernel (v2) instead
    # of the triton kernel when the production Inkling shape applies.
    SGLANG_OPT_USE_GATE_TOPK_JIT = EnvBool(True)
    # Inside the fused gate: replace the cublas gate linear with the
    # expert-per-block GEMV JIT kernel at small token counts (GateGemvMode).
    SGLANG_OPT_GATE_GEMV_MODE = EnvInt(GateGemvMode.PAIR)
    # Capture all multi-layer EAGLE draft-extend steps and the in-graph chain
    # rotation into ONE CUDA graph instead of one captured graph per step.
    SGLANG_ENABLE_SINGLE_CG_DRAFT = EnvBool(True)
    # Draft sampler uses the Gumbel-max trick (argmax(probs / Exp(1))) instead of
    # torch.multinomial, whose device-side validity assert breaks draft-graph replay.
    SGLANG_OPT_USE_GUMBEL_SAMPLE = EnvBool(True)
    # Multi-layer chain-MTP boundary-KV fix: widen the draft-extend window to
    # rewrite rejected-draft KV rows before reuse (acc_len repair; on by default).
    SGLANG_ENABLE_MTP_BOUNDARY_KV_FIX = EnvBool(True)
    SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP = EnvBool(True)
    SGLANG_OPT_USE_INKLING_SHEARED_BIAS = EnvBool(True)
    # Use feature-stacked GEMMs for the no-LoRA BF16 shared sink. Eligible LoRA
    # serving enables this layout independently of the flag.
    SGLANG_OPT_LINEARIZED_SHARED_SINK = EnvBool(True)
    # Use the autotuned JIT all-reduce, falling back to torch multimem for
    # shapes where it wins.
    SGLANG_OPT_USE_INKLING_CUSTOM_AR = EnvBool(True)
    # Fuse small-batch decode all-reduce, MLP convolution, and attention norm.
    # Requires the custom all-reduce; other shapes use the unfused path.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM = EnvBool(True)
    # Fuse eligible extend all-reduce, convolution, and cache updates.
    # Supports scattered or full-width state and requires the custom all-reduce.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV = EnvBool(True)
    # Fuse eligible convolution, QK norm, window, and KV-store prologue work.
    # Non-BF16 caches retain the backend KV store.
    SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE = EnvBool(True)
    # Override shared-expert selection: true uses grouped GEMM, false uses BMM.
    # When unset, selection follows model, quantization, and LoRA requirements.
    SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE = EnvBool(True)
    # Fold the conditional long-context log-scaling tau into its producers
    # instead of separate output-sized scale kernels: the fused attn
    # prologue's q path (bit-exact, before MXFP8 quantization there) and the
    # rel_logits projection's r OPERAND (the diagonal scale commutes through
    # the einsum, shrinking the pass by rel_extent/d_rel = 64x; rounding moves
    # before the GEMM). Flag-off keeps the standalone apply_log_scaling_tau
    # on the outputs.
    # Fold the MoE shared-expert partials into the custom AR kernels instead
    # of a separate {routed + shared} torch.add per MoE layer; some buckets
    # keep a pre-add during the AR stage-in. torch.add numerics
    # (bit-identical). Requires SGLANG_OPT_USE_INKLING_CUSTOM_AR.
    SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED = EnvBool(True)
    SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU = EnvBool(True)
    # Dispatch the rel_logits projection around einsum's hidden compaction
    # copy of the strided r operand (a view into the packed qkvr output):
    # zero-copy strided-batched matmul at small t, JIT row-compact + einsum
    # above the band, single-launch tau-folded kernel in the small-t tau
    # band. Bit-identical to the plain einsum; flag-off restores it.
    SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH = EnvBool(True)
    # Quantize and store MXFP8 K/V data and scales in one fused kernel.
    SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE = EnvBool(True)
    # Default reasoning effort in [0.0, 0.99] when omitted by a request.
    # An empty string falls back to the protocol default (0.9); the effort
    # directive is always emitted.
    SGLANG_INKLING_DEFAULT_REASONING_EFFORT = EnvStr("0.9")
    SGLANG_INKLING_RS_MM_PREPROCESS = EnvBool(True)

    # ===================================================================
    # DSA backend (GLM 5 and DeepSeek V3.2)
    # ===================================================================
    SGLANG_DSA_FUSE_TOPK = EnvBool(True)
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBool(False)
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    # Requires a DeepGEMM paged-MQA scheduler supporting batches > SM count
    # (0.2.3 does); set to 0 for an older DeepGEMM.
    SGLANG_DSA_FULL_BATCH_MQA = EnvBool(True)
    # prims_ts DSA on SM107 (VR200, Rubin): off keeps dev's SM100/SM103-only
    # startup check; on admits SM107 and needs a FlashInfer Prims-TS tree with
    # flashinfer-ai/flashinfer#8186 (sparse MLA planning on SM107).
    SGLANG_ENABLE_PRIMS_TS_SM107 = EnvBool(False)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)
    SGLANG_DISABLE_DSA_INDEXER_FUSION = EnvBool(False)
    # HIP analog of CUDA SGLANG_DISABLE_DSA_INDEXER_FUSION (default on).
    # Set 1 for the legacy split writer. Omits Hadamard; pre-quant logits match.
    # Separate from the CUDA flag: that one also packs wk+weights_proj.
    SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER = EnvBool(False)
    # Opt-in perf path for --dsa-prefill-backend flashmla_sparse_q8: fuse the
    # absorbed q bmm with the nope/rope concat + fp8 cast so q is written
    # directly in fp8 ("born fp8") and the standalone concat-cast kernel
    # disappears.  Not bit-exact vs the default path (same rounding stages,
    # different GEMM accumulation order), hence default OFF until accuracy-
    # gated (oracle + full-set gsm8k).
    SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q = EnvBool(False)
    # Opt-in: run the born-fp8 q-prep (absorbed bmm + concat + fp8 cast,
    # ~173us/layer-call) on alt_stream underneath the DSA indexer — the two
    # chains fork independently from the q_a_layernorm output.  Requires
    # SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q; eager-prefill-only via the born
    # predicate.  Coarse per-layer join keeps the single-slot born-q buffer
    # WAR-safe.
    SGLANG_ENABLE_DSA_Q8KV8_QPREP_OVERLAP = EnvBool(False)
    # Opt-in: fuse the Q8KV8 non-prefix KV prep — cast-concat k/k_rope
    # directly into the persistent fp8 kv buffer and zero the pad band in one
    # Triton kernel (replaces bf16 _cat + copy_ cast + zero_ tail).
    SGLANG_ENABLE_DSA_Q8KV8_KV_CAT_FUSION = EnvBool(False)
    # Q8KV8 born-fp8 q-prep codegen: "auto" = per-K Triton dispatch (default);
    # "cuda" = the hand-written SM90 WGMMA kernel (bitwise identical to the
    # Triton two_dot variant, 1.16-1.38x faster across GLM/DS shapes).
    SGLANG_OPT_Q8KV8_QPREP_VARIANT = EnvStr("auto")

    # ===================================================================
    # Muse Spark
    # ===================================================================
    # Fuse the unpooler all-gather into the FFN-boundary norm-residual kernel.
    SGLANG_ENABLE_MUSE_SPARK_AG_NORM_FUSION = EnvBool(True)
    SGLANG_USE_CUSTOM_ROUTER_SUM = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_USE_TRITON_ROUTER_SUM"
    )

    # ===================================================================
    # MiniMax M3
    # ===================================================================
    SGLANG_OPT_USE_BF16_ROUTER_GEMM = EnvBool(True)
    SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE = EnvBool(False)
    SGLANG_DISABLE_MSA = EnvBool(False)
    SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH = EnvBool(False)
    # Kill switch for the derived fp8 attention-GEMM mode (m3_fp8_attn_gemm_enabled):
    # forces the pre-fp8 behavior (bf16 indexer + widening sparse path, bf16 q)
    # even when kv_cache_dtype fp8_e4m3 + trtllm_mha + SM100 would activate it.
    SGLANG_DISABLE_M3_FP8_ATTN_GEMM = EnvBool(False)
    # Vision tokens per block in the ViT MLP and projector; <= 0 runs all at once.
    SGLANG_MINIMAX_VIT_TOKEN_BLOCK = EnvInt(32768)
    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)
    # Fused JIT store (minimax_store_kv_index) of main+index K/V instead of separate
    # set_*_buffer copies; falls back when main/index dtypes differ or non-CUDA.
    SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE = EnvBool(True)
    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)

    # MiniMax-M3 sparse-attention toggles for ROCm.
    # Share one index top-k across every N sparse layers; 1 (default) disables
    # sharing so every sparse layer computes its own selection. Values > 1 are a
    # lossy opt-in: skip layers attend KV blocks chosen by an earlier layer.
    # Only honored on ROCm gfx95 without two-batch overlap; elsewhere the
    # backend pins 1.
    SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ = EnvInt(1)
    # Opt-in decode indexer context partitioning (gfx950 TP4, SM90 TP4/TP8). Keeps
    # the index cache replicated; gathers Q and exchanges local top-k candidates.
    SGLANG_MINIMAX_M3_INDEXER_CP = EnvBool(False)
    # gfx95: lightning-indexer K cache in fp8_e4m3fn (bf16 q x fp8 k in the scorers);
    # main attention K/V keep kv_cache_dtype.
    SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE = EnvBool(True)
    # Run the sparse prefill main attention through AITER's Gluon paged attention
    # instead of the Triton kernel. Unsupported cases fall back to Triton.
    SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL = EnvBool(True)
    # MiniMax M3 NPU prefill MAIN-attention: route the sparse main attention through
    # the native Ascend FA op `torch.ops.npu.npu_fused_infer_attention_score` (FIA)
    # with a per-query CUSTOM block_table
    SGLANG_MINIMAX_NPU_PREFILL_FIA = EnvBool(True)
    # MiniMax-M3 NPU sparse INDEXER (decode + verify topk block selection): route
    # through the native AscendC packed indexer op instead of the Triton indexer.
    SGLANG_MINIMAX_NPU_NATIVE_INDEXER = EnvBool(False)
    # MiniMax-M3 NPU sparse MAIN-attention (decode-main + verify-main): route the
    # sparse main attention through the native AscendC sparse-attention op with the
    # cached block_table override.
    SGLANG_MINIMAX_NPU_NATIVE_ATTN = EnvBool(False)
    # MiniMax-M3 on ROCm force-disables custom all-reduce in its model override
    # (arg_groups/overrides.py) when aiter all-reduce fusion is off. Set this to
    # opt back in and keep custom/quick all-reduce enabled -- e.g. to run the
    # INT4 quick-reduce path via ROCM_QUICK_REDUCE_QUANTIZATION={INT4,INT6,INT8}.
    SGLANG_M3_ALLOW_CUSTOM_AR = EnvBool(False)

    # ===================================================================
    # Gemma 4
    # ===================================================================
    # Fused TP all-reduce + RMSNorm -> residual add -> RMSNorm -> per-128 FP8
    # quant of the dense layer stack as one NVLS kernel per site, for extend
    # batches of MIN..MAX tokens (SM90 with NVLink multicast, block-FP8 weights).
    # MIN_TOKENS 0 takes the per-TP-size default (DEFAULT_MIN_TOKENS in
    # srt/layers/communication/gemma4_ar_fusion.py).
    SGLANG_GEMMA4_AR_FUSION = EnvBool(False)
    SGLANG_GEMMA4_AR_FUSION_MIN_TOKENS = EnvInt(0)
    SGLANG_GEMMA4_AR_FUSION_MAX_TOKENS = EnvInt(8192)

    # ===================================================================
    # Kimi K3
    # ===================================================================
    # MNNVL fused all-reduce (bf16, TP8): zero-copy 1shot multicast-push for
    # small messages and in-place NVLS 2shot on symmetric-memory tensors for
    # large ones, with an optional fused residual add. Covers the KDA o_proj
    # output and the latent|shared MoE reduce; everything else falls back to
    # the regular all-reduce path. Auto-enabled on SM100/SM103 when
    # CustomAllReduceV2 with multicast is available; set 0/1 to override in
    # either direction (the K3.1 GB switch forces it on unless set). See
    # srt/layers/communication/k3_ar_fusion.py.
    SGLANG_K3_AR_FUSION = EnvBool(SGLANG_ENABLE_NV_MEGA_RUNTIME_K31_GB_OPTIMIZATION.get)
    # K3 AR fusion on VR200 (SM107): run the NVLS pull all-reduces of large
    # (prefill-sized, > 4 MB) messages -- the attention/KDA o_proj output, the
    # MoE latent+norm and the overlapped MoE shared AR -- with more CTAs. The
    # stock large-message fallbacks (4 / 24 / 4 blocks) were tuned on GB300;
    # the CTA count only partitions the rows (bit-identical results).
    SGLANG_K3_AR_LARGE_PULL_TUNING = EnvBool(False)
    # SM107 (VR200) joins the SM100/SM103 gates of the K3 paths here (the
    # fusion auto-probe and the SP collective; an SM107 GPU with no tuning JSON
    # takes GB300's for the same world size). Their JIT kernels build for the
    # SM100 family, which SM107 runs. Off: SM107 behaves as before. See
    # k3_ar_fusion.k3_sm_supported.
    SGLANG_ENABLE_K3_SM107 = EnvBool(False)
    # K3 SP-MoE fused residual + reduce-scatter and matching all-gather over
    # CustomAllReduceV2's MNNVL push workspace. Auto-probed for checked-in
    # TP4/TP8 GB300 and TP16 GB200/GB300 tuning; set 0/1 to override. See
    # srt/layers/communication/k3_sp_collective.py.
    SGLANG_K3_SP_COLLECTIVE = EnvBool(False)
    # DCP MLA decode: push each peer's head partials straight into its
    # CustomAllReduceV2 push slot and combine by LSE in one kernel, in place of
    # the pack + all-to-all + combine (W_UV-before-reduce path, 128-wide heads).
    # Rows per call up to SGLANG_DCP_PUSH_COMBINE_MAX_ROWS. See
    # kernels/ops/attention/dcp_push_combine.py.
    SGLANG_ENABLE_DCP_PUSH_COMBINE = EnvBool(True)
    SGLANG_DCP_PUSH_COMBINE_MAX_ROWS = EnvInt(_k31_gb_best(1536, 1024))
    # Keep K3's post-MoE residual stream token-sharded between consecutive
    # SP-MoE layers. The next attention-residual aggregation and snapshot
    # bank write run on the local shard, then only the normalized attention
    # input is all-gathered. Works with NCCL or SGLANG_K3_SP_COLLECTIVE.
    SGLANG_K3_SP_ATTN_RES = EnvBool(False)
    # Merge the router gate and routed_expert_down_proj weights so the K3 MoE
    # front reads hidden_states once, and run the top-k plus the bf16 cast in one
    # epilogue kernel. See kernels/ops/moe/moe_front.py. Default on.
    SGLANG_K3_FUSED_FRONT = EnvBool(True)
    # TP-shard the K3 MoE front's router + latent-down rows (KimiK3MoE._front_shard)
    # and the latent up_proj tail above the fused-GEMV range (k3_ar_fusion
    # gemm_ag_up_fits). Online weight updates are rejected while on.
    SGLANG_K3_SHARDED_MOE = EnvBool(True)
    # Kimi-K3.1 on GB300 (SM103): the measured per-M TGV tactics of the decode
    # dense GEMMs, the side-stream [f_a | b] TGV, and attention side streams up
    # to 288 rows (kimi_k3._k31_gb300_tuning). 0 restores the generic picks.
    SGLANG_K31_DECODE_GEMM_TACTICS = EnvBool(True)
    # Kimi-K3.1 on VR200 (SM107), default off: the per-M TGV tactics of the
    # decode dense GEMMs measured on SM107 (through the PD-decode DFLASH verify
    # rows), the side-stream [f_a | b] TGV, and the SM107 attention side-stream
    # row caps below (kimi_k3._k31_sm107_tuning). Off: SM107 keeps the generic
    # picks and the 128-row side streams.
    SGLANG_K31_SM107_GEMM_TACTICS = EnvBool(False)
    # Optional path of a JSON table that replaces the built-in SM107 table
    # (benchmark/kernels/k31_sm107/sweep_sm107_decode_gemms.py writes it).
    SGLANG_K31_SM107_GEMM_TABLE = EnvStr(None)
    # Under SGLANG_K31_SM107_GEMM_TACTICS on SM107: the largest graphed rows
    # whose KDA [f_a | b] GEMM runs on the side stream beside [q,k,v,g]
    # (default 1152 = the largest verify rows, the stable table's
    # side_cap_bfa), and whose MLA output gate runs on the side stream beside
    # attention (default 128, today's cap: raising it measured no gain).
    SGLANG_K31_SM107_BFA_SIDE_MAX_TOKENS = EnvInt(1152)
    SGLANG_K31_SM107_GATE_SIDE_MAX_TOKENS = EnvInt(128)
    # Use the ROCm radix-4 router for covered K3 top-k workloads.
    SGLANG_K3_RADIX4_TOPK = EnvBool(False)
    # Kimi-K3.1 decode routed MoE on SM100/SM103: replace flashinfer's deferred-finalize
    # trtllm-gen MXFP4 call with the JIT routing + persistent FC1/FC2 kernels
    # (kernels/ops/moe/k31_decode_moe.py) for covered TP-MoE shapes (896 local
    # experts, intermediate 384, top-16, <= 256 tokens); anything else runs flashinfer.
    SGLANG_K31_DECODE_MOE = EnvBool(False)

    # Kimi-K3.1 VR200 prefill paths, default off; tuning values are the ones the
    # prefill/decode-disaggregated c=40 / c=48 points ran, and those points turn
    # them on through benchmark/kimi_k31_vr200/mr_tree/mid_pd_prefill.env.
    # Barrier-free 2shot push band of SGLANG_K3_AR_FUSION past its 1shot window,
    # up to _MAX_KB of payload, through CustomAllReduceV2's push slots. _FINALIZE
    # keeps the deferred MoE finalize fused into the latent all-reduce in this
    # band; _MC forces the multicast fan-out for every variant.
    SGLANG_K3_AR_PUSH2 = EnvBool(False)
    SGLANG_K3_AR_PUSH2_MAX_KB = EnvInt(8192)
    SGLANG_K3_AR_PUSH2_FINALIZE = EnvBool(False)
    # Under SGLANG_K3_AR_FUSION, decode batches up to _MAX_TOKENS run the latent
    # MoE up_proj TP-sharded with a multicast all-gather (gemm_ag).
    SGLANG_K3_SHARDED_LATENT_UP = EnvBool(False)
    SGLANG_K3_SHARDED_LATENT_UP_MAX_TOKENS = EnvInt(512)
    # NVFP4 route+quant preparation staged before the K3 MoE top-k where it covers
    # the batch (SM107, NVFP4 MoE, under SGLANG_K3_AR_FUSION).
    SGLANG_K31_PF_NVFP4_ROUTE_QUANT = EnvBool(False)
    # attn_residual: the score kernel in 1024-wide H tiles and the run-time-nvb
    # combine kernel (aggregate_stream).
    SGLANG_K31_PF_ATTN_RES_TILED = EnvBool(False)
    # TP-shard the fused K3 MoE front's router + latent-down rows with the shared
    # experts' gate_up rows in one GEMM, all-gathered through CustomAllReduceV2,
    # for batches of _MIN_TOKENS to _MAX_TOKENS rows. Not SGLANG_K3_SHARDED_MOE.
    SGLANG_K3_SHARDED_MOE_FRONT = EnvBool(False)
    SGLANG_K3_SHARDED_MOE_FRONT_MIN_TOKENS = EnvInt(18)
    SGLANG_K3_SHARDED_MOE_FRONT_MAX_TOKENS = EnvInt(512)
    # K3.1 MoE routing in one kernel for batches of 1-512 rows: top-16, mxfp8 quant
    # of the routed input, padded-row mask and trtllm-gen routing metadata.
    SGLANG_K31_FUSED_ROUTING = EnvBool(False)
    # K3.1 M <= 8 prepared trtllm-gen routing metadata from route_quant_fused, the
    # prepared FP4 MoE bodies (kernels/ops/moe/trtllm_prepared_fp4.py) and the NVFP4
    # route+quant handoff.
    SGLANG_K31_PF_PREPARED_ROUTING = EnvBool(False)
    # K3.1 eager prefill on plain-TP MoE: sequence-parallel layer tail for EXTEND
    # batches of at least _MIN_TOKENS rows without CP or PP. _COMM: k3 (NVLS pull
    # reduce-scatter / multicast all-gather on CustomAllReduceV2) or nccl; the
    # _BLOCKS values are CTAs of the k3 kernels.
    SGLANG_K31_PREFILL_SP_TAIL = EnvBool(False)
    SGLANG_K31_PREFILL_SP_TAIL_MIN_TOKENS = EnvInt(10240)
    SGLANG_K31_PREFILL_SP_TAIL_COMM = EnvStr("k3")
    SGLANG_K31_PREFILL_SP_TAIL_RS_BLOCKS = EnvInt(96)
    SGLANG_K31_PREFILL_SP_TAIL_AG_BLOCKS = EnvInt(32)
    SGLANG_K31_PREFILL_SP_TAIL_SHARED_RS_BLOCKS = EnvInt(16)
    # Prefill forward all-gathers (DSA indexer top-k, DCP prefix gathers) on the
    # direct multicast all-gather through one symmetric buffer of _MAX_MB per GPU;
    # bitwise equal to NCCL. _PAD_MAX bounds padded / real rows; _BLOCKS = CTAs.
    SGLANG_K31_PF_CUSTOM_AG = EnvBool(False)
    SGLANG_K31_PF_CUSTOM_AG_MAX_MB = EnvFloat(768.0)
    SGLANG_K31_PF_CUSTOM_AG_PAD_MAX = EnvFloat(1.5)
    SGLANG_K31_PF_CUSTOM_AG_BLOCKS = EnvInt(32)
    # K3.1 KDA prefill kernels, read at import; 0 / the default is the stock path.
    # SEGSCAN: segments of the segmented h scan for chunks of at least _MIN_T
    # tokens (_TORCH_GLUE uses the torch setup glue). TUNE_PIN: pin the KDA
    # autotune configs. QKV_PREP: one-launch q/k/v prep. H_PREFETCH: prefetch the
    # next chunk's k tiles. CONV_BLOCK_N: causal-conv channels per program (256,
    # 512 or 1024).
    SGLANG_K31_KDA_SEGSCAN = EnvInt(0)
    SGLANG_K31_KDA_SEGSCAN_MIN_T = EnvInt(16384)
    SGLANG_K31_KDA_SEGSCAN_TORCH_GLUE = EnvBool(False)
    SGLANG_K31_KDA_TUNE_PIN = EnvInt(0)
    SGLANG_K31_KDA_QKV_PREP = EnvInt(0)
    SGLANG_K31_KDA_H_PREFETCH = EnvInt(0)
    SGLANG_K31_KDA_CONV_BLOCK_N = EnvInt(256)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS = EnvInt(2)
    SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN = EnvInt(6144)

    # ===================================================================
    # ERNIE 5.5
    # ===================================================================
    # Fusions not bit-exact with the eager ops: the input RMSNorm into mhc_pre,
    # and the VHA postmix into the wo_a fp8 quant.
    SGLANG_OPT_ERNIE5_5_FUSE_INEXACT = EnvBool(True)

    # ===================================================================
    # Symmetric memory
    # ===================================================================
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Qwen3.5 and GDN
    SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV = EnvBool(True)

    # ===================================================================
    # Plugin system
    # ===================================================================
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")
    # Read by the NVIDIA megakernel plugin, not by SGLang: a comma-separated
    # slot list that replaces the plugin's default. Empty runs the stock paths.
    SGLANG_NVIDIA_MEGAKERNEL_SLOTS = EnvTuple(())

    # ===================================================================
    # KV-Canary and Token-Oracle (testing only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_ENABLE_INFO = EnvBool(False)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_DETOKENIZER_SWAP_AT_OUTPUT = EnvInt(-1)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)

    # ===================================================================
    # Rust server
    # ===================================================================
    SGLANG_RUST_SERVER = EnvBool(False)
    # Build a missing Rust extension from source (auto), require a bundled or
    # cached extension (never), or rebuild the local cache entry (force).
    SGLANG_RUST_BUILD_MODE = EnvStr("auto")
    # Most batched requests one /generate HTTP call may expand into.
    SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ = EnvInt(4096)

    # ===================================================================
    # Weight Cache Daemon
    # ===================================================================
    # Paths the daemon and the engine ranks it serves must agree on. Both are
    # format templates and must keep the {device_uuid} placeholder: each daemon
    # is keyed by the physical GPU it runs on, so a GPU-independent path would
    # let one job's client discover another job's daemon.
    SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.sock"
    )
    SGLANG_WEIGHT_CACHE_READY_TEMPLATE = EnvStr(
        "/tmp/sglang_weight_cache_{device_uuid}.ready"
    )


envs = Envs()
EnvField._allow_set_name = False


def exportable_env_vars() -> dict[str, str]:
    return {
        field.name: _exportable_value(os.environ[field.name])
        for field in sorted(
            (value for value in vars(Envs).values() if isinstance(value, EnvField)),
            key=lambda field: field.name,
        )
        if not field.secret and field.name in os.environ
    }


def _exportable_value(value: str) -> str:
    try:
        value.encode()
    except UnicodeEncodeError:
        return (
            _NON_UTF8_PREFIX
            + base64.b64encode(value.encode(errors="surrogateescape")).decode()
        )
    return value


class _DeprecatedEnv:
    """One deprecated env var: warn if it is set, and optionally forward its
    (possibly transformed) value to a replacement env var."""

    def __init__(
        self,
        replacement: Optional[str] = None,
        transform: Optional[Callable[[str], str]] = None,
        note: Optional[str] = None,
    ):
        self.replacement = replacement
        self.transform = transform
        self.note = note

    def apply(self, old_name: str):
        if old_name not in os.environ:
            return
        message = f"Environment variable {old_name} is deprecated."
        if self.replacement is not None:
            message += f" Please use {self.replacement} instead."
        if self.note is not None:
            message += f" {self.note}"
        warnings.warn(message)
        if self.replacement is not None:
            value = os.environ[old_name]
            if self.transform is not None:
                value = self.transform(value)
            os.environ[self.replacement] = value


# The single registry for deprecated environment variables, processed once at
# import by _handle_deprecated_envs(). Add new deprecations here instead of
# ad-hoc warnings. For a rename where the old name must keep working through a
# descriptor, use EnvBoolWithAlias / EnvIntWithAlias instead.
_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {
    "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION": _DeprecatedEnv(
        note=(
            "Pass --flashinfer-allreduce-fusion-backend cutedsl instead. "
            "Without it an eligible model auto-enables the legacy mnnvl "
            "backend rather than the CuTe DSL fusion."
        )
    ),
    "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES": _DeprecatedEnv(
        note="One workspace per process is now an invariant, not a limit."
    ),
    # Removed without replacement.
    "SGLANG_ENABLE_CP_V2": _DeprecatedEnv(
        note="Strategy-based prefill context parallelism is now the only generic implementation."
    ),
    "SGLANG_TRACE_QWEN35_FINAL_NORM": _DeprecatedEnv(),
    "SGLANG_DSPARK_FAST_SAMPLING": _DeprecatedEnv(
        note="DSpark draft sampling always uses Gumbel-max."
    ),
    "SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV": _DeprecatedEnv(
        note="A one-row step of SGLANG_DSPARK_USE_GREEDY_DRAFT on a vanilla markov "
        "head fuses the bias GEMV."
    ),
    "SGLANG_DSPARK_FOLDED_SAMPLING": _DeprecatedEnv(
        note="A folded DSpark draft proposal samples; set "
        "SGLANG_DSPARK_USE_GREEDY_DRAFT=1 for an argmax-only draft, or "
        "SGLANG_DSPARK_FOLDED_PROPOSAL=0 to keep the proposal out of the graph."
    ),
    "SGLANG_SKIP_RUST_TESTS": _DeprecatedEnv(
        note="The Rust tests run in pr-test-rust-workspace.yml now."
    ),
    "SGLANG_QWEN35_NATIVE_FINAL_NORM": _DeprecatedEnv(),
    "SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK": _DeprecatedEnv(
        note="Buffer-mode anchor pinning is always on; set "
        "SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP=0 to disable it."
    ),
    "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": _DeprecatedEnv(
        note="Deterministic inference on triton no longer aligns prefill chunks."
    ),
    # Replaced by CLI flags.
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": _DeprecatedEnv(
        note="Please use '--enable-prefill-delayer' instead."
    ),
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": _DeprecatedEnv(
        note="Please use '--prefill-delayer-max-delay-passes' instead."
    ),
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": _DeprecatedEnv(
        note="Please use '--prefill-delayer-token-usage-low-watermark' instead."
    ),
}


def _handle_deprecated_envs():
    for old_name, deprecation in _DEPRECATED_ENVS.items():
        deprecation.apply(old_name)

    # Rewrite the legacy SGL_ prefix to SGLANG_ (names not covered above).
    for key, value in list(os.environ.items()):
        if key.startswith("SGL_") and key not in _DEPRECATED_ENVS:
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


def third_party_cache_defaults() -> Dict[str, str]:
    base = os.path.expanduser(envs.SGLANG_CACHE_DIR.get())
    return {
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "CUDA_CACHE_PATH": os.path.join(base, "nv"),
        # TileLang compiles the DeepSeek-V4 MHC prenorm kernels; left at its own
        # default the burst is invisible to anyone warming, mounting or baking
        # SGLANG_CACHE_DIR, and gets paid again on every cold container.
        "TILELANG_CACHE_DIR": os.path.join(base, "tilelang"),
        "HUMMING_CACHE_DIR": os.path.join(base, "humming", "cache"),
        "HUMMING_TMP_DIR": os.path.join(base, "humming", "tmp"),
        # FlashInfer appends ".cache/flashinfer" to this base itself, so this
        # is the base dir rather than the final cache dir.
        "FLASHINFER_WORKSPACE_BASE": base,
    }


def redirect_third_party_caches():
    """Point third-party JIT caches at SGLANG_CACHE_DIR, so a run's compiled
    kernels can be cleaned, warmed or volume-mounted as one directory.

    Must be called early. The redirect silently does nothing if either of
    these has already happened:

    - FlashInfer was imported. It resolves its workspace at import time.
    - Inductor made its first ``cache_dir()`` call. That call setdefaults
      TORCHINDUCTOR_CACHE_DIR itself.
    - Humming resolved its cache or temporary directory for the first time.
    """
    for key, value in third_party_cache_defaults().items():
        os.environ.setdefault(key, value)


_handle_deprecated_envs()

# Trigger auto-injection of CUDA coredump env vars when SGLANG_CUDA_COREDUMP=1.
# Best-effort; for strict guarantees, set CUDA_* env vars in the shell before
# launching Python. Imported conditionally to keep the default import of this
# module free of non-stdlib side effects.
if envs.SGLANG_CUDA_COREDUMP.get():
    import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip
