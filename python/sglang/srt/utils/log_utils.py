from __future__ import annotations

import functools
import gc
import json
import logging
import os
import re
import socket
import sys
import time
import warnings
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import parent_process
from typing import List, Optional, Sequence, Union

import orjson
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import is_flashinfer_available

logger = logging.getLogger(__name__)

__all__ = [
    "create_log_targets",
    "log_json",
    "configure_hf_hub_logger",
    "suppress_noisy_warnings",
    "filter_noisy_warnings",
    "suppress_other_loggers",
    "rank0_log",
    "configure_logger",
    "print_warning_once",
    "print_info_once",
    "log_info_on_rank0",
    "log_debug_on_rank0",
    "gc_object_counts",
    "configure_gc_warning",
    "freeze_gc",
    "configure_gc_logger",
]


def create_log_targets(
    *, targets: Optional[List[str]], name_prefix: str
) -> List[logging.Logger]:
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    return [_create_log_target(t, name_prefix) for t in targets]


def _create_log_target(target: str, name_prefix: str) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(directory: str, name_prefix: str) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    filename = os.path.join(directory, f"{hostname}_{rank}.log")
    handler = TimedRotatingFileHandler(
        filename, when="H", backupCount=0, encoding="utf-8"
    )
    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{hostname}_{rank}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)


def configure_hf_hub_logger():
    """Route Hugging Face Hub messages through the application's logger once."""
    from huggingface_hub.utils import logging as hf_logging

    # CLI model detection can contact Hub before configure_logger runs.
    # basicConfig leaves any existing application logging setup intact.
    logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    hub_logger = hf_logging.get_logger()
    for handler in hub_logger.handlers[:]:
        # Hub installs a plain StreamHandler and also propagates to the root.
        # Keep file handlers and custom handler subclasses intact.
        if type(handler) is logging.StreamHandler:
            hub_logger.removeHandler(handler)
    hf_logging.enable_propagation()


def suppress_noisy_warnings():
    """Suppress known noisy third-party output: warnings, and the HF Hub and httpx loggers."""
    configure_hf_hub_logger()
    filter_noisy_warnings()

    # Suppress noisy third-party HTTP loggers.
    # huggingface_hub uses httpx which logs every HTTP request at INFO level.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def filter_noisy_warnings():
    """Drop known noisy warnings from third-party libraries; logging is untouched.

    pytest re-enables DeprecationWarning per test, so test/pytest.ini repeats
    the DeprecationWarning entries below; keep the two in sync.
    """
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )
    warnings.filterwarnings(
        "ignore",
        message="The cuda.cudart module is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The cuda.nvrtc module is deprecated",
        category=FutureWarning,
    )

    # flashinfer and quack still import pre-4.8 cutlass-dsl paths. Matching only
    # those packages keeps sglang's own uses of the old paths visible.
    third_party = "flashinfer|quack"
    # (category, message regex, emitting packages, or None for any)
    noisy_warnings = [
        (
            DeprecationWarning,
            re.escape("Using `struct.scalar` as pointer is deprecated."),
            None,
        ),
        (
            UserWarning,
            re.escape(
                "NamedBarrier wait also arrives on the barrier. "
                "Routing call to NamedBarrier.arrive_and_wait()."
            ),
            None,
        ),
        (DeprecationWarning, r"builtin type \w+ has no __module__ attribute", None),
        (DeprecationWarning, r"cutlass\.\S+ is deprecated; use cutlass\.", third_party),
        (DeprecationWarning, r"Arch from arch\.py is deprecated", third_party),
    ]
    for cat, message, packages in noisy_warnings:
        module = rf"({packages})(\.|$)" if packages else ""
        warnings.filterwarnings("ignore", message=message, category=cat, module=module)

    # cutlass-dsl emits some of these inside `catch_warnings()+simplefilter("always")`,
    # which bypasses filterwarnings; override showwarning to drop them too.
    if not getattr(warnings.showwarning, "_sglang_noisy_warning_filter", False):
        prev_showwarning = warnings.showwarning

        def _filtered_showwarning(message, category, filename, *args, **kwargs):
            for cat, pattern, packages in noisy_warnings:
                if (
                    issubclass(category, cat)
                    and re.match(pattern, str(message))
                    and (
                        packages is None
                        or re.search(rf"[/\\]({packages})[/\\]", filename)
                    )
                ):
                    return
            prev_showwarning(message, category, filename, *args, **kwargs)

        _filtered_showwarning._sglang_noisy_warning_filter = True
        warnings.showwarning = _filtered_showwarning


def suppress_other_loggers():
    suppress_noisy_warnings()

    try:
        from vllm.logger import logger as vllm_default_logger
    except ImportError:
        return

    vllm_default_logger.setLevel(logging.WARN)
    logging.getLogger("vllm.distributed.device_communicators.pynccl").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.distributed.device_communicators.shm_broadcast").setLevel(
        logging.WARN
    )
    logging.getLogger("vllm.config").setLevel(logging.ERROR)


def rank0_log(msg: str):
    from sglang.srt.distributed import (
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized() or get_parallel().tp_rank == 0:
        logger.info(msg)


def configure_logger(server_args, prefix: str = ""):
    if SGLANG_LOGGING_CONFIG_PATH := os.getenv("SGLANG_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SGLANG_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting SGLANG_LOGGING_CONFIG_PATH from env with "
                f"{SGLANG_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(SGLANG_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = orjson.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    maybe_ms = ".%(msecs)03d" if envs.SGLANG_LOG_MS.get() else ""
    format = f"[%(asctime)s{maybe_ms}{prefix}] %(message)s"
    logging.basicConfig(
        # Runs before publish, and for multimodal_gen's ServerArgs, which
        # never publishes these bags -- so the record, not the bag.
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    configure_hf_hub_logger()

    # Suppress noisy httpx/httpcore loggers in every process that calls
    # configure_logger (main, scheduler, detokenizer). Spawned subprocesses
    # don't inherit the parent's logger state, so this must run here too.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Server-sent hub warnings (e.g. the unauthenticated-request / HF_TOKEN
    # hint) are deduplicated per process, so a TP-N launch repeats each one N
    # times. Keep them only in the launching process -- every worker (scheduler,
    # detokenizer, DP controller, ...) is spawned via multiprocessing, whether
    # or not it passes a log prefix -- where they are printed exactly once.
    if parent_process() is not None:
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    if is_flashinfer_available():
        from flashinfer.jit.core import logger as flashinfer_logger

        flashinfer_logger.setLevel(logging.ERROR)


@functools.lru_cache(None)
def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg)


@functools.lru_cache(None)
def print_info_once(msg: str) -> None:
    logger.info(msg)


def log_info_on_rank0(logger, msg):

    try:
        if torch.distributed.is_initialized() and get_parallel().tp_rank == 0:
            logger.info(msg)
    except Exception as e:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.info(f"{msg} (rank-check failed: {e})")
        else:
            logger.info(f"{msg} (rank-check failed: {e})")


def log_debug_on_rank0(logger, msg):
    """
    Log a debug message only on tensor model parallel rank 0.
    Falls back to logging if distributed is not initialized or error occurs.
    """

    try:
        if torch.distributed.is_initialized() and get_parallel().tp_rank == 0:
            logger.debug(msg)
    except Exception as e:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.debug(f"{msg} (rank-check failed: {e})")
        else:
            logger.debug(f"{msg} (rank-check failed: {e})")


def gc_object_counts():
    import gc

    g0 = len(gc.get_objects(0))
    g1 = len(gc.get_objects(1))
    g2 = len(gc.get_objects(2))
    return g0, g1, g2


def configure_gc_warning(warn_threshold_secs):
    import gc

    gc_start_time = {}

    def gc_callback(phase, info):
        gen = info.get("generation", "?")
        if phase == "start":
            gc_start_time[gen] = time.time()
        elif phase == "stop":
            duration = time.time() - gc_start_time.get(gen, time.time())
            if duration > warn_threshold_secs:
                g0, g1, g2 = gc_object_counts()
                logger.warn(
                    f"LONG GARBAGE COLLECTION DETECTED | Generation {gen} | Duration: {duration:.4f}s | # Objects: gen0={g0}, gen1={g1}, gen2={g2} | "
                    f"This may cause latency jitter. Consider calling the freeze_gc API after sending a few warmup requests."
                )

    gc.callbacks.append(gc_callback)


def freeze_gc(context: str, gc_threshold: Optional[Sequence[int]] = None):
    if gc_threshold is not None:
        gc.set_threshold(*gc_threshold)
    g0_before, g1_before, g2_before = gc_object_counts()
    gc.freeze()
    g0_after, g1_after, g2_after = gc_object_counts()
    logger.debug(
        f"Freezing GC in {context} process. "
        f"gen0: {g0_before}->{g0_after}, "
        f"gen1: {g1_before}->{g1_after}, "
        f"gen2: {g2_before}->{g2_after}"
    )


def configure_gc_logger():
    logger.info("Enable GC Logger")

    gc_start_time = {}

    def gc_callback(phase, info):
        gen = info.get("generation", "?")
        if phase == "start":
            gc_start_time[gen] = time.time()
            logger.info(f"GC start: Time {time.time()} | Generation {gen}")
        elif phase == "stop":
            duration = time.time() - gc_start_time.get(gen, time.time())
            collected = info.get("collected", "?")
            uncollectable = info.get("uncollectable", "?")
            logger.info(
                f"GC end: Time {time.time()} | Generation {gen} | "
                f"Duration: {duration:.4f}s | Collected: {collected} | Uncollectable: {uncollectable} "
                f"{'(LONG GC)' if duration > 0.1 else ''}"
            )

    gc.callbacks.append(gc_callback)
