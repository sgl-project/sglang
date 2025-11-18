# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/logger.py
"""Logging configuration for sglang.multimodal_gen."""
import argparse
import datetime
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache, partial
from logging import Logger
from types import MethodType
from typing import Any, cast

import sglang.multimodal_gen.envs as envs

SGL_DIFFUSION_CONFIGURE_LOGGING = envs.SGL_DIFFUSION_CONFIGURE_LOGGING
SGL_DIFFUSION_LOGGING_CONFIG_PATH = envs.SGL_DIFFUSION_LOGGING_CONFIG_PATH
SGL_DIFFUSION_LOGGING_LEVEL = envs.SGL_DIFFUSION_LOGGING_LEVEL
SGL_DIFFUSION_LOGGING_PREFIX = envs.SGL_DIFFUSION_LOGGING_PREFIX

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0;0m"

_FORMAT = (
    f"{SGL_DIFFUSION_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(filename)s: %(lineno)d] %(message)s"
)

# _FORMAT = "[%(asctime)s] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "sgl_diffusion": {
            "class": "sglang.multimodal_gen.runtime.utils.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "sgl_diffusion": {
            "class": "logging.StreamHandler",
            "formatter": "sgl_diffusion",
            "level": SGL_DIFFUSION_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "sgl_diffusion": {
            "handlers": ["sgl_diffusion"],
            "level": "WARNING",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["sgl_diffusion"],
        "level": "DEBUG",
    },
    "version": 1,
    "disable_existing_loggers": False,
}


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        logging.Formatter.__init__(self, fmt, datefmt, style)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ColoredFormatter(NewLineFormatter):
    """A logging formatter that adds color to log levels."""

    LEVEL_COLORS = {
        logging.ERROR: RED,
        logging.WARNING: YELLOW,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Adds color to the log level name."""
        original_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno)
        if color:
            record.levelname = f"{color}{original_levelname}{RESET}"

        formatted_message = super().format(record)

        if color:
            record.levelname = original_levelname

        return formatted_message


class SortedHelpFormatter(argparse.HelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


@lru_cache
def _print_info_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.info(msg, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str) -> None:
    # Set the stacklevel to 2 to print the original caller's line info
    logger.warning(msg, stacklevel=2)


def _get_rank_info():
    """Get rank and local rank from environment variables."""
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError):
        rank = 0
        local_rank = 0
    return rank, local_rank


def _log_process_aware(
    level: int,
    logger_self: Logger,
    msg: object,
    *args: Any,
    main_process_only: bool,
    local_main_process_only: bool,
    **kwargs: Any,
) -> None:
    """Helper function to log a message if the process rank matches the criteria."""
    rank, local_rank = _get_rank_info()
    is_main_process = rank == 0
    is_local_main_process = local_rank == 0

    should_log = (
        not main_process_only
        and not local_main_process_only
        or (main_process_only and is_main_process)
        or (local_main_process_only and is_local_main_process)
    )

    if should_log:
        # stacklevel=3 to show the original caller's location,
        # as this function is called by the patched methods.
        logger_self.log(level, msg, *args, stacklevel=3, **kwargs)


class _SGLDiffusionLogger(Logger):
    """
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the :class:`logging.Logger`
        instance to avoid conflicting with other libraries such as
        `intel_extension_for_pytorch.utils._logger`.
    """

    def info_once(self, msg: str) -> None:
        """
        As :meth:`info`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_info_once(self, msg)

    def warning_once(self, msg: str) -> None:
        """
        As :meth:`warning`, but subsequent calls with the same message
        are silently dropped.
        """
        _print_warning_once(self, msg)

    def info(  # type: ignore[override]
        self,
        msg: object,
        *args: Any,
        main_process_only: bool = True,
        local_main_process_only: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def debug(  # type: ignore[override]
        self,
        msg: object,
        *args: Any,
        main_process_only: bool = True,
        local_main_process_only: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def warning(  # type: ignore[override]
        self,
        msg: object,
        *args: Any,
        main_process_only: bool = False,
        local_main_process_only: bool = True,
        **kwargs: Any,
    ) -> None: ...

    def error(  # type: ignore[override]
        self,
        msg: object,
        *args: Any,
        main_process_only: bool = False,
        local_main_process_only: bool = True,
        **kwargs: Any,
    ) -> None: ...


def init_logger(name: str) -> _SGLDiffusionLogger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root sgl_diffusion logger has
    already been configured."""

    logger = logging.getLogger(name)

    # Patch instance methods
    setattr(logger, "info_once", MethodType(_print_info_once, logger))
    setattr(logger, "warning_once", MethodType(_print_warning_once, logger))

    def _create_patched_method(
        level: int,
        main_process_only_default: bool,
        local_main_process_only_default: bool,
    ):
        def _method(
            self: Logger,
            msg: object,
            *args: Any,
            main_process_only: bool = main_process_only_default,
            local_main_process_only: bool = local_main_process_only_default,
            **kwargs: Any,
        ) -> None:
            _log_process_aware(
                level,
                self,
                msg,
                *args,
                main_process_only=main_process_only,
                local_main_process_only=local_main_process_only,
                **kwargs,
            )

        return _method

    setattr(
        logger,
        "info",
        MethodType(_create_patched_method(logging.INFO, True, True), logger),
    )
    setattr(
        logger,
        "debug",
        MethodType(_create_patched_method(logging.DEBUG, True, True), logger),
    )
    setattr(
        logger,
        "warning",
        MethodType(_create_patched_method(logging.WARNING, False, True), logger),
    )
    setattr(
        logger,
        "error",
        MethodType(_create_patched_method(logging.ERROR, False, True), logger),
    )

    return cast(_SGLDiffusionLogger, logger)


logger = init_logger(__name__)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ["call", "return"]:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the sgl_diffusion root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == "call":
                    f.write(
                        f"{ts} Call to"
                        f" {func_name} in {filename}:{lineno}"
                        f" from {last_func_name} in {last_filename}:"
                        f"{last_lineno}\n"
                    )
                else:
                    f.write(
                        f"{ts} Return from"
                        f" {func_name} in {filename}:{lineno}"
                        f" to {last_func_name} in {last_filename}:"
                        f"{last_lineno}\n"
                    )
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str, root_dir: str | None = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    `log_file_path` is the path to the log file.
    `root_dir` is the root directory of the code to trace. If None, it is the
    sgl_diffusion root directory.

    Note that this call is thread-level, any threads calling this function
    will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "SGL_DIFFUSION_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only."
    )
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the sgl_diffusion root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))


def set_uvicorn_logging_configs():
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "[%(asctime)s] %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


def configure_logger(server_args, prefix: str = ""):
    log_format = f"[%(asctime)s{prefix}] %(message)s"
    datefmt = "%m-%d %H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=log_format,
        datefmt=datefmt,
        force=True,
    )

    set_uvicorn_logging_configs()


@contextmanager
def suppress_other_loggers(not_suppress_on_main_rank: bool = False):
    """
    A context manager to temporarily suppress specified loggers.

    Args:
        not_suppress_on_main_rank (bool): If True, loggers will not be
            suppressed on the main process (rank 0).
    """
    # This is a global setting that we want to apply to all ranks
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="The given NumPy array is not writable"
    )

    should_suppress = True
    if not_suppress_on_main_rank:
        rank, _ = _get_rank_info()
        if rank == 0:
            should_suppress = False

    loggers_to_suppress = ["urllib3"]
    original_levels = {}

    if should_suppress:
        for logger_name in loggers_to_suppress:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        if should_suppress:
            for logger_name, level in original_levels.items():
                logging.getLogger(logger_name).setLevel(level)
