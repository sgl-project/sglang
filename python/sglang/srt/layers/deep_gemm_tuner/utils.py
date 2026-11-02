import functools
import json
import logging
import os
import signal
import threading
from types import SimpleNamespace

logger = logging.getLogger(__name__)

MAX_DECODE_BS = 1025


def check_main_thread():
    return threading.current_thread() is threading.main_thread()


def _state():
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(
        "sail_deepgemm_tuner",
        lambda: SimpleNamespace(configs={}, non_tuned_mnk_set=set()),
    )


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if "DG_CACHE_DIR" in os.environ:
        path = os.getenv("DG_CACHE_DIR")
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.join(os.path.expanduser("~"), ".sail_sglang")


@functools.lru_cache(maxsize=None)
def get_default_deep_gemm_dump_file():
    path = f"{get_default_user_dir()}/deepgemm"
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "mnk.json")


def handle_deepgemm_signal(signum, frame):
    from sglang.srt.runtime_context import get_parallel

    cases = _state().non_tuned_mnk_set
    if not cases:
        return
    rank, world_size = 0, 1
    for name in ("moe_ep_group", "tp_group"):
        try:
            group = getattr(get_parallel(), name)
            rank, world_size = group.rank, group.world_size
            break
        except (AttributeError, AssertionError, ValueError):
            continue
    if rank == 0:
        dump_file_path = get_default_deep_gemm_dump_file()
        with open(dump_file_path, "w", encoding="utf-8") as f:
            json.dump([list(t) for t in sorted(cases)], f, indent=4)
        logger.info(
            "DeepGEMM tuning: untuned shapes saved to %s; run "
            "`python -m sglang.tune_deepgemm_int8 --tp %s` for offline tuning.",
            dump_file_path,
            world_size,
        )


def install_signal_handler():
    # Installed only after the worker has bound its GPU, and chained to the
    # previous handler so normal exit and resource cleanup still run.
    if not check_main_thread():
        return
    previous = signal.getsignal(signal.SIGINT)
    if getattr(previous, "_sglang_deepgemm_tuner", False):
        return

    def handler(signum, frame):
        try:
            handle_deepgemm_signal(signum, frame)
        except Exception:
            logger.exception("Failed to save DeepGEMM tuning shapes")
        finally:
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

    handler._sglang_deepgemm_tuner = True
    signal.signal(signal.SIGINT, handler)


def append_non_tuned_case(M, N, K, NUM_GROUP, nopad, dtype):
    if M < MAX_DECODE_BS or (nopad and M < MAX_DECODE_BS * 8):
        # The vendor offline tuner uses the (M, K, N, groups, nopad, dtype) order.
        _state().non_tuned_mnk_set.add((M, K, N, NUM_GROUP, nopad, dtype))


def get_deep_gemm_config(M, N, K, num_groups, nopad=False, dtype="int8"):
    # deep gemm tuner does not support fp4 / int4 yet
    if dtype == "fp4" or dtype == "int4":
        return None

    key = (M, N, K, num_groups, nopad, dtype)
    configs = _state().configs
    if key not in configs:
        from deep_gemm.deep_gemm_tuner import get_deep_gemm_luts

        configs[key] = get_deep_gemm_luts(M, N, K, num_groups, nopad=nopad, dtype=dtype)
        if configs[key] is None:
            append_non_tuned_case(M, N, K, num_groups, nopad, dtype)
    return configs[key]
