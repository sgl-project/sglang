import functools
import json
import os
import sys
import threading

from deep_gemm.deep_gemm_tuner import get_deep_gemm_luts

from sglang.srt.distributed.parallel_state import get_moe_ep_group, get_tp_group
from sglang.srt.utils import logger

MAX_DECODE_BS = 1025


def check_main_thread():
    current_thread = threading.current_thread()
    if current_thread.name == "MainThread":
        return True
    else:
        return False


non_tuned_mnk_set = set()


@functools.lru_cache(maxsize=None)
def get_default_user_dir():
    if "DG_CACHE_DIR" in os.environ:
        path = os.getenv("DG_CACHE_DIR")
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.expanduser("~") + "/.ppu_sglang"


@functools.lru_cache(maxsize=None)
def get_default_deep_gemm_dump_file():
    path = f"{get_default_user_dir()}/deepgemm"
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "mnk.json")


def handle_deepgemm_signal(signum, frame):
    dump_file_path = get_default_deep_gemm_dump_file()
    json_ready_data = [list(t) for t in list(non_tuned_mnk_set)]
    rank = -1
    world_size = 1
    try:
        rank = get_moe_ep_group().rank
        world_size = get_moe_ep_group().world_size
    except:
        try:
            rank = get_tp_group().rank
            world_size = get_tp_group().world_size
        except:
            pass
    if rank == 0:
        with open(dump_file_path, "w") as f:
            json.dump(json_ready_data, f, indent=4)
            logger.info(
                f"DeepGemm Tuner: Saved non-tuned configs{json_ready_data} in {dump_file_path}, please run `python -m sglang.tune_deepgemm_int8 --tp {world_size}` to get a better performance"
            )
    sys.exit(0)


def append_non_tuned_case(M, N, K, NUM_GROUP, nopad, dtype):
    if M < MAX_DECODE_BS or (nopad and M < MAX_DECODE_BS * 8):
        # note that we save configs with N before N
        non_tuned_mnk_set.add((M, K, N, NUM_GROUP, nopad, dtype))


@functools.lru_cache(maxsize=None)
def get_deep_gemm_config(M, N, K, num_groups, nopad=False, dtype="int8"):
    # deep gemm tuner does not support fp4 yet
    if dtype == "fp4":
        return None

    best_config = get_deep_gemm_luts(M, N, K, num_groups, nopad=nopad, dtype=dtype)
    if best_config is None:
        append_non_tuned_case(M, N, K, num_groups, nopad, dtype)

    return best_config
