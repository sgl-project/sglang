import functools
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _call_once(fn: Callable):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(fn, "_has_been_called", False):
            logger.debug("Function {} has already been called.", fn.__name__)
            return

        fn._has_been_called = True
        return fn(*args, **kwargs)

    return wrapper


def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for NPU backend.
    """

    # NPU only works with "ascend" attention backend for now
    args.attention_backend = "ascend"
    args.prefill_attention_backend = "ascend"
    args.decode_attention_backend = "ascend"
    if args.page_size is None:
        args.page_size = 128

    # NPU does not support CustomAllReduce
    args.disable_custom_all_reduce = True

    # handles hierarchical cache configs
    if args.enable_hierarchical_cache:
        args.hicache_io_backend = "kernel_ascend"
        if args.use_mla_backend():
            args.hicache_mem_layout = "page_first_kv_split"
        else:
            args.hicache_mem_layout = "page_first_direct"
