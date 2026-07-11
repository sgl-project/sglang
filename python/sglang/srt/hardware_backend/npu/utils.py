import functools
import logging
import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_npu_memory_capacity, is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
_is_npu = is_npu()
indexer_weight_stream = None
gva_is_inited = False
device_print_registered = False


class NPUACLFormat(IntEnum):
    ACL_FORMAT_UNDEFINED = -1
    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29


class FusedMoEMode(IntEnum):
    FUSED_DEEP_MOE = 1
    DISPATCH_FFN_COMBINE = 2


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

    # NPU memory settings
    decode = args.cuda_graph_config.decode
    npu_mem = get_npu_memory_capacity()
    if npu_mem <= 32 * 1024:
        # Ascend 910B4,910B4_1
        # (chunked_prefill_size 4k, max_bs 16 if tp < 4 else 64)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 4 * 1024
        if decode.max_bs is None:
            if args.tp_size < 4:
                decode.max_bs = 16
            else:
                decode.max_bs = 64
    elif npu_mem <= 64 * 1024:
        # Ascend 910B1,910B2,910B2C,910B3,910_9391,910_9392,910_9381,910_9382,910_9372,910_9362
        # (chunked_prefill_size 8k, max_bs 64 if tp < 4 else 256)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 8 * 1024
        if decode.max_bs is None:
            if args.tp_size < 4:
                decode.max_bs = 64
            else:
                decode.max_bs = 256

    # NPU does not support CustomAllReduce
    args.disable_custom_all_reduce = True

    # handles hierarchical cache configs
    if args.enable_hierarchical_cache:
        args.hicache_io_backend = "kernel_ascend"
        if args.use_mla_backend():
            args.hicache_mem_layout = "page_first_kv_split"
        else:
            args.hicache_mem_layout = "page_first_direct"


@_call_once
def init_npu_backend():
    """
    Initialize NPU backend. This function should be called only once.
    """

    assert _is_npu, "NPU backend initialization called on non-NPU device."

    try:
        import custom_ops  # noqa: F401
        import sgl_kernel_npu  # noqa: F401
    except ImportError as e:
        logger.warning("NPU custom kernel packages unavailable: %s", e)

    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    # Re-mock torch.cuda.is_available cuz transfer_to_npu mocks it True
    torch.cuda.is_available = lambda: False

    torch_npu.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)


def _is_nz_aligned(tensor: torch.Tensor) -> bool:
    """Check whether the last two dims satisfy FRACTAL_NZ alignment rules.

    Ascend FRACTAL_NZ requires:
      BF16 / FP16 : both dims divisible by 16
      INT8         : k % 16 == 0  and  n % 32 == 0
      INT4         : k % 16 == 0  and  n % 64 == 0
      FP4          : both dims divisible by 64
    """
    if tensor.dim() < 2:
        return False
    k, n = tensor.shape[-2], tensor.shape[-1]
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return k % 16 == 0 and n % 16 == 0
    if tensor.dtype == torch.int8:
        return k % 16 == 0 and n % 32 == 0
    if tensor.dtype in (torch.uint8, torch.int32):
        # INT4 is typically packed into uint8/int32; be conservative
        return k % 16 == 0 and n % 64 == 0
    return True


def npu_format_cast(
    tensor: torch.Tensor,
    acl_format: NPUACLFormat = NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
    *,
    customize_dtype=None,
    input_dtype=None,
) -> torch.Tensor:
    """
    Cast a tensor to a specific NPU ACL format.

    Args:
        tensor (torch.Tensor): The input tensor.
        acl_format (NPUACLFormat): The target NPU ACL format.
        customize_dtype / input_dtype: packed-FP4 unpack kwargs (e.g.
            ``customize_dtype=torch.float8_e4m3fn``,
            ``input_dtype=torch.float4_e2m1fn_x2``). When either is set the unpack
            kwargs are forwarded to the op and the ``_is_nz_aligned`` ND fallback
            is skipped: the FP4 matmul strictly requires FRACTAL_NZ, so a silent
            ND fallback would corrupt results.

    Returns:
        torch.Tensor: The tensor cast to the specified NPU ACL format.
    """

    if not _is_npu:
        return tensor

    if envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
        return tensor

    if tensor.device == torch.device("cpu"):
        logger.warning_once(
            "Warning: The conversion from 'ND' to 'NZ' does not work on the CPU. "
            "Please disable offloading, otherwise the performance will be "
            "significantly reduced. --dit-cpu-offload false"
        )
        return tensor

    # Skip format cast for meta tensors (used in offloader)
    if tensor.device.type == "meta":
        return tensor

    # Packed-FP4 → FRACTAL_NZ: forward the unpack kwargs to the op, and skip the
    # _is_nz_aligned ND fallback — the FP4 matmul strictly requires NZ, so a
    # silent ND fallback would corrupt results.
    if customize_dtype is not None or input_dtype is not None:
        return torch.ops.npu.npu_format_cast(
            tensor,
            int(acl_format),
            customize_dtype=customize_dtype,
            input_dtype=input_dtype,
        )

    if acl_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ and not _is_nz_aligned(tensor):
        k, n = tensor.shape[-2], tensor.shape[-1]
        logger.warning_once(
            "Skipping FRACTAL_NZ format cast: tensor shape (%d, %d) dtype %s "
            "is not aligned to NZ requirements. Falling back to 'ND' format, "
            "which may reduce NPU performance.",
            k,
            n,
            tensor.dtype,
        )
        return tensor

    return torch.ops.npu.npu_format_cast(tensor, acl_format.value)


def get_indexer_weight_stream():
    global indexer_weight_stream
    if indexer_weight_stream is None:
        indexer_weight_stream = torch.npu.Stream()
    return indexer_weight_stream


def init_zbal(world_size, gpu_id, world_rank, do_check=True):
    """
    init zbal, if is mix alloc mode, only register for sma & comm
    """
    zbal_mem_size = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get()
    if not zbal_mem_size > 0:
        return 1

    global gva_is_inited
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        # use lazy init for mix alloc
        return 1
    else:
        if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
            ret = zbal_init(
                world_size,
                gpu_id,
                world_rank,
                zbal_mem_size * (1024**2),
                ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
            )
        else:
            ret = zbal_init(world_size, gpu_id, world_rank, zbal_mem_size * (1024**2))

        gva_is_inited = True

        if do_check and not ret:
            logger.error("[ZBAL] zbal init failed!")
            sys.exit(-1)

        return ret


def lazy_init_zbal_gva_mem(
    device, gpu_id, world_rank, world_size, cpu_group=None, do_check=True
):
    """
    lazy init zbal gva mem, keep weights and kv remains alloc by dma vmm to avoid memory fragment
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info(
            "lazy init is supported only in mix alloc mode, this action will be passed"
        )
        return 1

    global gva_is_inited
    from sglang.srt.utils.common import get_available_gpu_memory

    # TODO need to use allgather if you want use total_memory stats from mem_get_info as unbalance os
    total_memory = 61.2  # 2.5GB for other (workspace & os) outside torch
    free_gpu_memory = get_available_gpu_memory(
        device,
        gpu_id,
        distributed=world_size > 1,
        cpu_group=cpu_group,
        empty_cache=True,
    )

    used_memory = total_memory - free_gpu_memory

    used_memory_in_mb = int(used_memory * 1024)
    gva_in_mb = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() - used_memory_in_mb
    gva_in_mb = gva_in_mb - gva_in_mb % 128  # align to 128MB
    print(f"[ZBAL] rank {world_rank} allocated {gva_in_mb} MB gva space.")

    assert not gva_is_inited, "zbal gva should be inited only once"
    # zbal_set_logger_level(0)
    if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
        res = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            gva_in_mb * (1024**2),
            ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
        )
    else:
        res = zbal_init(world_size, gpu_id, world_rank, gva_in_mb * (1024**2))

    gva_is_inited = True
    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        sys.exit(-1)
    return res


share_stream = None
routed_stream = None


def get_share_stream():
    global share_stream
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(share_stream, 8, 16)


def get_routed_stream():
    global routed_stream
    return routed_stream


def set_routed_stream(stream):
    global routed_stream
    routed_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(routed_stream, 16, 32)


def wait_share_stream():
    stream = get_share_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def wait_routed_stream():
    stream = get_routed_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states)
    return shared_output


def process_routed_expert(hidden_states, topk_output, forward_func):
    stream = get_routed_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_routed_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states, topk_output)
    return shared_output


def _mark_op_side_effectful(op: Any) -> None:
    torch.fx.node.has_side_effect(op)
    default_overload = getattr(op, "default", None)
    if default_overload is not None:
        torch.fx.node.has_side_effect(default_overload)


def _ensure_device_print_registered() -> None:
    global device_print_registered
    if device_print_registered:
        return
    try:
        # Mark device_print ops side-effectful so FX/Inductor does not DCE or reorder these debug callbacks.
        import sgl_kernel_npu  # noqa: F401

        _mark_op_side_effectful(torch.ops.npu.device_print)
        _mark_op_side_effectful(torch.ops.npu.device_print_tensor)
        device_print_registered = True
    except AttributeError as exc:
        raise RuntimeError(
            "device_print ops are available in sgl_kernel_npu. "
            "Please install the latest sgl_kernel_npu from source."
        ) from exc


def device_print(
    value: (
        torch.Tensor
        | int
        | float
        | bool
        | str
        | torch.dtype
        | torch.device
        | torch.Size
    ),
) -> None:
    """Print one value from a device callback.

    This helper is intended for debugging. To stay replay-safe under
    ``torch.npu.graph`` capture/replay, the underlying callback payloads are
    retained instead of being reclaimed after the first host callback runs.
    Avoid using it in hot paths or long-running high-frequency loops, otherwise
    there may be memory issues due to too many retained payloads.

    Supported usage:

        >>> from sglang.srt.hardware_backend.npu.utils import device_print
        >>> device_print(x)
        >>> device_print("already formatted text")
        >>> device_print(7)

    Unsupported usage:

        >>> device_print("x =", x)
        >>> device_print("This is ", x, "and this is ", y)

    If you need device-time tensor values, pass the tensor itself. If you need
    text, pass one final string that is already formatted.

    Tensor values are copied to host on the current stream before the callback
    prints them, so printing remains ordered with respect to the surrounding
    device work.

    DO NOT FORMAT A DEVICE TENSOR INTO A STRING YOURSELF AND THEN PRINT, for example:

        >>> device_print(f"x = {x}")
        >>> device_print("x = " + str(x))
    """
    _ensure_device_print_registered()

    if isinstance(value, torch.Tensor):
        torch.ops.npu.device_print_tensor(value)
    elif isinstance(
        value, (str, int, float, bool, torch.dtype, torch.device, torch.Size)
    ):
        torch.ops.npu.device_print(str(value))
    else:
        raise TypeError(
            f"Unsupported device_print value type: {type(value)!r}. "
            "Use exactly one argument: device_print(tensor), device_print('formatted text')."
        )
