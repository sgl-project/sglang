"""Host shared-memory staging for XPU tensors crossing a process boundary.

PyTorch has no XPU tensor IPC: ``UntypedStorage`` exposes ``_share_cuda_`` but no
``_share_xpu_``, and ``torch.multiprocessing.reductions.reduce_tensor`` has no XPU
branch, so pickling an XPU tensor with ``ForkingPickler`` falls through to the CPU
storage path and dies with ``RuntimeError: _share_fd_: only available on CPU``.
Under Ray the fd fallback instead fails as
``AuthenticationError: digest received was wrong``, because fd passing needs a
shared authkey that a trainer actor and an mp.spawn-ed scheduler never share.

The fallback stages the bytes through a POSIX shared-memory segment: the producer
copies device -> host SHM, the payload carries only the segment name, and the
consumer maps it, copies host -> its own XPU device, and unlinks. Contract, which
differs from CUDA IPC in ways the weight-load consumers do not rely on:

  * the transfer is a copy, so the rebuilt tensor is contiguous and aliases nothing;
  * each payload stages its own segment, so a producer that serializes one payload
    per TP rank holds ``tp_size`` host copies of a bucket at once;
  * a payload may be deserialized exactly once - the consumer owns the unlink.

A producer that waits for the consumer (the synchronous engine APIs) then calls
``discard_staged_segments`` to reclaim payloads no consumer took, e.g. after a
failed update. A producer that hands a payload to another process and returns
cannot know when the transfer ended, so there the consumer's unlink is the only
reclaim, and an operator clears a leak with ``rm /dev/shm/sgl_shm_xputensor_*``.

TODO(siju-samuel): drop this module once ``reduce_tensor`` routes XPU tensors
(https://github.com/intel/torch-xpu-ops/issues/1678).
"""

import logging
import os
import re
import sys
from multiprocessing import resource_tracker, shared_memory
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.multiprocessing import reductions

from sglang.srt.utils.common import is_xpu
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)

_is_xpu = is_xpu()

_SEGMENT_KIND = "xputensor"
# make_shm_name("xputensor") -> sgl_shm_xputensor_<pid>_<8 hex>
_SEGMENT_NAME = re.compile(rf"sgl_shm_{_SEGMENT_KIND}_\d+_[0-9a-f]{{8}}")

# Names of segments staged here, oldest first, for discard_staged_segments. A
# producer that never releases would otherwise grow this without bound, so the
# oldest names are forgotten past the cap; by then a consumer has taken them.
_MAX_STAGED_NAMES = 1024
_staged_names: Dict[str, None] = {}


def monkey_patch_xpu_tensor_reductions() -> None:
    """Make ForkingPickler stage XPU tensors through host SHM. Idempotent."""
    if not _is_xpu or hasattr(reductions, "_reduce_tensor_before_xpu_patch"):
        return

    reductions._reduce_tensor_before_xpu_patch = reductions.reduce_tensor
    reductions.reduce_tensor = _reduce_tensor_maybe_xpu
    reductions.init_reductions()
    logger.info(
        "XPU has no GPU tensor IPC: serialized XPU tensors are staged through "
        "host shared memory, one host copy per payload."
    )


def discard_staged_segments() -> None:
    """Unlink segments no consumer took. Safe only once the consumers are done."""
    while _staged_names:
        _unlink_staged(_staged_names.popitem()[0])


def _reduce_tensor_maybe_xpu(tensor: torch.Tensor) -> Tuple[Callable, Tuple]:
    if (
        tensor.device.type == "xpu"
        and tensor.layout == torch.strided
        and not tensor.is_nested
        and tensor.is_leaf
    ):
        return _reduce_xpu_tensor(tensor)
    # Nested and sparse tensors reduce component-wise through this same module
    # global, and a non-leaf tensor gets torch's own autograd refusal.
    return reductions._reduce_tensor_before_xpu_patch(tensor)


def _reduce_xpu_tensor(tensor: torch.Tensor) -> Tuple[Callable, Tuple]:
    torch.utils.hooks.warn_if_has_hooks(tensor)
    source = tensor.detach().contiguous()
    shm_name, nbytes = _stage_to_shm(source)
    return (
        _rebuild_xpu_tensor_from_shm,
        (
            type(tensor),
            tuple(source.shape),
            source.dtype,
            tensor.requires_grad,
            shm_name,
            nbytes,
        ),
    )


def _stage_to_shm(source: torch.Tensor) -> Tuple[str, int]:
    nbytes = source.numel() * source.element_size()
    # SharedMemory rejects size=0, and an empty tensor still needs a name.
    segment_bytes = max(nbytes, 1)
    shm = None
    try:
        shm = shared_memory.SharedMemory(
            create=True, size=segment_bytes, name=make_shm_name(_SEGMENT_KIND)
        )
        if sys.platform == "linux":
            # tmpfs pages are allocated at write time, so a full /dev/shm would
            # kill the process with SIGBUS; reserving turns that into ENOSPC.
            os.posix_fallocate(shm._fd, 0, segment_bytes)
        host = torch.frombuffer(shm.buf, dtype=torch.uint8)[:nbytes]
        host.copy_(source.reshape(-1).view(torch.uint8))
        # The device-to-host copy is stream-ordered; the bytes have to have
        # landed before another process maps the segment.
        torch.xpu.current_stream(source.device).synchronize()
        del host  # nothing may point into the mapping once it is closed
    except OSError as e:
        _close_and_unlink(shm)
        raise RuntimeError(
            f"Failed to stage a {segment_bytes} B XPU tensor through shared "
            f"memory ({e}). XPU has no GPU tensor IPC, so /dev/shm holds one "
            f"host copy of every serialized tensor: enlarge /dev/shm or shrink "
            f"the weight-sync bucket size."
        ) from e
    except BaseException:
        _close_and_unlink(shm)
        raise

    name = shm.name
    shm.close()
    # The consumer unlinks, so stop tracking here: resource_tracker would unlink
    # the segment at producer exit, before a consumer that starts later maps it.
    resource_tracker.unregister(shm._name, "shared_memory")
    if len(_staged_names) >= _MAX_STAGED_NAMES:
        del _staged_names[next(iter(_staged_names))]
    _staged_names[name] = None
    return name, nbytes


def _rebuild_xpu_tensor_from_shm(
    tensor_cls: type,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    requires_grad: bool,
    shm_name: str,
    nbytes: int,
) -> torch.Tensor:
    """Map the staged segment onto this process' current XPU device, then unlink.

    The producer's device index is deliberately not carried over: it is not
    meaningful under a different ZE_AFFINITY_MASK, and every consumer of a
    weight-sync payload wants the tensor on its own device anyway.
    """
    # The name arrives inside a pickle from an unauthenticated-by-default request
    # path, and this function both reads and unlinks what it names, so accept only
    # the shape _stage_to_shm produces.
    if not _SEGMENT_NAME.fullmatch(shm_name):
        raise RuntimeError(
            f"Refusing to map shared memory {shm_name!r}: not a staged XPU tensor segment"
        )

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Shared-memory segment {shm_name} holding a staged XPU tensor is "
            f"gone. Deserializing unlinks it, so each payload may be consumed "
            f"only once, and only on the node that produced it."
        ) from e

    device = torch.device("xpu", torch.xpu.current_device())
    try:
        host = torch.frombuffer(shm.buf, dtype=torch.uint8)[:nbytes]
        out = host.view(dtype).reshape(shape).to(device, copy=True)
        torch.xpu.current_stream(device).synchronize()
        del host
    finally:
        _close_and_unlink(shm)

    if tensor_cls is torch.nn.parameter.Parameter:
        # Integer tensors must get requires_grad through the constructor, as in
        # torch.multiprocessing.reductions.rebuild_cuda_tensor.
        return torch.nn.parameter.Parameter(out, requires_grad=requires_grad)
    out.requires_grad = requires_grad
    return out


def _close_and_unlink(shm: Optional[shared_memory.SharedMemory]) -> None:
    if shm is None:
        return
    _staged_names.pop(shm.name, None)
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "Failed to unlink the staged XPU tensor segment %s", shm.name, exc_info=True
        )


def _unlink_staged(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return  # a consumer took it
    except OSError:
        logger.warning(
            "Failed to reopen the staged XPU tensor segment %s", name, exc_info=True
        )
        return
    _close_and_unlink(shm)
