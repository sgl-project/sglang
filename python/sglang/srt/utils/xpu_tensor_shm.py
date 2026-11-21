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
consumer maps it and copies host -> its own XPU device. The transfer is a copy,
so the rebuilt tensor is contiguous and aliases nothing.

Each scheduler reads payload[tp_rank], so a payload has one reader per pipeline
stage. Who unlinks its segments depends on whether the producer can tell that
every reader is done:

  * With one pipeline stage, ``Engine.update_weights_from_tensor`` returns only
    after the whole TP group has loaded -- the replying rank waits on a TP
    barrier -- so inside ``staged_xpu_segments(pp_size=1)`` the producer owns the
    segments: a tensor serialized once per TP rank is staged once, and leaving
    the block unlinks exactly what it staged, whether or not anyone read it.
  * Otherwise the readers unlink it: each segment counts, in a trailer, the
    readers that have finished, and the pp_size-th unlinks. Stage 0 replies for
    the engine while other stages may still be loading, and stages read in no
    fixed order, so neither the producer nor any one stage can tell it is last.
    This covers producers that hand a payload off and return
    (``weight_sync.utils.update_weights`` on its non-root ranks); a crash
    between staging and loading leaves such a payload's segment in /dev/shm.

TODO(siju-samuel): drop this module once ``reduce_tensor`` routes XPU tensors
(https://github.com/intel/torch-xpu-ops/issues/1678).
"""

import contextlib
import contextvars
import fcntl
import logging
import os
import re
import sys
from multiprocessing import resource_tracker, shared_memory
from typing import Callable, Dict, Iterator, Optional, Tuple
from unittest.mock import patch

import torch
from torch.multiprocessing import reductions

from sglang.srt.utils.device import is_xpu
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)

_is_xpu = is_xpu()

_SEGMENT_KIND = "xputensor"
# Trailer after the tensor bytes: how many readers have finished with the segment.
_READ_COUNT_BYTES = 8
# make_shm_name("xputensor") -> sgl_shm_xputensor_<pid>_<8 hex>
_SEGMENT_NAME = re.compile(rf"sgl_shm_{_SEGMENT_KIND}_\d+_[0-9a-f]{{8}}")

# Source key -> (segment name, source tensor) for the innermost open scope. Holding
# the tensor keeps its storage address from being reused while the scope is open,
# which is what makes the address a sound dedup key.
_scope: contextvars.ContextVar[Optional[Dict[tuple, Tuple[str, torch.Tensor]]]] = (
    contextvars.ContextVar("xpu_tensor_shm_scope", default=None)
)


def monkey_patch_xpu_tensor_reductions() -> None:
    """Make ForkingPickler stage XPU tensors through host SHM. Idempotent."""
    if not _is_xpu or hasattr(reductions, "_reduce_tensor_before_xpu_patch"):
        return

    reductions._reduce_tensor_before_xpu_patch = reductions.reduce_tensor
    reductions.reduce_tensor = _reduce_tensor_maybe_xpu
    reductions.init_reductions()
    logger.info(
        "XPU has no GPU tensor IPC: serialized XPU tensors are staged through "
        "host shared memory."
    )


@contextlib.contextmanager
def staged_xpu_segments(pp_size: int) -> Iterator[None]:
    """Own the shared memory behind every XPU tensor serialized in the block.

    Wrap both the serialization and the request that carries the payloads, for an
    engine with ``pp_size`` pipeline stages. With one stage the request returns
    after every reader is done, so the segments are unlinked on exit, no reader
    unlinks them, and a tensor serialized more than once in the block shares one
    segment. Other stages may still be reading when the request returns, so with
    more than one the block owns nothing and the last stage to read unlinks
    instead. Scopes nest; the innermost one decides. A no-op off XPU.
    """
    monkey_patch_xpu_tensor_reductions()
    staged: Optional[Dict[tuple, Tuple[str, torch.Tensor]]] = (
        {} if pp_size == 1 else None
    )
    token = _scope.set(staged)
    try:
        yield
    finally:
        _scope.reset(token)
        for name, _ in (staged or {}).values():
            _unlink_by_name(name)


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
    staged = _scope.get()
    if staged is None:
        shm_name = _stage_to_shm(tensor.detach().contiguous(), producer_owned=False)
    else:
        key = _source_key(tensor)
        if key not in staged:
            name = _stage_to_shm(tensor.detach().contiguous(), producer_owned=True)
            staged[key] = (name, tensor)
        shm_name = staged[key][0]
    return (
        _rebuild_xpu_tensor_from_shm,
        (
            type(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.requires_grad,
            shm_name,
            tensor.numel() * tensor.element_size(),
            staged is None,
        ),
    )


def _source_key(tensor: torch.Tensor) -> tuple:
    # _version moves on every in-place write, so a tensor modified between two
    # serializations in one scope is staged again rather than served stale.
    return (
        tensor.device,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor._version,
    )


def _stage_to_shm(source: torch.Tensor, producer_owned: bool) -> str:
    nbytes = source.numel() * source.element_size()
    # tmpfs zero-fills new pages, so the read count starts at 0.
    segment_bytes = nbytes + _READ_COUNT_BYTES
    shm = host = None
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
        # landed before the mapping is closed below and another process maps the
        # segment. Costs ~10-25us, because a copy into unpinned host memory has
        # already blocked in copy_() -- grouping these waits at the payload
        # boundary measured no faster and would hold every segment mapped.
        torch.xpu.current_stream(source.device).synchronize()
        host = None  # nothing may point into the mapping once it is closed
    except OSError as e:
        host = None
        _close_and_unlink(shm)
        raise RuntimeError(
            f"Failed to stage a {segment_bytes} B XPU tensor through shared "
            f"memory ({e}). XPU has no GPU tensor IPC, so /dev/shm holds one "
            f"host copy of every serialized tensor: enlarge /dev/shm or shrink "
            f"the weight-sync bucket size."
        ) from e
    except BaseException:
        host = None
        _close_and_unlink(shm)
        raise

    shm.close()
    if not producer_owned:
        # The consumer unlinks a hand-off payload, so stop tracking it here:
        # resource_tracker would unlink it at producer exit, before a consumer
        # that starts later has mapped it.
        resource_tracker.unregister(shm._name, "shared_memory")
    return shm.name


def _rebuild_xpu_tensor_from_shm(
    tensor_cls: type,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    requires_grad: bool,
    shm_name: str,
    nbytes: int,
    reader_unlinks: bool,
) -> torch.Tensor:
    """Map the staged segment onto this process' current XPU device.

    The producer's device index is deliberately not carried over: it is not
    meaningful under a different ZE_AFFINITY_MASK, and every consumer of a
    weight-sync payload wants the tensor on its own device anyway.
    """
    # The name arrives inside a pickle from an unauthenticated-by-default request
    # path, and this function reads what it names and may unlink it, so accept
    # only the shape _stage_to_shm produces.
    if not _SEGMENT_NAME.fullmatch(shm_name):
        raise RuntimeError(
            f"Refusing to map shared memory {shm_name!r}: not a staged XPU tensor segment"
        )

    try:
        shm = _attach_untracked(shm_name)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Shared-memory segment {shm_name} holding a staged XPU tensor is "
            f"gone: a payload its readers unlink may be read once per pipeline "
            f"stage, and only on the node that produced it."
        ) from e

    device = torch.device("xpu", torch.xpu.current_device())
    host = None
    try:
        host = torch.frombuffer(shm.buf, dtype=torch.uint8)[:nbytes]
        out = host.view(dtype).reshape(shape).to(device, copy=True)
        # Same as the producer's wait: the mapping goes away below, so the
        # host-to-device copy has to have drained first.
        torch.xpu.current_stream(device).synchronize()
    finally:
        host = None  # a view into the mapping would make close() raise
        last_reader = reader_unlinks and _count_read(shm, nbytes) == _readers()
        shm.close()
        if last_reader:
            _unlink_untracked(shm)

    if tensor_cls is torch.nn.parameter.Parameter:
        # Integer tensors must get requires_grad through the constructor, as in
        # torch.multiprocessing.reductions.rebuild_cuda_tensor.
        return torch.nn.parameter.Parameter(out, requires_grad=requires_grad)
    out.requires_grad = requires_grad
    return out


def _readers() -> int:
    """One scheduler per pipeline stage reads each payload."""
    from sglang.srt.runtime_context import get_parallel

    try:
        return get_parallel().pp_size
    except ValueError:
        # The parallel config is unpublished, so this is no engine's scheduler
        # and not part of a pipeline: the only reader.
        return 1


def _count_read(shm: shared_memory.SharedMemory, nbytes: int) -> int:
    """Record this reader as finished and return how many now are.

    Stages read in no fixed order -- one holding a request behind
    pause_generation relays it to the next stage before reading it -- so the
    segment counts its readers instead of trusting a stage to be last.
    """
    fcntl.flock(shm._fd, fcntl.LOCK_EX)
    try:
        with shm.buf[nbytes : nbytes + _READ_COUNT_BYTES] as count:
            reads = int.from_bytes(count, "little") + 1
            count[:] = reads.to_bytes(_READ_COUNT_BYTES, "little")
        return reads
    finally:
        fcntl.flock(shm._fd, fcntl.LOCK_UN)


# A spawned scheduler shares its Engine's resource_tracker, so a reader that
# registered and unregistered would race the producer and the other readers for
# one entry. Readers stay out of it; Python < 3.13 cannot attach untracked.
def _attach_untracked(name: str) -> shared_memory.SharedMemory:
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
        return shared_memory.SharedMemory(name=name)


def _unlink_untracked(shm: shared_memory.SharedMemory) -> None:
    try:
        if sys.version_info >= (3, 13):
            shm.unlink()  # attached with track=False, so this skips the tracker
        else:
            with patch(
                "multiprocessing.resource_tracker.unregister", lambda *a, **k: None
            ):
                shm.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "Failed to unlink the staged XPU tensor segment %s", shm.name, exc_info=True
        )


def _close_and_unlink(shm: Optional[shared_memory.SharedMemory]) -> None:
    if shm is None:
        return
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "Failed to unlink the staged XPU tensor segment %s", shm.name, exc_info=True
        )


def _unlink_by_name(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    except OSError:
        logger.warning(
            "Failed to reopen the staged XPU tensor segment %s", name, exc_info=True
        )
        return
    _close_and_unlink(shm)
