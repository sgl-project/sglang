"""CANN host mappings for NPU Triton PLE lookups, with per-table ownership."""

import logging
import mmap
import weakref

import torch

logger = logging.getLogger(__name__)


def _device_pointer(runtime, base):
    getter = getattr(runtime, "host_get_device_pointer", None)
    if getter is None:
        return 0
    pointer, result = getter(base, 0)
    return int(pointer) if result == 0 and pointer else 0


def _release_mapping(storage_owner, device, runtime, base, owned):
    # Raw addresses captured in graphs do not retain CPU storage. Callers must
    # retire those graphs before closing the table or unloading the model.
    try:
        with torch.npu.device(device):
            torch.npu.synchronize()
            if owned:
                result = runtime.host_unregister(base)
                if result:
                    logger.warning("PLE UVA unregister returned %s", result)
    except Exception as error:
        logger.warning("PLE UVA cleanup failed: %s", error)


class NpuUvaTable:
    """Keep the mapped host storage alive while Triton uses its device address."""

    def __init__(self, weight, pointer, device, runtime, base, owned):
        self.device_ptr = pointer
        self.device = device
        # The Parameter has a bound weight_loader; retain detached storage to
        # avoid keeping the embedding and this finalizer's target alive forever.
        self._cleanup = weakref.finalize(
            self, _release_mapping, weight.detach(), device, runtime, base, owned
        )

    def close(self):
        """Release after callers have retired every graph using this table."""
        self._cleanup()
        self.device_ptr = 0

    def get_device_pointer(self, device: torch.device) -> int:
        if not self.device_ptr:
            raise RuntimeError("PLE UVA table is closed")
        if device.type != "npu" or device.index != self.device:
            raise ValueError("PLE UVA table and IDs must use the same NPU device")
        return self.device_ptr


def create_ple_uva_table(weight: torch.Tensor, *, backend: str) -> NpuUvaTable:
    """Map existing pinned/file storage through CANN's public Python API."""
    if backend not in ("pinned", "file"):
        raise ValueError(f"Invalid NPU PLE backend: {backend}")
    if weight.device.type != "cpu" or not weight.is_contiguous() or not weight.numel():
        raise ValueError("NPU PLE UVA requires a nonempty contiguous CPU table")
    if backend == "pinned" and not weight.is_pinned():
        raise ValueError("NPU PLE pinned backend requires pinned CPU storage")

    owned = False
    try:
        import acl.rt as runtime

        device = torch.npu.current_device()
        storage = weight.untyped_storage()
        host_ptr = storage.data_ptr()
        base = host_ptr - host_ptr % mmap.PAGESIZE
        offset = host_ptr - base
        nbytes = (
            (offset + storage.nbytes() + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        ) * mmap.PAGESIZE
        pointer = _device_pointer(runtime, base)
        if not pointer:
            # aclrtHostRegisterType MAPPED=0; this registers the existing range
            # without allocating or copying a replacement for the file mapping.
            pointer, result = runtime.host_register(base, nbytes, 0)
            if result or not pointer:
                # A different component may have registered it after the query.
                pointer = _device_pointer(runtime, base)
                if not pointer:
                    raise RuntimeError(f"host registration returned {result}")
            else:
                owned = True
        pointer = (
            int(pointer) + offset + weight.storage_offset() * weight.element_size()
        )
        return NpuUvaTable(weight, pointer, device, runtime, base, owned)
    except (ImportError, OSError, AttributeError, RuntimeError) as error:
        if owned:
            try:
                result = runtime.host_unregister(base)
                if result:
                    raise RuntimeError(f"host unregister returned {result}")
            except (
                ImportError,
                OSError,
                AttributeError,
                RuntimeError,
            ) as cleanup_error:
                raise RuntimeError(
                    f"PLE UVA rollback failed: {cleanup_error}"
                ) from error
        raise RuntimeError(
            f"NPU PLE backend={backend} requires UVA; mapping failed: {error}. "
            "Check the CANN acl.rt binding, driver, OS kernel, and host-memory "
            "mapping support. CPU staging is not supported."
        ) from error
