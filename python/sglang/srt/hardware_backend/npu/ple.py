"""NPU PLE UVA lookup for pinned and file-backed BF16 host tables.

UVA is required: unsupported storage, registration, or alias APIs raise during
embedding construction. There is no CPU staging or segmented graph fallback.
File mappings use MAPPED only; this module never silently locks the full table.
"""

import ctypes
import logging
import weakref
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _acl_runtime():
    library = ctypes.CDLL("libascendcl.so")
    signatures = {
        "aclrtHostGetDevicePointer": [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
        ],
        "aclrtHostUnregister": [ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    if hasattr(library, "aclrtHostRegisterV2"):
        function = library.aclrtHostRegisterV2
        function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    else:
        function = library.aclrtHostRegister
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
    function.restype = ctypes.c_int
    return library


def _mapped_npu_tensor(weight, pointer, device):
    """Wrap mapped storage through torch_npu's AOTI C ABI without copying it."""
    import torch_npu

    library = ctypes.CDLL(
        str(Path(torch_npu.__file__).resolve().parent / "lib/libtorch_npu.so")
    )
    create = library.aoti_torch_create_tensor_from_blob_npu
    dtype = library.aoti_torch_dtype_bfloat16
    device_type = library.aoti_torch_device_type_npu
    steal = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs
    for query in (dtype, device_type):
        query.argtypes, query.restype = [], ctypes.c_int32
    create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create.restype = ctypes.c_int32
    sizes = (ctypes.c_int64 * weight.ndim)(*weight.shape)
    strides = (ctypes.c_int64 * weight.ndim)(*weight.stride())
    handle = ctypes.c_void_p()
    result = create(
        pointer,
        weight.ndim,
        sizes,
        strides,
        0,
        dtype(),
        device_type(),
        device,
        ctypes.byref(handle),
    )
    if result or not handle.value:
        raise RuntimeError(f"NPU mapped tensor creation returned {result}")
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule_new.restype = ctypes.py_object
    capsule = capsule_new(handle, None, None)
    try:
        tensor = steal([capsule])[0]
    except Exception:
        delete = library.aoti_torch_delete_tensor_object
        delete.argtypes, delete.restype = [ctypes.c_void_p], ctypes.c_int32
        delete(handle)
        raise
    if tensor.data_ptr() != pointer or tensor.dtype != weight.dtype:
        raise RuntimeError("NPU mapped tensor did not preserve storage metadata")
    return tensor


def _release_mapping(storage_owner, device, library, base, owned):
    # The finalizer holds CPU storage until all uses of the non-owning NPU
    # alias finish. Only registrations made here may be unregistered here.
    try:
        with torch.npu.device(device):
            torch.npu.synchronize()
            if owned:
                result = library.aclrtHostUnregister(base)
                if result:
                    logger.warning("PLE UVA unregister returned %s", result)
    except Exception as error:
        logger.warning("PLE UVA cleanup failed: %s", error)


class NpuUvaTable:
    """Keep the alias and its registered CPU storage alive together."""

    def __init__(self, weight, tensor, device, library, base, owned):
        self.tensor = tensor
        # Parameter attributes include a bound weight_loader pointing back to
        # the embedding. Keep only detached storage here, or the finalizer's
        # registry would indirectly keep its own target alive forever.
        self._cleanup = weakref.finalize(
            tensor, _release_mapping, weight.detach(), device, library, base, owned
        )

    def close(self):
        """Release after callers have retired every graph using this table."""
        self._cleanup()
        self.tensor = None


def create_ple_uva_table(weight: torch.Tensor, *, backend: str) -> NpuUvaTable:
    """Register pinned/file host storage or fail before model graph capture."""
    if backend not in ("pinned", "file"):
        raise ValueError(f"Invalid NPU PLE backend: {backend}")
    if weight.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"NPU PLE UVA requires bfloat16 storage, got {weight.dtype}"
        )
    if weight.device.type != "cpu" or not weight.is_contiguous() or not weight.numel():
        raise ValueError("NPU PLE UVA requires a nonempty contiguous CPU table")
    if backend == "pinned" and not weight.is_pinned():
        raise ValueError("NPU PLE pinned backend requires pinned CPU storage")
    library, owned, base = None, False, weight.untyped_storage().data_ptr()
    try:
        device = torch.npu.current_device()
        library = _acl_runtime()
        mapped = ctypes.c_void_p()
        result = library.aclrtHostGetDevicePointer(base, ctypes.byref(mapped), 0)
        if result or not mapped.value:
            nbytes = weight.untyped_storage().nbytes()
            if hasattr(library, "aclrtHostRegisterV2"):
                # MAPPED only: do not convert the file backend into a locked
                # whole-table allocation. Unsupported mappings must fail.
                result = library.aclrtHostRegisterV2(base, nbytes, 0x2)
            else:
                result = library.aclrtHostRegister(
                    base, nbytes, 0, ctypes.byref(mapped)
                )
            if result:
                raise RuntimeError(f"host registration returned {result}")
            owned = True
            result = library.aclrtHostGetDevicePointer(base, ctypes.byref(mapped), 0)
            if result or not mapped.value:
                raise RuntimeError(f"mapped device address query returned {result}")
        pointer = mapped.value + weight.storage_offset() * weight.element_size()
        tensor = _mapped_npu_tensor(weight, pointer, device)
        return NpuUvaTable(weight, tensor, device, library, base, owned)
    except (OSError, AttributeError, RuntimeError) as error:
        if owned:
            result = library.aclrtHostUnregister(base)
            if result:
                raise RuntimeError(
                    f"PLE UVA rollback unregister returned {result}"
                ) from error
        raise RuntimeError(
            f"NPU PLE backend={backend} requires UVA; mapping/alias setup failed: {error}. "
            "CPU staging is not supported; check the installed CANN, driver, "
            "OS kernel, torch_npu APIs, and host-memory mapping support."
        ) from error


def gather_ple_uva_rows(table, flat_ids, output, *, vocab_start, vocab_end):
    """Device-only row selection, TP masking, and reuse of the caller's buffer."""
    if table.tensor is None:
        raise RuntimeError("PLE UVA table is closed")
    if table.tensor.device != flat_ids.device:
        raise ValueError("PLE UVA table and IDs must use the same NPU device")
    in_range = (flat_ids >= vocab_start) & (flat_ids < vocab_end)
    local_ids = torch.where(in_range, flat_ids - vocab_start, 0).long()
    rows = output.view(flat_ids.numel(), -1)
    torch.index_select(table.tensor, 0, local_ids, out=rows)
    rows.masked_fill_(~in_range[:, None], 0)
    return output
