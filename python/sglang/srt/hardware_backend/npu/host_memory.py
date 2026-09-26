"""Device-visible mappings of pinned host memory on Ascend.

An Ascend kernel cannot dereference a raw host pointer: the MTE unit reads it as
a device DDR address and faults with "The DDR address of the MTE instruction is
out of range". CANN offers the CUDA-UVA equivalent -- ``aclrtHostRegister``
page-locks a host range and returns a *device* address that AI Core / vector
kernels can read directly.

Measured on CANN 9.0 / Ascend 910 with the Qwen4-Exp PLE gather:

* reads through the mapped address match the host table exactly;
* a 1 GiB table costs no device memory (a true mapping, not a copy), and 32 GiB
  still registers in ~0.24 s;
* host writes made *after* registration are visible to the device;
* random-row gather from host memory runs at ~19 GiB/s (~36 GiB/s when the same
  table is device-resident).

Note: ``aclrtHostMemMapCapabilities`` reports ``ACL_ERROR_RT_FEATURE_NOT_SUPPORT``
(207000) for every HAC unit on this device even though registration works, so it
must not be used as a gate -- trust the return code of the register call.
"""

import logging
import mmap
from typing import Optional

from sglang.srt.platforms.interface import HostMemoryMapping

logger = logging.getLogger(__name__)

# aclrtHostRegisterType: map the host range into the device address space.
ACL_HOST_REGISTER_MAPPED = 0


def _load_acl_rt():
    import acl.rt as acl_rt

    return acl_rt


def _aligned_range(host_ptr: int, nbytes: int) -> tuple[int, int, int]:
    """Return page-aligned ``(base, size, offset)`` for a host range."""
    page_size = mmap.PAGESIZE
    base = host_ptr - host_ptr % page_size
    offset = host_ptr - base
    size = ((offset + nbytes + page_size - 1) // page_size) * page_size
    return base, size, offset


def _get_device_pointer(acl_rt, host_ptr: int) -> Optional[int]:
    """Return an existing device mapping without taking ownership of it."""
    getter = getattr(acl_rt, "host_get_device_pointer", None)
    if getter is None:
        return None
    try:
        # NOTE: the binding returns (dev_ptr, ret), not (ret, dev_ptr).
        dev_ptr, ret = getter(host_ptr, 0)
    except Exception:  # an unregistered range is an expected miss
        return None
    if ret == 0 and dev_ptr:
        return int(dev_ptr)
    return None


def register_host_memory(host_ptr: int, nbytes: int) -> Optional[HostMemoryMapping]:
    """Map ``[host_ptr, host_ptr + nbytes)`` into the device address space.

    Existing mappings (including mappings owned by torch_npu's pinned-memory
    allocator) are borrowed and must not be unregistered here. New mappings are
    page-aligned to satisfy ``aclrtHostRegister`` and marked as owned.
    """
    if host_ptr <= 0 or nbytes <= 0:
        return None
    registered_host_ptr, registered_nbytes, offset = _aligned_range(host_ptr, nbytes)
    try:
        acl_rt = _load_acl_rt()

        existing_ptr = _get_device_pointer(acl_rt, registered_host_ptr)
        if existing_ptr is not None:
            logger.info(
                "Using an existing host-memory device mapping: host=%#x -> device=%#x",
                registered_host_ptr,
                existing_ptr,
            )
            return HostMemoryMapping(
                device_ptr=existing_ptr + offset,
                registered_host_ptr=registered_host_ptr,
                owned=False,
            )

        # NOTE: the binding returns (dev_ptr, ret), not (ret, dev_ptr).
        dev_ptr, ret = acl_rt.host_register(
            registered_host_ptr, registered_nbytes, ACL_HOST_REGISTER_MAPPED
        )
    except Exception as exc:  # pragma: no cover - depends on the CANN install
        logger.warning("aclrtHostRegister unavailable: %s", exc)
        return None
    if ret != 0 or not dev_ptr:
        # Another component may have registered the range between the lookup and
        # the register call. Query once more and borrow that mapping if present.
        existing_ptr = _get_device_pointer(acl_rt, registered_host_ptr)
        if existing_ptr is not None:
            return HostMemoryMapping(
                device_ptr=existing_ptr + offset,
                registered_host_ptr=registered_host_ptr,
                owned=False,
            )
        logger.warning(
            "aclrtHostRegister failed for %#x (%d bytes): ret=%s",
            registered_host_ptr,
            registered_nbytes,
            ret,
        )
        return None
    logger.info(
        "Mapped %d bytes of host memory for device access: host=%#x -> device=%#x",
        registered_nbytes,
        registered_host_ptr,
        dev_ptr,
    )
    return HostMemoryMapping(
        device_ptr=int(dev_ptr) + offset,
        registered_host_ptr=registered_host_ptr,
        owned=True,
    )


def unregister_host_memory(mapping: HostMemoryMapping) -> None:
    """Release an owned mapping; borrowed mappings are left untouched."""
    if not mapping.owned:
        return
    try:
        acl_rt = _load_acl_rt()

        ret = acl_rt.host_unregister(mapping.registered_host_ptr)
    except Exception as exc:  # pragma: no cover - depends on the CANN install
        logger.warning("aclrtHostUnregister unavailable: %s", exc)
        return
    if ret != 0:
        logger.warning(
            "aclrtHostUnregister failed for %#x: ret=%s",
            mapping.registered_host_ptr,
            ret,
        )
