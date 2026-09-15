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
from typing import Optional

logger = logging.getLogger(__name__)

# aclrtHostRegisterType: map the host range into the device address space.
ACL_HOST_REGISTER_MAPPED = 0


def register_host_memory(host_ptr: int, nbytes: int) -> Optional[int]:
    """Page-lock ``[host_ptr, host_ptr + nbytes)`` and return a device address.

    Returns ``None`` when the range could not be mapped.
    """
    if nbytes <= 0:
        return None
    try:
        import acl.rt as acl_rt

        # NOTE: the binding returns (dev_ptr, ret), not (ret, dev_ptr).
        dev_ptr, ret = acl_rt.host_register(host_ptr, nbytes, ACL_HOST_REGISTER_MAPPED)
    except Exception as exc:  # pragma: no cover - depends on the CANN install
        logger.warning("aclrtHostRegister unavailable: %s", exc)
        return None
    if ret != 0 or not dev_ptr:
        logger.warning(
            "aclrtHostRegister failed for %#x (%d bytes): ret=%s", host_ptr, nbytes, ret
        )
        return None
    logger.info(
        "Mapped %d bytes of host memory for device access: host=%#x -> device=%#x",
        nbytes,
        host_ptr,
        dev_ptr,
    )
    return dev_ptr


def unregister_host_memory(host_ptr: int) -> None:
    """Undo :func:`register_host_memory`; must run before the range is freed."""
    if not host_ptr:
        return
    try:
        import acl.rt as acl_rt

        ret = acl_rt.host_unregister(host_ptr)
    except Exception as exc:  # pragma: no cover - depends on the CANN install
        logger.warning("aclrtHostUnregister unavailable: %s", exc)
        return
    if ret != 0:
        logger.warning("aclrtHostUnregister failed for %#x: ret=%s", host_ptr, ret)
