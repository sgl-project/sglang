"""FlashInfer symbols the pinned release (0.6.15.post1) does not ship yet."""

from flashinfer.cuda_utils import checkCudaErrors

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


def is_multicast_supported(device_idx: int) -> bool:
    """Return True if the device supports NVLink multicast (cuMulticastCreate; SM90+ NVLink)."""
    # Copied from flashinfer.comm.mnnvl, which first ships it in 0.6.16.
    try:
        multicast_supported = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                device_idx,
            )
        )
        return multicast_supported != 0
    except Exception:
        return False
