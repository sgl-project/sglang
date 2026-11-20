import ctypes

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    transfer_hicache_one_layer_mla,
    use_hicache_tma_kernel,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="VMM host allocations need CUDA.",
)

NUM_LAYERS = 4
ROWS = 64
# page_size=1 never tiles a TMA chunk, so the loads take the register kernel,
# which resolves host tensors through runtime::get_device_accessible_ptr.
PAGE_SIZE = 1


class VmmHostBuffer:
    """Pinned HOST_NUMA memory mapped read-write for one GPU and the CPU."""

    def __init__(self, nbytes: int, device_index: int):
        try:
            from cuda.bindings import driver as cu

            from sglang.srt.utils.cuda_vmm_utils import check_drv
        except ImportError as e:
            pytest.skip(f"cuda.bindings is unavailable: {e}")
        self._cu, self._check = cu, check_drv
        torch.cuda.set_device(device_index)
        torch.cuda.init()
        dev = check_drv(cu.cuDeviceGet(device_index), "cuDeviceGet")
        attr = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
        numa = check_drv(cu.cuDeviceGetAttribute(attr, dev), "cuDeviceGetAttribute")
        if numa < 0:
            pytest.skip("the device reports no host NUMA node")

        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = numa
        flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        gran = check_drv(cu.cuMemGetAllocationGranularity(prop, flag), "granularity")
        self.size = (nbytes + gran - 1) // gran * gran
        err, handle = cu.cuMemCreate(self.size, prop, 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            pytest.skip(f"HOST_NUMA cuMemCreate is unsupported here: {err}")
        self._handle = handle
        self._va = check_drv(cu.cuMemAddressReserve(self.size, 0, 0, 0), "reserve")
        check_drv(cu.cuMemMap(self._va, self.size, 0, handle, 0), "cuMemMap")
        descs = []
        for loc_type, loc_id in (
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device_index),
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, numa),
        ):
            desc = cu.CUmemAccessDesc()
            desc.location.type = loc_type
            desc.location.id = loc_id
            desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            descs.append(desc)
        check_drv(cu.cuMemSetAccess(self._va, self.size, descs, len(descs)), "access")
        buf = (ctypes.c_uint8 * self.size).from_address(int(self._va))
        self.tensor = torch.frombuffer(buf, dtype=torch.uint8)

    def close(self):
        cu, check = self._cu, self._check
        self.tensor = None
        torch.cuda.synchronize()
        check(cu.cuMemUnmap(self._va, self.size), "cuMemUnmap")
        check(cu.cuMemAddressFree(self._va, self.size), "cuMemAddressFree")
        check(cu.cuMemRelease(self._handle), "cuMemRelease")


@pytest.mark.parametrize("dim", [256, 576])
def test_load_interior_slices_of_vmm_host_memory(dim):
    """Each per-layer slice of a VMM host pool loads its own bytes, not layer 0's."""
    dtype = torch.bfloat16
    element_size = dim * dtype.itemsize
    if use_hicache_tma_kernel(
        element_size=element_size, block_quota=4, page_size=PAGE_SIZE
    ):
        pytest.skip("this configuration takes the TMA kernel")
    if not can_use_hicache_jit_kernel(element_size=element_size, page_size=PAGE_SIZE):
        pytest.skip(f"no register kernel for {element_size}-byte rows")

    vmm = VmmHostBuffer(
        nbytes=NUM_LAYERS * ROWS * element_size,
        device_index=torch.cuda.current_device(),
    )
    try:
        host = vmm.tensor[: NUM_LAYERS * ROWS * element_size].view(dtype)
        host = host.view(NUM_LAYERS, ROWS, dim)
        rows = torch.arange(ROWS, dtype=torch.float32).unsqueeze(1).expand(ROWS, dim)
        for layer in range(NUM_LAYERS):
            host[layer].copy_(rows + 64 * layer)  # exact in bf16
        indices = torch.arange(ROWS, dtype=torch.int64, device="cuda")
        for layer in range(NUM_LAYERS):
            dst = torch.zeros(ROWS, dim, dtype=dtype, device="cuda")
            transfer_hicache_one_layer_mla(
                dst, indices, host[layer], indices, page_size=PAGE_SIZE
            )
            torch.cuda.synchronize()
            assert torch.equal(dst.cpu(), host[layer]), f"layer {layer}"
    finally:
        vmm.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
