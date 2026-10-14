# SPDX-License-Identifier: Apache-2.0
"""Process-local mappings of exportable CUDA HOST_NUMA allocations.

The daemon allocates once. Clients import the CUDA allocation FD and establish
CPU/GPU access without cudaHostRegister. Tensor users must retire before close.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def _checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA host VMM operation failed: {result}")
    return result[1] if len(result) == 2 else result[1:]


def _cuda_driver():
    try:
        from cuda.bindings import driver
    except ImportError as error:
        raise RuntimeError("CUDA host VMM requires cuda.bindings.driver") from error
    # Establish Torch's current primary context; do not retain a second context.
    torch.empty(0, device="cuda")
    return driver, torch.cuda.current_device()


class _CudaArrayView:
    def __init__(self, address, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (int(address), False),
            "version": 3,
        }


class HostVmmAllocation:
    def __init__(self, driver, device):
        self.driver, self.device = driver, device
        self.fd = None
        self.handle = None
        self.address = None
        self.mapped = False
        self.bytes = None
        self.closed = False
        self.gpu_view = False
        self.mapping_timings = {}

    @staticmethod
    def _props(driver, numa_id):
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = numa_id
        return prop

    @staticmethod
    def _require_support(driver, device):
        for name in (
            "CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
            "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
        ):
            attribute = getattr(driver.CUdevice_attribute, name, None)
            if (
                attribute is None
                or _checked(driver.cuDeviceGetAttribute(attribute, device)) != 1
            ):
                raise RuntimeError(f"CUDA host VMM requires {name} on device {device}")

    @classmethod
    def create(cls, nbytes):
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Invalid host VMM logical size")
        driver, device = _cuda_driver()
        result = cls(driver, device)
        try:
            cls._require_support(driver, device)
            numa_id = _checked(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
                )
            )
            result.numa_id = max(0, numa_id)
            result.nbytes = nbytes
            prop = cls._props(driver, result.numa_id)
            granularity = _checked(
                driver.cuMemGetAllocationGranularity(
                    prop,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            )
            result.allocation_nbytes = (
                (nbytes + granularity - 1) // granularity * granularity
            )
            tic = time.perf_counter()
            result.handle = _checked(
                driver.cuMemCreate(result.allocation_nbytes, prop, 0)
            )
            result.fd = int(
                _checked(
                    driver.cuMemExportToShareableHandle(
                        result.handle, prop.requestedHandleTypes, 0
                    )
                )
            )
            os.set_inheritable(result.fd, False)
            result._map()
            logger.info(
                "Engram VMM create: bytes=%d, allocation_bytes=%d, numa=%d, elapsed=%.3fs",
                nbytes,
                result.allocation_nbytes,
                result.numa_id,
                time.perf_counter() - tic,
            )
            return result
        except Exception:
            result.close()
            raise

    @classmethod
    def from_fd(cls, fd, nbytes, allocation_nbytes, numa_id, *, gpu_view=False):
        """Adopt fd, including on failure; verify the imported CUDA allocation."""
        result = None
        try:
            if (
                type(nbytes) is not int
                or nbytes <= 0
                or type(allocation_nbytes) is not int
                or allocation_nbytes < nbytes
                or type(numa_id) is not int
                or numa_id < 0
            ):
                raise ValueError("Invalid host VMM allocation descriptor")
            driver, device = _cuda_driver()
            result = cls(driver, device)
            result.fd = fd
            result.gpu_view = gpu_view
            result.nbytes, result.allocation_nbytes, result.numa_id = (
                nbytes,
                allocation_nbytes,
                numa_id,
            )
            cls._require_support(driver, device)
            tic = time.perf_counter()
            result.handle = _checked(
                driver.cuMemImportFromShareableHandle(
                    fd,
                    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                )
            )
            prop = _checked(
                driver.cuMemGetAllocationPropertiesFromHandle(result.handle)
            )
            if (
                prop.type != driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
                or prop.location.type
                != driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
                or prop.location.id != numa_id
            ):
                raise ValueError(
                    "Imported allocation is not the declared HOST_NUMA backing"
                )
            granularity = _checked(
                driver.cuMemGetAllocationGranularity(
                    prop,
                    driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                )
            )
            if (
                allocation_nbytes
                != (nbytes + granularity - 1) // granularity * granularity
            ):
                raise ValueError("Host VMM allocation size/granularity mismatch")
            result._map()
            logger.info(
                "Engram VMM import: bytes=%d, allocation_bytes=%d, numa=%d, elapsed=%.3fs",
                nbytes,
                allocation_nbytes,
                numa_id,
                time.perf_counter() - tic,
            )
            return result
        except Exception:
            if result is not None:
                result.close()
            else:
                os.close(fd)
            raise

    def _map(self):
        driver = self.driver
        tic = time.perf_counter()
        self.address = _checked(
            driver.cuMemAddressReserve(self.allocation_nbytes, 0, 0, 0)
        )
        self.mapping_timings["reserve"] = time.perf_counter() - tic
        tic = time.perf_counter()
        _checked(
            driver.cuMemMap(self.address, self.allocation_nbytes, 0, self.handle, 0)
        )
        self.mapping_timings["map"] = time.perf_counter() - tic
        self.mapped = True
        host = driver.CUmemAccessDesc()
        host.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        host.location.id = self.numa_id
        host.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        gpu = driver.CUmemAccessDesc()
        gpu.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        gpu.location.id = self.device
        gpu.flags = host.flags
        descriptors = [gpu] if self.gpu_view else [host, gpu]
        tic = time.perf_counter()
        _checked(
            driver.cuMemSetAccess(
                self.address, self.allocation_nbytes, descriptors, len(descriptors)
            )
        )
        self.mapping_timings["access"] = time.perf_counter() - tic
        if self.gpu_view:
            # Inference only needs GPU access. A CUDA tensor view represents
            # HOST_NUMA backing without eagerly installing CPU PTEs for it.
            self.bytes = torch.as_tensor(
                _CudaArrayView(self.address, self.nbytes), device=f"cuda:{self.device}"
            )
            if self.bytes.data_ptr() != int(self.address):
                raise RuntimeError("CUDA host VMM view unexpectedly copied storage")
        else:
            self.bytes = torch.frombuffer(
                (ctypes.c_uint8 * self.nbytes).from_address(int(self.address)),
                dtype=torch.uint8,
            )
        logger.info(
            "Engram VMM mapping: gpu_view=%s, bytes=%d, steps=%s",
            self.gpu_view,
            self.nbytes,
            self.mapping_timings,
        )

    @property
    def device_ptr(self):
        return int(self.address)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.bytes = None
        # Attempt all cleanup steps, even if a context has already shut down.
        failures = []
        if self.mapped:
            try:
                _checked(self.driver.cuMemUnmap(self.address, self.allocation_nbytes))
            except Exception as error:
                failures.append(error)
            self.mapped = False
        if self.address is not None:
            try:
                _checked(
                    self.driver.cuMemAddressFree(self.address, self.allocation_nbytes)
                )
            except Exception as error:
                failures.append(error)
            self.address = None
        if self.handle is not None:
            try:
                _checked(self.driver.cuMemRelease(self.handle))
            except Exception as error:
                failures.append(error)
            self.handle = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if failures:
            logger.warning("CUDA host VMM cleanup failures: %s", failures)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
