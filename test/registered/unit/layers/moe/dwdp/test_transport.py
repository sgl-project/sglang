"""Local weight conversion must recover from allocation OOM without losing handles."""

from __future__ import annotations

import ctypes
import importlib
import sys
import unittest
from collections import Counter
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Result(IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2


class _Driver:
    CUresult = _Result
    CUmemAllocationGranularity_flags = SimpleNamespace(
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=1
    )

    def __init__(self, faults):
        self.faults = faults
        self.calls = Counter()
        self.events = []
        self.reservations = {}
        self.handles = {}
        self.mappings = {}
        self.device = 7
        self.next_handle = 100
        self.next_address = 4096

    def step(self, operation, *args):
        self.calls[operation] += 1
        self.events.append((operation, self.device, args))
        fault = self.faults.get(operation, {}).get(self.calls[operation], 0)
        if isinstance(fault, BaseException):
            raise fault
        return _Result(fault)

    def cuMemAddressReserve(self, size, alignment, address, flags):
        error = self.step("reserve", size)
        if error:
            return (error,)
        self.next_address += 4096
        self.reservations[self.next_address] = size
        return error, self.next_address

    def cuMemCreate(self, size, prop, flags):
        error = self.step("create", size, prop.location.id)
        if error:
            return (error,)
        self.next_handle += 1
        self.handles[self.next_handle] = bytearray(size)
        return error, self.next_handle

    def cuMemMap(self, address, size, offset, handle, flags):
        error = self.step("map", address, size, handle)
        if not error:
            assert address in self.reservations and handle in self.handles
            self.mappings[address] = (size, handle)
        return (error,)

    def cuMemSetAccess(self, address, size, descriptors, count):
        return (self.step("access"),)

    def cuMemUnmap(self, address, size):
        error = self.step("unmap", address)
        if not error:
            assert self.mappings[address][0] == size
            del self.mappings[address]
        return (error,)

    def cuMemRelease(self, handle):
        error = self.step("release", handle)
        if not error:
            assert handle in self.handles, "duplicate or unowned handle release"
            del self.handles[handle]
        return (error,)

    def cuMemAddressFree(self, address, size):
        error = self.step("free", address)
        if not error and address in self.mappings:
            error = _Result.CUDA_ERROR_INVALID_VALUE
        if not error:
            assert self.reservations.pop(address) == size
        return (error,)

    def cuMemcpyDtoD(self, target, source, nbytes):
        error = self.step("copy", target, nbytes)
        if not error:
            base = next(
                a for a, (size, _) in self.mappings.items() if a <= target < a + size
            )
            _, handle = self.mappings[base]
            offset = target - base
            self.handles[handle][offset : offset + nbytes] = ctypes.string_at(
                source, nbytes
            )
        return (error,)

    @contextmanager
    def use_device(self, device):
        previous, self.device = self.device, device
        try:
            yield
        finally:
            self.device = previous


class TestLocalWeightHandles(CustomTestCase):
    def setUp(self):
        self.driver = _Driver({})
        cuda_package = ModuleType("cuda")
        bindings = ModuleType("cuda.bindings")
        bindings.driver = self.driver
        cuda_package.bindings = bindings
        # Parent package initializers load GPU runners unrelated to transport.
        layers = importlib.import_module("sglang.srt.layers")
        utils = importlib.import_module("sglang.srt.utils")
        self.enterContext(patch.object(layers, "moe", None, create=True))
        self.enterContext(patch.object(utils, "cuda_vmm_utils", None, create=True))
        moe_path = Path(next(iter(layers.__path__))) / "moe"
        packages = {}
        for name, path in (
            ("sglang.srt.layers.moe", moe_path),
            ("sglang.srt.layers.moe.dwdp", moe_path / "dwdp"),
        ):
            package = ModuleType(name)
            package.__path__ = [str(path)]
            packages[name] = package
        self.enterContext(
            patch.dict(
                sys.modules,
                {"cuda": cuda_package, "cuda.bindings": bindings, **packages},
            )
        )
        self.vmm = importlib.import_module("sglang.srt.utils.cuda_vmm_utils")
        self.transport = importlib.import_module("sglang.srt.layers.moe.dwdp.transport")
        layers.moe = packages["sglang.srt.layers.moe"]
        utils.cuda_vmm_utils = self.vmm
        self.enterContext(patch.object(self.vmm, "_drv", self.driver))
        self.enterContext(patch.object(self.transport, "cuda", self.driver))
        self.enterContext(
            patch.object(self.vmm, "make_rw_access_desc", lambda device: device)
        )
        self.enterContext(
            patch.object(self.transport, "get_device_granularity", lambda device: 16)
        )
        self.enterContext(
            patch.object(
                self.transport,
                "make_device_allocation_prop",
                lambda device: SimpleNamespace(location=SimpleNamespace(id=device)),
            )
        )
        self.enterContext(patch.object(torch.cuda, "device", self.driver.use_device))
        self.enterContext(
            patch.object(torch.cuda, "empty_cache", lambda: self.driver.step("clear"))
        )
        self.enterContext(
            patch.object(torch.cuda, "synchronize", lambda: self.driver.step("sync"))
        )

    def convert(self, count=2):
        self.params = {
            (i, "weight"): torch.arange(16, dtype=torch.uint8) + i for i in range(count)
        }
        specs = {i: {"weight": SimpleNamespace(expert_bytes=1)} for i in range(count)}
        return self.transport._copy_local_weights_to_handles(
            sorted(self.params),
            self.params,
            specs,
            SimpleNamespace(local_expert_start=3, local_expert_end=19),
            0,
        )

    def assert_released(self):
        self.assertEqual(self.driver.reservations, {})
        self.assertEqual(self.driver.mappings, {})
        self.assertEqual(self.driver.handles, {})
        released = [args[0] for op, _, args in self.driver.events if op == "release"]
        self.assertEqual(len(released), len(set(released)))
        self.assertEqual(self.driver.device, 7)

    def test_success_and_each_shard_oom_retry_transfer_ownership(self):
        for create_faults in ({}, {1: 2}, {2: 2}, {1: 2, 3: 2}):
            with self.subTest(create_faults=create_faults):
                self.driver.__init__({"create": create_faults})
                handles, sizes = self.convert()
                self.assertEqual(sizes, {(0, "weight"): 32, (1, "weight"): 32})
                self.assertEqual(set(handles.values()), set(self.driver.handles))
                for (index, _), handle in handles.items():
                    self.assertEqual(
                        self.driver.handles[handle][3:19],
                        bytes(range(index, index + 16)),
                    )
                    self.assertEqual(
                        self.params[(index, "weight")].untyped_storage().nbytes(), 0
                    )
                self.assertEqual(
                    [args for op, _, args in self.driver.events if op == "create"],
                    [(32, 0)] * (2 + len(create_faults)),
                )
                self.assertEqual(self.driver.calls["reserve"], 2)
                self.assertEqual(self.driver.calls["create"], 2 + len(create_faults))
                self.assertEqual(self.driver.calls["release"], 0)
                clear_devices = [
                    device for op, device, _ in self.driver.events if op == "clear"
                ]
                self.assertEqual(clear_devices, [0] * len(create_faults) + [7])
                self.assertEqual(self.driver.reservations, {})
                self.assertEqual(self.driver.mappings, {})
                for handle in handles.values():
                    self.driver.cuMemRelease(handle)
                self.assert_released()

    def test_retry_failure_releases_prior_handles_and_keeps_source_state(self):
        primary = LookupError("retry failed")
        for retry_fault in (2, primary):
            with self.subTest(retry_fault=retry_fault):
                self.driver.__init__({"create": {2: 2, 3: retry_fault}})
                with self.assertRaises(
                    RuntimeError if retry_fault == 2 else LookupError
                ) as caught:
                    self.convert()
                if retry_fault is primary:
                    self.assertIs(caught.exception, primary)
                else:
                    self.assertIs(
                        type(caught.exception), self.vmm.VmmAllocationOutOfMemory
                    )
                self.assertEqual(self.driver.calls["create"], 3)
                self.assertEqual(self.driver.calls["clear"], 1)
                self.assertEqual(
                    [p.untyped_storage().nbytes() for p in self.params.values()],
                    [0, 16],
                )
                self.assert_released()

    def test_other_driver_oom_does_not_trigger_retry(self):
        for operation, label in (
            ("reserve", "cuMemAddressReserve"),
            ("map", "cuMemMap"),
            ("access", "cuMemSetAccess"),
            ("copy", "cuMemcpyDtoD"),
        ):
            with self.subTest(operation=operation):
                self.driver.__init__({operation: {2: 2}})
                with self.assertRaisesRegex(RuntimeError, label):
                    self.convert()
                self.assertEqual(self.driver.calls["clear"], 0)
                self.assertEqual(
                    self.driver.calls["create"], 1 if operation == "reserve" else 2
                )
                self.assert_released()

    def test_cache_failure_preserves_error_and_releases_handles(self):
        for faults, expected_sizes in (
            ({"create": {2: 2}, "clear": {1: LookupError("retry clear")}}, [0, 16]),
            ({"clear": {1: LookupError("tail clear")}}, [0, 0]),
        ):
            with self.subTest(faults=faults):
                self.driver.__init__(faults)
                primary = faults["clear"][1]
                with self.assertRaises(LookupError) as caught:
                    self.convert()
                self.assertIs(caught.exception, primary)
                self.assertEqual(
                    [p.untyped_storage().nbytes() for p in self.params.values()],
                    expected_sizes,
                )
                self.assert_released()

    def test_rollback_failures_keep_primary_error_and_continue_releasing(self):
        primary = LookupError("tail clear")
        cleanup = OSError("release failure")
        self.driver.faults = {"clear": {1: primary}, "release": {1: cleanup}}
        with self.assertRaises(LookupError) as caught:
            self.convert()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(len(primary.__notes__), 1)
        self.assertEqual(self.driver.calls["release"], 2)
        self.assertEqual(set(self.driver.handles), {101})
        self.assertEqual(self.driver.reservations, {})
        self.assertEqual(self.driver.mappings, {})
        self.driver.cuMemRelease(101)

    def test_non_oom_create_failure_does_not_retry(self):
        self.driver.faults = {"create": {1: 1}}
        with self.assertRaisesRegex(RuntimeError, "cuMemCreate") as caught:
            self.convert(1)
        self.assertIs(type(caught.exception), RuntimeError)
        self.assertEqual(self.driver.calls["create"], 1)
        self.assertEqual(self.driver.calls["clear"], 0)
        self.assert_released()

    def test_close_failure_does_not_prevent_other_handle_releases(self):
        primary, cleanup = LookupError("copy failed"), OSError("close failed")
        self.driver.faults = {"copy": {2: primary}, "unmap": {2: cleanup}}
        with self.assertRaises(LookupError) as caught:
            self.convert()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(self.driver.calls["release"], 2)
        self.assertEqual(self.driver.handles, {})
        self.assertEqual(len(self.driver.mappings), 1)
        self.assertEqual(len(self.driver.reservations), 1)
        self.assertEqual(self.driver.device, 7)

    def test_mapping_rollback_error_preserves_primary_and_does_not_retry(self):
        primary, cleanup = LookupError("access failed"), OSError("unmap failed")
        self.driver.faults = {"access": {1: primary}, "unmap": {1: cleanup}}
        with self.assertRaises(LookupError) as caught:
            self.convert(1)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cleanup)
        self.assertEqual(self.driver.calls["create"], 1)
        self.assertEqual(self.driver.calls["clear"], 0)
        self.assertEqual(self.driver.handles, {})
        self.assertEqual(len(self.driver.mappings), 1)
        self.assertEqual(len(self.driver.reservations), 1)


if __name__ == "__main__":
    unittest.main()
