"""
CPU-only unit tests for the weight cache's vmm_fd transport.

The CUDA driver calls (cuMemCreate, fd export/import, cuMemMap) need a GPU and
are left to the GPU tests. Everything around them runs here, with host memory
standing in for the VMM allocations:

  - the packing plan: slots, alignment, allocation sizes, aliasing
  - the daemon's export path (WeightCacheDaemon._export_state ->
    VmmFdTransportBackend.prepare_export): values, ties, and freeing every
    original as its allocation is filled
  - the packing plan's sizes and the free device memory, logged before
    anything is allocated or copied, and the warning when the free memory is
    below the largest allocation
  - the fetch_state exchange over a socketpair, then rebuilding every tensor
    on the engine side
  - engine-side layout validation
  - fd passing, including that every fd is closed on success, on error and on
    a malformed message, and that received fds are close-on-exec
  - a client that breaks off a fetch part-way, followed by a healthy one
  - a loaded model keeps its transport mappings after the model object goes
  - --weight-cache-transport selection and the non-fatal daemon-exit watcher
"""

import argparse
import array
import contextlib
import copy
import gc
import io
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from sglang.srt.utils import cuda_vmm_utils
from sglang.srt.weight_cache import ipc_loader, transport
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import CacheConfig, recv_msg, send_msg
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    VMM_FD_BACKEND,
    TorchIpcTransportBackend,
    VmmFdTransportBackend,
    check_served_transport,
    create_daemon_transport_backend,
    get_client_transport_backend,
    recv_fds,
    resolve_daemon_transport,
    send_fds,
)
from sglang.srt.weight_cache.vmm_layout import (
    SLOT_ALIGNMENT,
    plan_vmm_pack,
    validate_vmm_layout,
    view_from_entry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CPU = torch.device("cpu")
GRANULARITY = 4096


def _cpu_storage(pointer: int, nbytes: int) -> torch.UntypedStorage:
    return torch._C._construct_storage_from_data_pointer(pointer, CPU, nbytes)


def _cpu_tensor_from_pointer(
    pointer, nbytes, *, shape=None, dtype=torch.uint8, device_id
) -> torch.Tensor:
    """Host-memory stand-in for cuda_vmm_utils.tensor_from_pointer."""
    storage = _cpu_storage(pointer, nbytes)
    return torch.empty(0, dtype=dtype).set_(storage, 0, shape or (nbytes,))


class _HostReservation:
    """Host-memory stand-in for cuda_vmm_utils.VmmReservation."""

    def __init__(self, size, prop, device_id, *, alignment=0, requested_address=0):
        self.size = size
        self.buffer = torch.zeros(size, dtype=torch.uint8)
        self.base = self.buffer.data_ptr()
        self.mapped = []
        self.close_calls = 0

    def map(self, offset, size, *, retain_handle):
        self.mapped.append((offset, size))
        return f"handle@{offset}"

    def close(self, *, release_handles=True):
        self.close_calls += 1


@contextlib.contextmanager
def _host_vmm(free_bytes=1 << 40):
    """Run the vmm_fd backend's packing on host memory, with ``free_bytes`` of
    free device memory."""
    with (
        mock.patch.object(cuda_vmm_utils, "VmmReservation", _HostReservation),
        mock.patch.object(
            cuda_vmm_utils, "tensor_from_pointer", _cpu_tensor_from_pointer
        ),
        mock.patch("torch.cuda.synchronize"),
        mock.patch("torch.cuda.empty_cache") as empty_cache,
        mock.patch("torch.cuda.mem_get_info", return_value=(free_bytes, free_bytes)),
    ):
        yield empty_cache


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _memfds(contents):
    fds = []
    for index, data in enumerate(contents):
        fd = os.memfd_create(f"vmm-fd-test-{index}")
        os.write(fd, data)
        fds.append(fd)
    return fds


def _close_all(fds):
    for fd in fds:
        os.close(fd)


class _FdExportDriver:
    """Stand-in for the CUDA driver in the daemon's fd export: each export of a
    retained allocation handle opens a fresh memfd that names the handle."""

    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUmemAllocationHandleType = SimpleNamespace(
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR=1
    )

    def __init__(self):
        self.exported = 0  # fds opened so far

    def cuMemExportToShareableHandle(self, handle, handle_type, flags):
        self.exported += 1
        return self.CUresult.CUDA_SUCCESS, _memfds([handle.encode()])[0]


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.proj = torch.nn.Linear(4, 6)
        self.head = torch.nn.Linear(4, 10, bias=False)
        self.head.weight = self.embed.weight  # tied
        # Post-processing can leave a parameter as a transposed view.
        self.proj_t = torch.nn.Parameter(torch.randn(6, 4).t(), requires_grad=False)
        self.register_buffer("scale", torch.tensor([0.5]))
        self.register_buffer("empty", torch.empty(0, 3))
        self.register_buffer(
            "cache",
            torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            persistent=False,
        )


def _cache_config() -> CacheConfig:
    return CacheConfig(
        model_path="tiny",
        model_arch="TinyModel",
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        attn_dp_size=1,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="torch.float32",
        revision="",
        device_capability="",
        torch_version="2.0.0",
    )


def _make_daemon(model, backend) -> WeightCacheDaemon:
    """A daemon with a loaded model, without the GPU/distributed setup."""
    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.model = model
    daemon.gpu_id = 0
    daemon.transport_name = backend.name
    daemon.transport_backend = backend
    daemon.state_entries = {}
    daemon.config = _cache_config()
    daemon.preloaded_weights_bytes = 0
    return daemon


def _slot_key(entry):
    return entry["alloc_index"], entry["offset"], entry["storage_nbytes"]


class TestVmmPackPlan(CustomTestCase):
    def _plan(self, state, max_alloc_bytes=1 << 20):
        return plan_vmm_pack(
            state,
            device=CPU,
            granularity=GRANULARITY,
            max_alloc_bytes=max_alloc_bytes,
        )

    def test_slots_are_aligned_and_inside_their_allocation(self):
        state = {
            f"t{i}": (torch.randn(n), True) for i, n in enumerate([3, 100, 1, 700])
        }
        plan = self._plan(state)
        self.assertEqual(len(plan.alloc_sizes), 1)
        self.assertEqual(plan.alloc_sizes[0] % GRANULARITY, 0)
        offsets = [plan.entries[f"t{i}"]["offset"] for i in range(4)]
        self.assertEqual(offsets, sorted(offsets))  # first-use order
        for slot in plan.slots:
            self.assertEqual(slot.offset % SLOT_ALIGNMENT, 0)
            self.assertLessEqual(
                slot.offset + slot.nbytes, plan.alloc_sizes[slot.alloc_index]
            )
        validate_vmm_layout(plan.entries, plan.alloc_sizes)

    def test_allocations_are_capped(self):
        state = {f"t{i}": (torch.zeros(256), True) for i in range(10)}  # 1 KiB each
        plan = self._plan(state, max_alloc_bytes=4096)
        self.assertEqual(
            [plan.entries[f"t{i}"]["alloc_index"] for i in range(10)],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
        )
        self.assertEqual(plan.alloc_sizes, [4096, 4096, 4096])

    def test_storage_larger_than_the_cap_gets_its_own_allocation(self):
        state = {
            "small": (torch.zeros(16), True),
            "big": (torch.zeros(5000), True),  # 20000 bytes
            "after": (torch.zeros(16), True),
        }
        plan = self._plan(state, max_alloc_bytes=4096)
        self.assertEqual(
            [plan.entries[n]["alloc_index"] for n in ("small", "big", "after")],
            [0, 1, 2],
        )
        self.assertEqual(plan.alloc_sizes, [4096, 20480, 4096])

    def test_tied_and_view_tensors_share_one_slot(self):
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        state = {
            "w": (base, True),
            "tied": (base, True),
            "w_t": (base.t(), True),
            "row": (base[1], False),
            "as_int": (base.view(torch.int32), False),
            "other": (torch.ones(3), True),
        }
        plan = self._plan(state)
        entries = plan.entries
        shared = {_slot_key(entries[n]) for n in ("w", "tied", "w_t", "row", "as_int")}
        self.assertEqual(len(shared), 1)
        self.assertNotIn(_slot_key(entries["other"]), shared)
        self.assertEqual(len(plan.slots), 2)
        self.assertEqual(entries["w_t"]["stride"], [1, 6])
        self.assertEqual(entries["row"]["storage_offset"], 6)
        self.assertEqual(entries["as_int"]["dtype"], "int32")
        self.assertEqual(entries["w"]["storage_nbytes"], 96)

    def test_overlapping_storages_merge_into_one_slot(self):
        backing = torch.arange(64, dtype=torch.uint8)
        a = torch.empty(0, dtype=torch.uint8).set_(
            _cpu_storage(backing.data_ptr(), 48), 0, (48,)
        )
        b = torch.empty(0, dtype=torch.int16).set_(
            _cpu_storage(backing.data_ptr() + 16, 48), 0, (24,)
        )
        plan = self._plan({"a": (a, True), "b": (b, True)})
        self.assertEqual(len(plan.slots), 1)
        self.assertEqual(plan.slots[0].nbytes, 64)
        self.assertEqual(_slot_key(plan.entries["a"]), _slot_key(plan.entries["b"]))
        self.assertEqual(plan.entries["b"]["storage_offset"], 8)  # 16 bytes / int16

    def test_overlap_that_breaks_element_alignment_is_rejected(self):
        backing = torch.zeros(64, dtype=torch.uint8)
        a = torch.empty(0, dtype=torch.uint8).set_(
            _cpu_storage(backing.data_ptr(), 48), 0, (48,)
        )
        b = torch.empty(0, dtype=torch.int16).set_(
            _cpu_storage(backing.data_ptr() + 1, 48), 0, (24,)
        )
        with self.assertRaisesRegex(ValueError, "element size"):
            self._plan({"a": (a, True), "b": (b, True)})

    def test_tensor_on_another_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected cpu"):
            self._plan({"m": (torch.empty(2, device="meta"), True)})

    def test_empty_tensor_gets_a_zero_byte_slot(self):
        plan = self._plan({"e": (torch.empty(0, 3), False), "x": (torch.ones(2), True)})
        self.assertEqual(plan.entries["e"]["storage_nbytes"], 0)
        self.assertEqual(plan.entries["e"]["shape"], [0, 3])
        validate_vmm_layout(plan.entries, plan.alloc_sizes)


class TestVmmExportAndImport(CustomTestCase):
    """Daemon export through the real backend and daemon code, the
    fetch_state exchange over a socketpair, then the engine-side import."""

    def test_round_trip(self):
        torch.manual_seed(0)
        model = _TinyModel()
        model_ref = weakref.ref(model)
        # Model/loader callbacks can create cycles that refcounting cannot free.
        model.python_cycle = [model]
        names = list(model.state_dict()) + ["cache"]
        tensors = dict(model.named_parameters(remove_duplicate=False))
        tensors.update(model.named_buffers(remove_duplicate=False))
        expected = {name: tensors[name].detach().clone() for name in names}
        originals = {
            name: StorageWeakRef(tensors[name].untyped_storage())
            for name in names
            if tensors[name].untyped_storage().nbytes()
        }
        del tensors

        # One slot per allocation, so packing runs over several allocations.
        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=SLOT_ALIGNMENT
        )
        freed_at_each_empty_cache = []
        with _host_vmm() as empty_cache:
            empty_cache.side_effect = lambda: freed_at_each_empty_cache.append(
                {name for name, ref in originals.items() if ref.expired()}
            )
            daemon = _make_daemon(model, backend)
            del model
            daemon._export_state()
        self.assertIsNone(model_ref())
        self.assertIsNone(daemon.model)
        entries = daemon.state_entries
        reservation = backend._reservation

        self.assertEqual(list(entries), names)
        self.assertGreater(len(backend._alloc_sizes), 3)
        self.assertEqual(
            reservation.mapped, list(zip(backend._alloc_offsets, backend._alloc_sizes))
        )
        self.assertEqual(backend.exported_bytes, reservation.size)

        # The cache is emptied once before packing, when the plan still holds
        # every original, and then after each allocation. Every original is
        # freed once its allocation is filled, not at the end.
        before_packing, *freed_at_each_allocation = freed_at_each_empty_cache
        self.assertEqual(before_packing, set())
        self.assertEqual(len(freed_at_each_allocation), len(backend._alloc_sizes))
        for alloc_index, freed in enumerate(freed_at_each_allocation):
            packed_so_far = {
                name
                for name in originals
                if entries[name]["alloc_index"] <= alloc_index
            }
            self.assertEqual(freed, packed_so_far)

        # fetch_state: layout first, then one fd per allocation.
        before = _open_fd_count()
        exported = _memfds([b"alloc-%d" % i for i in range(len(backend._alloc_sizes))])
        exported_inodes = [os.fstat(fd).st_ino for fd in exported]
        received_inodes = []
        client = VmmFdTransportBackend()

        def fake_map(fds, sizes):
            # Stand-in for import + cuMemMap: map the same host memory.
            received_inodes.extend(os.fstat(fd).st_ino for fd in fds)
            client._reservation = SimpleNamespace(
                base=reservation.base, close=lambda: None
            )
            client._device = CPU
            client._alloc_sizes = list(sizes)
            client._alloc_offsets = transport._prefix_offsets(sizes)

        a, b = socket.socketpair()
        try:
            with (
                mock.patch.object(backend, "_export_fds", return_value=exported),
                mock.patch.object(client, "_map_imported", side_effect=fake_map),
                mock.patch.object(
                    cuda_vmm_utils, "tensor_from_pointer", _cpu_tensor_from_pointer
                ),
            ):
                backend.send_fetch_state_response(
                    a, config={"k": "v"}, entries=entries, pid=4321
                )
                response = recv_msg(b)
                self.assertEqual(response["transport_backend"], VMM_FD_BACKEND)
                self.assertEqual(response["allocation_sizes"], backend._alloc_sizes)
                response = client.recv_fetch_state_response(b, response)
                imported = {
                    name: client.import_tensor(entry)
                    for name, entry in response["entries"].items()
                }
        finally:
            a.close()
            b.close()
        self.assertEqual(received_inodes, exported_inodes)
        self.assertEqual(_open_fd_count(), before)  # both sides closed every fd

        for name in names:
            tensor = imported[name]
            self.assertEqual(tensor.shape, expected[name].shape, name)
            self.assertEqual(tensor.dtype, expected[name].dtype, name)
            self.assertEqual(tensor.stride(), expected[name].stride(), name)
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        storage_ptr = {
            name: imported[name].untyped_storage().data_ptr()
            for name in names
            if imported[name].numel()
        }
        self.assertEqual(storage_ptr["embed.weight"], storage_ptr["head.weight"])
        self.assertEqual(
            len(set(storage_ptr.values())), len(storage_ptr) - 1
        )  # only the tie shares

        client.close()
        backend.close()
        self.assertEqual(reservation.close_calls, 1)

    def test_cached_buffer_owner_is_released_before_packing(self):
        from sglang.srt.layers.rotary_embedding import factory

        model = _TinyModel()
        cached_layer = torch.nn.Module()
        cached_layer.register_buffer("view", model.cache[1], persistent=False)
        owner_ref = weakref.ref(cached_layer)
        original = StorageWeakRef(model.cache.untyped_storage())
        cache = {"example": cached_layer}
        daemon = _make_daemon(
            model,
            VmmFdTransportBackend(device=CPU, granularity=GRANULARITY),
        )
        del model, cached_layer
        with mock.patch.object(factory, "_ROPE_DICT", cache), _host_vmm():
            daemon._export_state()
        self.assertFalse(cache)
        self.assertIsNone(owner_ref())
        self.assertTrue(original.expired())
        self.assertIsNone(daemon.model)
        daemon.transport_backend.close()

    def test_export_log_reports_the_allocations(self):
        # vmm_fd entries carry no handle blob, so a handle-size estimate would
        # always read zero.
        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=SLOT_ALIGNMENT
        )
        daemon = _make_daemon(_TinyModel(), backend)
        with (
            _host_vmm(),
            self.assertLogs("sglang.srt.weight_cache.daemon", "INFO") as logs,
        ):
            daemon._export_state()
        (line,) = [msg for msg in logs.output if "non-persistent buffers" in msg]
        self.assertGreater(backend.num_allocations, 1)
        self.assertTrue(
            line.endswith(
                f"transport=vmm_fd, {backend.num_allocations} allocation(s), "
                f"{backend.exported_bytes / (1 << 30):.2f} GiB"
            ),
            line,
        )
        self.assertNotIn("metadata size", line)
        backend.close()

    def test_state_dict_hook_output_is_exported_not_the_live_tensor(self):
        model = torch.nn.Linear(2, 3, bias=False)
        live_weight = model.weight.detach().clone()

        def double_weight(module, state_dict, prefix, local_metadata):
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"] * 2

        model._register_state_dict_hook(double_weight)
        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=1 << 20
        )
        with _host_vmm():
            daemon = _make_daemon(model, backend)
            daemon._export_state()
        entry = daemon.state_entries["weight"]
        pointer = (
            backend._reservation.base
            + backend._alloc_offsets[entry["alloc_index"]]
            + entry["offset"]
        )
        exported = view_from_entry(
            _cpu_storage(pointer, entry["storage_nbytes"]), entry, CPU
        )
        torch.testing.assert_close(exported, live_weight * 2, rtol=0, atol=0)
        # The live parameter was not repointed at the hook's output.
        torch.testing.assert_close(model.weight.detach(), live_weight, rtol=0, atol=0)


class TestPackPlanLog(CustomTestCase):
    """prepare_export logs the sizes of its packing plan and the free device
    memory before it allocates or copies anything, and warns, without
    stopping, when the free memory is below the largest allocation."""

    # 20000 bytes in "big": with a 4 KiB cap it gets an allocation of its own,
    # of 20480 bytes.
    LARGEST_ALLOCATION = 20480

    @staticmethod
    def _state_with_a_large_storage():
        return {
            "small": (torch.arange(16, dtype=torch.float32), True),
            "big": (torch.arange(5000, dtype=torch.float32), True),
            "after": (torch.ones(16), True),
        }

    def _pack(self, state, max_alloc_bytes, free_bytes=1 << 40):
        """Pack ``state`` on host memory; returns the backend, the plan line
        and the warnings."""
        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=max_alloc_bytes
        )
        with (
            _host_vmm(free_bytes),
            self.assertLogs(transport.logger, "INFO") as logs,
        ):
            backend.prepare_export(state)
        self.addCleanup(backend.close)
        (line,) = [msg for msg in logs.output if "pack plan" in msg]
        warnings = [msg for msg in logs.output if msg.startswith("WARNING:")]
        return backend, line, warnings

    def test_a_storage_larger_than_the_cap_sets_the_largest_allocation(self):
        cap = GRANULARITY
        backend, line, _ = self._pack(self._state_with_a_large_storage(), cap)
        largest = max(backend._alloc_sizes)
        self.assertEqual(largest, self.LARGEST_ALLOCATION)
        self.assertGreater(largest, cap)
        self.assertIn(
            f"3 slot(s) in 3 allocation(s), cap {cap} bytes, largest slot 20000 "
            f"bytes (big), largest allocation {largest} bytes, "
            f"{sum(backend._alloc_sizes)} bytes in total",
            line,
        )

    def test_storages_smaller_than_the_cap_keep_every_allocation_within_it(self):
        # The cap is a multiple of the granularity, as 1 GiB is on CUDA.
        cap = GRANULARITY
        backend, line, _ = self._pack(
            {f"t{i}": (torch.zeros(256), True) for i in range(10)},  # 1 KiB each
            cap,
        )
        largest = max(backend._alloc_sizes)
        self.assertLessEqual(largest, cap)
        self.assertIn(
            f"largest slot 1024 bytes (t0), largest allocation {largest} bytes", line
        )

    def test_the_plan_line_reports_the_free_memory(self):
        _, line, warnings = self._pack(
            self._state_with_a_large_storage(), GRANULARITY, free_bytes=123456789
        )
        self.assertTrue(line.endswith(", 123456789 bytes free"), line)
        self.assertEqual(warnings, [])

    def test_short_free_memory_warns_and_packing_goes_on(self):
        free = self.LARGEST_ALLOCATION - 1
        short, _, warnings = self._pack(
            self._state_with_a_large_storage(), GRANULARITY, free_bytes=free
        )
        (warning,) = warnings
        self.assertIn(
            f"{free} bytes of device memory are free, less than the largest "
            f"allocation ({self.LARGEST_ALLOCATION} bytes)",
            warning,
        )
        # Packed the same as with enough free memory.
        enough, _, _ = self._pack(self._state_with_a_large_storage(), GRANULARITY)
        self.assertEqual(short._alloc_sizes, enough._alloc_sizes)
        self.assertEqual(short.exported_bytes, enough.exported_bytes)
        self.assertTrue(
            torch.equal(short._reservation.buffer, enough._reservation.buffer)
        )

    def test_free_memory_that_fits_the_largest_allocation_does_not_warn(self):
        _, _, warnings = self._pack(
            self._state_with_a_large_storage(),
            GRANULARITY,
            free_bytes=self.LARGEST_ALLOCATION,
        )
        self.assertEqual(warnings, [])

    def test_the_plan_is_logged_before_anything_is_allocated_or_copied(self):
        events = []

        class RecordingReservation(_HostReservation):
            def __init__(self, *args, **kwargs):
                events.append("reserve")
                super().__init__(*args, **kwargs)

            def map(self, offset, size, *, retain_handle):
                events.append("map")
                return super().map(offset, size, retain_handle=retain_handle)

        pack_slot = transport.pack_slot

        def recording_pack_slot(slot, dst):
            events.append("copy")
            pack_slot(slot, dst)

        def recording_info(msg, *args):
            events.append(msg % args)

        def recording_mem_get_info(device):
            events.append("mem_get_info")
            return 1 << 40, 1 << 40

        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=SLOT_ALIGNMENT
        )
        self.addCleanup(backend.close)
        with (
            _host_vmm() as empty_cache,
            mock.patch.object(cuda_vmm_utils, "VmmReservation", RecordingReservation),
            mock.patch.object(transport, "pack_slot", recording_pack_slot),
            mock.patch.object(transport.logger, "info", side_effect=recording_info),
            mock.patch("torch.cuda.mem_get_info", side_effect=recording_mem_get_info),
        ):
            empty_cache.side_effect = lambda: events.append("empty_cache")
            backend.prepare_export({f"t{i}": (torch.randn(8), True) for i in range(3)})
        self.assertEqual(events[:2], ["empty_cache", "mem_get_info"])
        self.assertIn("pack plan", events[2])
        self.assertLessEqual({"reserve", "map", "copy"}, set(events[3:]))


class TestValidateVmmLayout(CustomTestCase):
    def setUp(self):
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        plan = plan_vmm_pack(
            {"w": (base, True), "w_t": (base.t(), True), "b": (torch.ones(3), False)},
            device=CPU,
            granularity=GRANULARITY,
            max_alloc_bytes=1 << 20,
        )
        self.entries, self.sizes = plan.entries, plan.alloc_sizes

    def _assert_rejected(self, mutate, sizes=None):
        entries = copy.deepcopy(self.entries)
        mutate(entries)
        with self.assertRaises(ValueError):
            validate_vmm_layout(entries, self.sizes if sizes is None else sizes)

    def test_accepts_the_plan(self):
        validate_vmm_layout(self.entries, self.sizes)

    def test_rejects_bad_allocation_sizes(self):
        for sizes in ([0], [-4096], [True], "4096", None, [4096.0]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                validate_vmm_layout(self.entries, sizes)

    def test_rejects_entries_outside_their_allocation(self):
        def set_field(field, value):
            return lambda entries: entries["w"].__setitem__(field, value)

        for mutate in (
            set_field("alloc_index", 1),
            set_field("alloc_index", -1),
            set_field("offset", self.sizes[0]),
            set_field("offset", 8),  # not slot-aligned
            set_field("storage_nbytes", self.sizes[0] + 1),
            set_field("storage_offset", -1),
            set_field("offset", True),
        ):
            self._assert_rejected(mutate)

    def test_rejects_a_view_larger_than_its_storage(self):
        self._assert_rejected(lambda e: e["w"].__setitem__("shape", [5, 6]))
        self._assert_rejected(lambda e: e["w_t"].__setitem__("storage_offset", 1))

    def test_rejects_malformed_views(self):
        for field, value in (
            ("dtype", "not_a_dtype"),
            ("dtype", "Tensor"),
            ("shape", [4]),
            ("stride", [6, -1]),
            ("shape", "4,6"),
            ("is_param", 1),
        ):
            with self.subTest(field=field, value=value):
                self._assert_rejected(
                    lambda e, f=field, v=value: e["w"].__setitem__(f, v)
                )
        self._assert_rejected(lambda e: e["w"].pop("stride"))

    def test_rejects_overlapping_slots(self):
        self._assert_rejected(
            lambda e: e["b"].__setitem__("offset", self.entries["w"]["offset"])
        )


class TestFdPassing(CustomTestCase):
    def test_fds_arrive_in_order(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"zero", b"one", b"two"])
        try:
            send_fds(a, fds)
            received = recv_fds(b, 3)
            self.assertEqual(
                [os.pread(fd, 8, 0) for fd in received], [b"zero", b"one", b"two"]
            )
            _close_all(received)
        finally:
            _close_all(fds)
            a.close()
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_out_of_order_fd_closes_what_was_received(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"x", b"y"])
        try:
            transport._send_fd(a, fds[0], 0)
            transport._send_fd(a, fds[1], 5)
            with self.assertRaisesRegex(RuntimeError, "expected #1"):
                recv_fds(b, 2)
        finally:
            _close_all(fds)
            a.close()
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_peer_hangup_closes_what_was_received(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"x"])
        try:
            transport._send_fd(a, fds[0], 0)
            a.close()
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                recv_fds(b, 3)
        finally:
            _close_all(fds)
            b.close()
        self.assertEqual(_open_fd_count(), before)

    @staticmethod
    def _send_raw(sock, payload, fds):
        rights = array.array("i", fds).tobytes()
        sock.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)])

    def test_short_header_with_an_fd_closes_the_fd(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"x"])
        try:
            self._send_raw(a, b"\0", fds)  # one byte of the header, then hang up
            a.close()
            with self.assertRaisesRegex(RuntimeError, "truncated fd header: 1 <"):
                recv_fds(b, 1)
        finally:
            _close_all(fds)
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_header_split_across_reads_still_arrives(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"x"])
        header = transport._FD_INDEX_STRUCT.pack(0)
        try:
            self._send_raw(a, header[:1], fds)
            a.sendall(header[1:])
            received = recv_fds(b, 1)
            self.assertEqual(os.pread(received[0], 8, 0), b"x")
            _close_all(received)
        finally:
            _close_all(fds)
            a.close()
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_extra_fds_in_one_message_are_refused_and_closed(self):
        before = _open_fd_count()
        a, b = socket.socketpair()
        fds = _memfds([b"x", b"y"])
        try:
            self._send_raw(a, transport._FD_INDEX_STRUCT.pack(0), fds)
            with self.assertRaisesRegex(RuntimeError, "truncated|expected one fd"):
                recv_fds(b, 1)
        finally:
            _close_all(fds)
            a.close()
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_received_fds_are_close_on_exec(self):
        a, b = socket.socketpair()
        fds = _memfds([b"x"])
        try:
            send_fds(a, fds)
            received = recv_fds(b, 1)
            self.assertFalse(os.get_inheritable(received[0]))
            _close_all(received)
        finally:
            _close_all(fds)
            a.close()
            b.close()

    def test_daemon_closes_exported_fds_when_sending_fails(self):
        before = _open_fd_count()
        backend = VmmFdTransportBackend()
        backend._alloc_sizes = [GRANULARITY, GRANULARITY]
        a, b = socket.socketpair()
        b.close()
        exported = _memfds([b"a", b"b"])
        with mock.patch.object(backend, "_export_fds", return_value=exported):
            with self.assertRaises(OSError):
                backend.send_fetch_state_response(a, config={}, entries={}, pid=1)
        a.close()
        self.assertEqual(_open_fd_count(), before)

    def _send_layout(self, entries, sizes, fd_contents):
        """Daemon side of fetch_state; returns (daemon socket, engine socket)."""
        backend = VmmFdTransportBackend()
        backend._alloc_sizes = sizes
        a, b = socket.socketpair()
        with mock.patch.object(
            backend, "_export_fds", return_value=_memfds(fd_contents)
        ):
            backend.send_fetch_state_response(a, config={}, entries=entries, pid=1)
        return a, b

    def test_engine_rejects_a_bad_layout_before_reading_any_fd(self):
        before = _open_fd_count()
        bad = {
            "x": {
                "alloc_index": 3,
                "offset": 0,
                "storage_nbytes": 4,
                "storage_offset": 0,
                "shape": [1],
                "stride": [1],
                "dtype": "float32",
                "is_param": True,
            }
        }
        a, b = self._send_layout(bad, [GRANULARITY], [b"a"])
        client = VmmFdTransportBackend()
        try:
            with mock.patch.object(client, "_map_imported") as map_imported:
                with self.assertRaisesRegex(ValueError, "does not fit"):
                    client.recv_fetch_state_response(b, recv_msg(b))
            map_imported.assert_not_called()
        finally:
            a.close()
            b.close()  # drops the fd still queued on the socket
        self.assertEqual(_open_fd_count(), before)

    def test_engine_closes_received_fds_when_mapping_fails(self):
        before = _open_fd_count()
        a, b = self._send_layout({}, [GRANULARITY, GRANULARITY], [b"a", b"b"])
        client = VmmFdTransportBackend()
        try:
            with mock.patch.object(
                client, "_map_imported", side_effect=RuntimeError("map failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "map failed"):
                    client.recv_fetch_state_response(b, recv_msg(b))
        finally:
            a.close()
            b.close()
        self.assertEqual(_open_fd_count(), before)

    def test_close_is_idempotent(self):
        backend = VmmFdTransportBackend()
        reservation = mock.Mock()
        backend._reservation = reservation
        backend._storages[(0, 0, 0)] = object()
        backend.close()
        backend.close()
        reservation.close.assert_called_once_with()
        self.assertEqual(backend._storages, {})


class TestAttachAfterAFailedAttach(CustomTestCase):
    """A client that breaks off a fetch_state part-way leaves the daemon able
    to serve the next one: every fd exported or received for the broken
    request is closed, the daemon keeps its allocation handles, and the next
    client gets fresh fds for them and reads every tensor.

    The daemon handles each connection with its request handler, as serve()
    does; memfds and host memory stand in for the driver's fd export and the
    engine's mapping, so the handles here are placeholders (the GPU tests map
    real allocations). Both sides run in this process, so the fd count covers
    both."""

    # Where the client gives up, in the order of the exchange.
    BREAK_OFF_POINTS = (
        "after_request",  # before the daemon answers: its send fails
        "after_response",  # whatever fds were sent are left on the socket
        "midway_through_fds",  # the daemon's send fails after half of them
        "mapping_fails",  # after every fd arrived
        "after_mapping",  # without importing any tensor
    )
    # The daemon's send fails at these points; at the others it may finish.
    DAEMON_SEND_FAILS = ("after_request", "midway_through_fds")

    def setUp(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tensors = dict(model.named_parameters(remove_duplicate=False))
        tensors.update(model.named_buffers(remove_duplicate=False))
        self.expected = {name: t.detach().clone() for name, t in tensors.items()}
        del tensors
        # One slot per allocation, so that a fetch passes several fds.
        backend = VmmFdTransportBackend(
            device=CPU, granularity=GRANULARITY, max_alloc_bytes=SLOT_ALIGNMENT
        )
        self.daemon = _make_daemon(model, backend)
        del model
        with _host_vmm():
            self.daemon._export_state()
        self.addCleanup(backend.close)
        self.driver = _FdExportDriver()
        for patcher in (
            mock.patch.object(
                cuda_vmm_utils, "_get_cuda_driver", return_value=self.driver
            ),
            mock.patch.object(
                cuda_vmm_utils, "tensor_from_pointer", _cpu_tensor_from_pointer
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Per mapping, the handle that each received fd was exported from.
        self.mapped = []

    def _serve(self, conn, errors):
        """Handle one connection in a thread, as WeightCacheDaemon.serve() does."""

        def handle():
            try:
                self.daemon._handle_connection(conn)
            except Exception as e:  # serve() logs it and goes on
                errors.append(e)
            finally:
                conn.close()

        thread = threading.Thread(target=handle)
        thread.start()
        return thread

    def _map(self, client, fds, sizes):
        """Stand-in for import + cuMemMap: note the handles the fds name, then
        map the daemon's host memory."""
        self.mapped.append([os.pread(fd, 64, 0).decode() for fd in fds])
        client._reservation = SimpleNamespace(
            base=self.daemon.transport_backend._reservation.base, close=lambda: None
        )
        client._device = CPU
        client._alloc_sizes = list(sizes)
        client._alloc_offsets = transport._prefix_offsets(sizes)

    def _attach(self, break_off=None):
        """One fetch_state on a fresh connection, which the client gives up at
        ``break_off``. Returns copies of every tensor the client imported
        (None if it gave up) and the errors the daemon raised."""
        daemon_sock, client_sock = socket.socketpair()
        for sock in (daemon_sock, client_sock):
            sock.settimeout(10)
        half = len(self.daemon.transport_backend._handles) // 2
        client_gone = threading.Event()
        send_fd = transport._send_fd

        def send_fd_after_half_waits_for_the_client(sock, fd, index):
            if break_off == "midway_through_fds" and index == half:
                # Not an OSError, so that the test cannot mistake it for the
                # failed send it expects.
                if not client_gone.wait(10):
                    raise AssertionError("the client did not close its socket")
            send_fd(sock, fd, index)

        request = {"type": "fetch_state", "config": _cache_config().to_dict()}
        send_msg(client_sock, request)
        if break_off == "after_request":
            client_sock.close()
        errors = []
        client = VmmFdTransportBackend()
        with mock.patch.object(
            transport, "_send_fd", send_fd_after_half_waits_for_the_client
        ):
            thread = self._serve(daemon_sock, errors)
            try:
                if break_off == "after_request":
                    return None, errors
                response = recv_msg(client_sock)
                self.assertEqual(response["status"], "ok")
                if break_off == "after_response":
                    return None, errors
                if break_off == "midway_through_fds":
                    _close_all(recv_fds(client_sock, half))
                    client_sock.close()
                    return None, errors
                if break_off == "mapping_fails":
                    with (
                        mock.patch.object(
                            client,
                            "_map_imported",
                            side_effect=RuntimeError("map failed"),
                        ),
                        self.assertRaisesRegex(RuntimeError, "map failed"),
                    ):
                        client.recv_fetch_state_response(client_sock, response)
                    return None, errors
                with mock.patch.object(
                    client,
                    "_map_imported",
                    side_effect=lambda fds, sizes: self._map(client, fds, sizes),
                ):
                    client.recv_fetch_state_response(client_sock, response)
                if break_off == "after_mapping":
                    return None, errors
                # Copy the values out while the mapping is still valid.
                imported = {
                    name: client.import_tensor(entry).clone()
                    for name, entry in response["entries"].items()
                }
                return imported, errors
            finally:
                client.close()
                client_sock.close()
                client_gone.set()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())

    def test_a_failed_attachment_is_followed_by_a_healthy_one(self):
        backend = self.daemon.transport_backend
        handles = list(backend._handles)
        self.assertGreater(len(handles), 1)
        before = _open_fd_count()
        for break_off in self.BREAK_OFF_POINTS:
            with self.subTest(break_off=break_off):
                exported = self.driver.exported
                _, errors = self._attach(break_off)
                if break_off in self.DAEMON_SEND_FAILS:
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], OSError)  # e.g. EPIPE
                # The broken request got fresh fds, and all of them are closed.
                self.assertEqual(self.driver.exported - exported, len(handles))
                self.assertEqual(_open_fd_count(), before)

                self.mapped.clear()
                imported, errors = self._attach()
                self.assertEqual(errors, [])
                self.assertEqual(_open_fd_count(), before)
                # Fresh fds for the same retained handles, in order.
                self.assertEqual(backend._handles, handles)
                self.assertEqual(self.mapped, [handles])
                self.assertEqual(set(imported), set(self.expected))
                for name, tensor in self.expected.items():
                    torch.testing.assert_close(imported[name], tensor, rtol=0, atol=0)


class TestTransportSelection(CustomTestCase):
    def test_daemon_resolution(self):
        self.assertEqual(resolve_daemon_transport("auto"), TORCH_IPC_BACKEND)
        self.assertEqual(resolve_daemon_transport(None), TORCH_IPC_BACKEND)
        self.assertEqual(resolve_daemon_transport("torch_ipc"), TORCH_IPC_BACKEND)
        self.assertEqual(resolve_daemon_transport("vmm_fd"), VMM_FD_BACKEND)
        with self.assertRaises(ValueError):
            resolve_daemon_transport("nccl")

    def test_daemon_backend_creation(self):
        self.assertIsInstance(
            create_daemon_transport_backend(TORCH_IPC_BACKEND, 0),
            TorchIpcTransportBackend,
        )
        with self.assertRaises(ValueError):
            create_daemon_transport_backend("auto", 0)

    def test_engine_accepts_auto_or_a_match_and_refuses_a_mismatch(self):
        for requested, served in (
            ("auto", TORCH_IPC_BACKEND),
            ("auto", VMM_FD_BACKEND),
            (None, VMM_FD_BACKEND),
            (VMM_FD_BACKEND, VMM_FD_BACKEND),
        ):
            check_served_transport(requested, served)
        with self.assertRaisesRegex(RuntimeError, "--weight-cache-transport vmm_fd"):
            check_served_transport(VMM_FD_BACKEND, TORCH_IPC_BACKEND)
        with self.assertRaises(RuntimeError):
            check_served_transport(TORCH_IPC_BACKEND, VMM_FD_BACKEND)

    def test_engine_backend_for_vmm_fd(self):
        self.assertIsInstance(
            get_client_transport_backend(VMM_FD_BACKEND), VmmFdTransportBackend
        )

    def test_cli_flag(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model-path", "m"])
        self.assertEqual(args.weight_cache_transport, "auto")
        args = parser.parse_args(
            ["--model-path", "m", "--weight-cache-transport", "vmm_fd"]
        )
        self.assertEqual(args.weight_cache_transport, "vmm_fd")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--model-path", "m", "--weight-cache-transport", "x"])


class TestLoadedModelKeepsItsMappings(CustomTestCase):
    def test_a_parameter_outliving_the_model_stays_mapped(self):
        """The imported tensors do not own the transport's mappings, and a
        parameter can outlive the model object, so the loader must not release
        the mappings when the model goes away."""
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat

        closed = []

        class _Recording(TorchIpcTransportBackend):
            def import_tensor(self, entry):
                return torch.full(entry["shape"], 3.0)

            def close(self):
                closed.append(self)

        loader = ipc_loader.WeightCacheModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE)
        )
        backend = loader._transport_backend = _Recording()
        meta_model = torch.nn.Linear(2, 3, bias=False, device="meta")
        with mock.patch.object(
            ipc_loader, "_initialize_model", return_value=meta_model
        ):
            model = loader._load_zero_copy_mode(
                SimpleNamespace(dtype=torch.float32),
                None,
                {"weight": {"shape": [3, 2], "is_param": True}},
                None,
            )
        weight = model.weight
        del model, meta_model
        gc.collect()
        try:
            self.assertEqual(closed, [])
            self.assertIn(backend, ipc_loader._LOADED_MODEL_TRANSPORTS)
            self.assertEqual(weight.sum().item(), 18.0)
        finally:
            ipc_loader._LOADED_MODEL_TRANSPORTS.remove(backend)


class TestDaemonExitWatcher(CustomTestCase):
    def test_non_fatal_watcher_only_logs(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat

        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        loader = ipc_loader.WeightCacheModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            weight_cache_mode="client",
        )
        with (
            mock.patch.object(ipc_loader, "_DAEMON_LIVENESS_POLL_INTERVAL", 0.01),
            mock.patch.object(ipc_loader.os, "kill", wraps=os.kill) as kill,
            self.assertLogs(ipc_loader.logger, "WARNING") as logs,
        ):
            loader._start_daemon_liveness_watchdog(child.pid, fatal=False)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and any(
                t.name == "weight-cache-daemon-watchdog" for t in threading.enumerate()
            ):
                time.sleep(0.01)
        self.assertIn("serving continues", "\n".join(logs.output))
        self.assertNotIn(
            mock.call(os.getpid(), ipc_loader.signal.SIGKILL), kill.call_args_list
        )


if __name__ == "__main__":
    unittest.main()
