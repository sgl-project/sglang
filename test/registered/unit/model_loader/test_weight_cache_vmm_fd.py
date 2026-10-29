"""
CPU-only unit tests for the weight cache's vmm_fd transport.

The CUDA driver calls (cuMemCreate, fd export/import, cuMemMap) need a GPU and
are left to the GPU tests. Everything around them runs here, with host memory
standing in for the VMM allocations:

  - the packing plan: slots, alignment, allocation sizes, aliasing
  - the daemon's export path (WeightCacheDaemon._export_state ->
    VmmFdTransportBackend.prepare_export): values, ties, and freeing every
    original as its allocation is filled
  - the fetch_state exchange over a socketpair, then rebuilding every tensor
    on the engine side
  - engine-side layout validation
  - fd passing, including that every fd is closed on success, on error and on
    a malformed message, and that received fds are close-on-exec
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
from sglang.srt.weight_cache.protocol import recv_msg
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
def _host_vmm():
    """Run the vmm_fd backend's packing on host memory."""
    with (
        mock.patch.object(cuda_vmm_utils, "VmmReservation", _HostReservation),
        mock.patch.object(
            cuda_vmm_utils, "tensor_from_pointer", _cpu_tensor_from_pointer
        ),
        mock.patch("torch.cuda.synchronize"),
        mock.patch("torch.cuda.empty_cache") as empty_cache,
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


def _make_daemon(model, backend) -> WeightCacheDaemon:
    """A daemon with a loaded model, without the GPU/distributed setup."""
    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.model = model
    daemon.gpu_id = 0
    daemon.transport_name = backend.name
    daemon.transport_backend = backend
    daemon.state_entries = {}
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
        freed_at_each_allocation = []
        with _host_vmm() as empty_cache:
            empty_cache.side_effect = lambda: freed_at_each_allocation.append(
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

        # Every original is freed once its allocation is filled, not at the end.
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
