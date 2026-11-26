# SPDX-License-Identifier: Apache-2.0
"""The weight cache's vmm_fd transport on Intel XPU hardware.

Two processes and the real transport, with no model loaded: a daemon packs
tensors into Level Zero allocations and serves the layout plus one dma-buf fd
per allocation over a unix socket; an engine imports them, maps them at
addresses it reserves for itself, and rebuilds every tensor from the wire
metadata.

Equal bytes alone would not show that the weights were reused rather than
reloaded -- a copy satisfies that too. Three checks do:

- the dma-buf's ``(st_dev, st_ino)`` is the same in both processes, which is the
  kernel's own identity for the buffer;
- the two processes map it at different addresses, so neither inherited the
  other's; and
- a write the daemon makes *after* the engine has mapped is visible to the
  engine, which no import-time copy could explain.

The CPU-only half of this transport is in
test/registered/unit/model_loader/test_weight_cache_vmm_fd_xpu.py, and the
daemon-to-engine server path in test_weight_cache_daemon_xpu.py.
"""

import multiprocessing as mp
import os
import socket
import tempfile
import time
import unittest

import torch

from sglang.srt.layers.moe.utils import XPU_MOE_LD_PADDING_BYTES
from sglang.srt.utils import xpu_vmm_utils as vmm
from sglang.srt.weight_cache.protocol import recv_msg
from sglang.srt.weight_cache.transport import VMM_FD_BACKEND
from sglang.srt.weight_cache.vmm_fd_xpu import (
    XpuVmmFdTransportBackend,
    allocation_granularity,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=180, suite="extra-test-1-gpu-xpu")

DEVICE_INDEX = 0
_CONNECT_TIMEOUT_S = 60.0
_JOIN_TIMEOUT_S = 300.0

# Small enough that several allocations are needed for the tensors below, so
# the layout is exercised across allocation boundaries rather than inside one.
_MAX_ALLOC_BYTES = 2 << 20
_CACHE_ELEMENTS = 1 << 20  # 2 MiB of bfloat16, so it spills past the cap

_LATE_VALUE = 7.5
_ENGINE_VALUE = -3.25
_EXPERT_K_PAD = XPU_MOE_LD_PADDING_BYTES // torch.float32.itemsize


def _unsupported_reason():
    try:
        return vmm.vmm_unsupported_reason(device_index=DEVICE_INDEX)
    except Exception as exc:  # the loader itself may be missing
        return str(exc)


_SKIP_REASON = None if torch.xpu.is_available() else "no XPU device"
if _SKIP_REASON is None:
    _SKIP_REASON = _unsupported_reason()
requires_vmm = unittest.skipIf(
    _SKIP_REASON is not None, f"Level Zero VMM unavailable: {_SKIP_REASON}"
)


def _padded_expert_weight(device):
    experts, n_dim, k_dim = 3, 5, 8
    buffer = torch.full(
        (experts, n_dim, k_dim + _EXPERT_K_PAD),
        -11.0,
        dtype=torch.float32,
        device=device,
    )
    buffer[:, :, :k_dim] = (
        torch.arange(
            experts * n_dim * k_dim, dtype=torch.float32, device=device
        ).reshape(experts, n_dim, k_dim)
        % 97
    )
    return buffer[:, :, :k_dim]


def _expected_tensors(device):
    """The daemon's tensors: a storage shared by two views, a plain vector, a
    buffer large enough to need an allocation of its own, and an empty one.

    Every value is an exact small integer, so the same tensor is produced on CPU
    and on XPU and the comparison needs no tolerance.
    """
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
    cache = torch.arange(_CACHE_ELEMENTS, dtype=torch.float32, device=device) % 97
    return {
        "w": (base, True),
        "w_t": (base.t(), True),
        "bias": (torch.arange(6, dtype=torch.float32, device=device) - 3, True),
        "cache": (cache.to(torch.bfloat16), False),
        "empty": (torch.empty(0, 3, dtype=torch.float32, device=device), False),
        "experts.w13_weight": (_padded_expert_weight(device), True),
    }


def _open_fds() -> int:
    # listdir allocates one descriptor for the directory itself; discount it.
    return len(os.listdir("/proc/self/fd")) - 1


def _daemon(path: str, out, ready, engine_wrote, late_written, stop) -> None:
    """Pack tensors into Level Zero allocations and serve them.

    Serves every client that connects until ``stop`` is set, and between accepts
    answers the two handshakes the visibility test needs: read ``bias`` once the
    engine has written it, then overwrite it once the engine has mapped.
    """
    try:
        torch.xpu.set_device(DEVICE_INDEX)
        device = torch.device("xpu", DEVICE_INDEX)
        backend = XpuVmmFdTransportBackend(
            device=device,
            granularity=allocation_granularity(DEVICE_INDEX),
            max_alloc_bytes=_MAX_ALLOC_BYTES,
        )
        entries = backend.prepare_export(_expected_tensors(device))
        out["num_allocations"] = backend.num_allocations
        out["exported_bytes"] = backend.exported_bytes
        out["reservation_base"] = backend._reservation_base
        out["identities"] = _daemon_identities(backend)
        out["pid"] = os.getpid()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(4)
        server.settimeout(0.5)
        fds_before_serving = _open_fds()
        ready.set()

        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                _answer_handshakes(
                    backend, entries, device, out, engine_wrote, late_written
                )
                continue
            try:
                backend.send_fetch_state_response(
                    conn, config={"test": "xpu"}, entries=entries, pid=os.getpid()
                )
            finally:
                conn.close()
        out["fds_leaked_while_serving"] = _open_fds() - fds_before_serving
        server.close()
        backend.close()
        out["error"] = None
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        ready.set()
        raise


def _daemon_identities(backend):
    """The kernel's identity for each packed allocation."""
    identities = []
    for physical in backend._handles:
        fd = physical.export_dma_buf()
        try:
            identity = vmm.dma_buf_identity(fd)
            identities.append((identity.st_dev, identity.st_ino))
        finally:
            os.close(fd)
    return identities


def _answer_handshakes(backend, entries, device, out, engine_wrote, late_written):
    """Read, then rewrite, the daemon's own view of the packed ``bias``."""
    bias = backend.import_tensor(entries["bias"])
    if engine_wrote.is_set() and "bias_the_daemon_read" not in out:
        torch.xpu.synchronize(device)
        out["bias_the_daemon_read"] = bias.detach().cpu().tolist()
    elif late_written.is_set() and "late_write_done" not in out:
        bias.fill_(_LATE_VALUE)
        torch.xpu.synchronize(device)
        out["late_write_done"] = True


class _Engine:
    """The engine side: connect, import, and rebuild every tensor.

    ``identities`` comes from the fds as they arrive, which is the only place
    the engine sees them: an imported allocation cannot be exported again.
    """

    class _Backend(XpuVmmFdTransportBackend):
        def _map_imported(self, fds, sizes):
            self.identities = [
                (identity.st_dev, identity.st_ino)
                for identity in (vmm.dma_buf_identity(fd) for fd in fds)
            ]
            super()._map_imported(fds, sizes)

    def __init__(self, path: str):
        self.backend = self._Backend()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT_S)
        sock.connect(path)
        try:
            self.response = self.backend.recv_fetch_state_response(sock, recv_msg(sock))
        finally:
            sock.close()
        self.tensors = {
            name: self.backend.import_tensor(entry)
            for name, entry in self.response["entries"].items()
        }

    @property
    def base(self) -> int:
        return self.backend._reservation_base

    def identities(self):
        return self.backend.identities

    def close(self):
        self.backend.close()


@requires_vmm
class TestXpuWeightCacheTransport(CustomTestCase):
    """One daemon process, one or more engine attachments, real transport."""

    @classmethod
    def setUpClass(cls):
        cls.ctx = mp.get_context("spawn")
        cls.manager = cls.ctx.Manager()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "manager", None) is not None:
            cls.manager.shutdown()

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="sglang-wcd-xpu-")
        self.path = os.path.join(self.directory, "weights.sock")
        self.out = self.manager.dict()
        self.ready = self.ctx.Event()
        self.engine_wrote = self.ctx.Event()
        self.late_written = self.ctx.Event()
        self.stop = self.ctx.Event()
        self.daemon = self.ctx.Process(
            target=_daemon,
            args=(
                self.path,
                self.out,
                self.ready,
                self.engine_wrote,
                self.late_written,
                self.stop,
            ),
        )
        self.daemon.start()
        self.addCleanup(self._cleanup)
        self.assertTrue(
            self.ready.wait(timeout=_JOIN_TIMEOUT_S),
            f"daemon never became ready: {dict(self.out)}",
        )
        self.assertIsNone(self.out.get("error"), self.out.get("error"))
        torch.xpu.set_device(DEVICE_INDEX)

    def _cleanup(self):
        self.stop.set()
        self.late_written.set()
        self.daemon.join(_JOIN_TIMEOUT_S)
        if self.daemon.is_alive():
            self.daemon.terminate()
            self.daemon.join(_JOIN_TIMEOUT_S)
        for path in (self.path,):
            if os.path.exists(path):
                os.unlink(path)
        if os.path.isdir(self.directory):
            os.rmdir(self.directory)

    def _attach(self) -> _Engine:
        engine = _Engine(self.path)
        self.addCleanup(engine.close)
        return engine

    def test_the_engine_rebuilds_every_tensor_from_the_daemons_memory(self):
        engine = self._attach()
        expected = _expected_tensors(torch.device("cpu"))

        self.assertEqual(engine.response["transport_backend"], VMM_FD_BACKEND)
        self.assertEqual(engine.response["device_type"], "xpu")
        self.assertGreater(
            self.out["num_allocations"], 1, "the cap should have split the tensors"
        )
        self.assertEqual(
            len(engine.response["allocation_sizes"]), self.out["num_allocations"]
        )
        self.assertEqual(set(engine.tensors), set(expected))

        for name, (reference, _) in expected.items():
            tensor = engine.tensors[name]
            with self.subTest(tensor=name):
                self.assertEqual(tensor.device.type, "xpu")
                self.assertEqual(tensor.shape, reference.shape)
                self.assertEqual(tensor.dtype, reference.dtype)
                self.assertEqual(tensor.stride(), reference.stride())
                torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)

        self.assertEqual(
            engine.tensors["w"].untyped_storage().data_ptr(),
            engine.tensors["w_t"].untyped_storage().data_ptr(),
            "two views of one daemon storage must share one storage here too",
        )
        self.assertFalse(engine.tensors["w_t"].is_contiguous())

    def test_a_padded_moe_expert_weight_keeps_its_row_stride_and_its_slack(self):
        engine = self._attach()
        tensor = engine.tensors["experts.w13_weight"]
        reference = _padded_expert_weight(torch.device("cpu"))

        self.assertFalse(tensor.is_contiguous())
        self.assertEqual(tensor.stride(), reference.stride())
        self.assertEqual(tensor.stride(-2), tensor.shape[-1] + _EXPERT_K_PAD)
        self.assertEqual(
            tensor.untyped_storage().nbytes(),
            reference.untyped_storage().nbytes(),
        )
        self.assertGreater(
            tensor.untyped_storage().nbytes(),
            tensor.numel() * tensor.element_size(),
        )
        torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)

    def test_the_engine_maps_the_daemons_buffers_at_its_own_addresses(self):
        """Equal bytes would also hold for a copy; the kernel's dma-buf identity
        and a separate address range are what show the memory is shared."""
        engine = self._attach()
        self.assertEqual(
            [tuple(identity) for identity in self.out["identities"]],
            engine.identities(),
            "the engine holds different dma-bufs, so nothing was shared",
        )
        self.assertNotEqual(
            engine.base,
            self.out["reservation_base"],
            "both processes mapped at one address, so the engine did not choose "
            "its own",
        )

    def test_two_engines_hold_the_same_allocations_at_once(self):
        """One physical handle carries one live mapping on Arc Pro B60, so
        concurrent clients only work because each holds a handle of its own for
        the one dma-buf."""
        first = self._attach()
        second = self._attach()
        self.assertEqual(first.identities(), second.identities())
        self.assertNotEqual(first.base, second.base)
        reference = _expected_tensors(torch.device("cpu"))["w"][0]
        for engine in (first, second):
            torch.testing.assert_close(
                engine.tensors["w"].cpu(), reference, rtol=0, atol=0
            )
        # Dropping one must leave the other's mapping intact.
        second.close()
        torch.xpu.synchronize(torch.device("xpu", DEVICE_INDEX))
        torch.testing.assert_close(first.tensors["w"].cpu(), reference, rtol=0, atol=0)

    def _await(self, key):
        deadline = time.monotonic() + _JOIN_TIMEOUT_S
        while self.out.get(key) is None and time.monotonic() < deadline:
            self.assertTrue(self.daemon.is_alive(), f"daemon died: {dict(self.out)}")
            time.sleep(0.2)
        self.assertIsNotNone(
            self.out.get(key), f"daemon never reported {key}: {dict(self.out)}"
        )
        return self.out[key]

    def test_writes_after_the_engine_mapped_are_visible_both_ways(self):
        """Neither direction could be explained by a copy taken at import."""
        engine = self._attach()
        device = torch.device("xpu", DEVICE_INDEX)

        engine.tensors["bias"].fill_(_ENGINE_VALUE)
        torch.xpu.synchronize(device)
        self.engine_wrote.set()
        self.assertEqual(
            self._await("bias_the_daemon_read"),
            [_ENGINE_VALUE] * 6,
            "the engine's write is not visible to the daemon",
        )

        self.late_written.set()
        self._await("late_write_done")
        torch.xpu.synchronize(device)
        torch.testing.assert_close(
            engine.tensors["bias"].cpu(),
            torch.full((6,), _LATE_VALUE, dtype=torch.float32),
            rtol=0,
            atol=0,
            msg="the daemon's later write is not visible to the engine",
        )

    def test_an_engine_can_detach_and_attach_again_while_the_daemon_lives(self):
        """An engine restart must get a fresh fd per allocation and map into a
        range of its own, with the daemon's weights unchanged."""
        first = _Engine(self.path)
        first_base = first.base
        torch.testing.assert_close(
            first.tensors["w"].cpu(),
            _expected_tensors(torch.device("cpu"))["w"][0],
            rtol=0,
            atol=0,
        )
        first.close()

        second = self._attach()
        torch.testing.assert_close(
            second.tensors["w"].cpu(),
            _expected_tensors(torch.device("cpu"))["w"][0],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            [tuple(identity) for identity in self.out["identities"]],
            second.identities(),
            "the re-attached engine mapped different memory",
        )
        self.assertIsNotNone(first_base)

    def test_an_attachment_leaks_no_descriptors(self):
        baseline = None
        for round_index in range(4):
            engine = _Engine(self.path)
            engine.close()
            if round_index == 0:
                baseline = _open_fds()  # after one warm-up round
            else:
                self.assertEqual(
                    _open_fds(), baseline, f"descriptor leak by round {round_index}"
                )

    def test_a_refused_layout_leaves_no_mapping_and_no_descriptors(self):
        """A layout the engine rejects must cost neither a mapping nor an fd;
        the fds still queued on the socket die with it."""
        before = _open_fds()
        backend = XpuVmmFdTransportBackend()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT_S)
        sock.connect(self.path)
        try:
            response = recv_msg(sock)
            # An allocation index that does not exist: the layout check must
            # refuse it before a single fd is read off the socket.
            name = next(iter(response["entries"]))
            response["entries"][name]["alloc_index"] = len(response["allocation_sizes"])
            with self.assertRaises(ValueError):
                backend.recv_fetch_state_response(sock, response)
        finally:
            sock.close()
        backend.close()
        self.assertIsNone(backend._reservation)
        self.assertEqual(_open_fds(), before)

    def test_the_engine_keeps_its_weights_after_the_daemon_exits(self):
        """The engine's own physical handle holds the dma-buf, so the weights
        stay valid; this is what lets the loader treat a daemon exit as a log
        line rather than a fatal error."""
        engine = self._attach()
        expected = _expected_tensors(torch.device("cpu"))["cache"][0]

        self.stop.set()
        self.daemon.join(_JOIN_TIMEOUT_S)
        self.assertFalse(self.daemon.is_alive())
        self.assertEqual(self.daemon.exitcode, 0, dict(self.out))

        torch.xpu.synchronize(torch.device("xpu", DEVICE_INDEX))
        torch.testing.assert_close(
            engine.tensors["cache"].cpu(), expected, rtol=0, atol=0
        )
        # A kernel over the mapped weights, not just a read back.
        total = engine.tensors["w"].sum().item()
        self.assertEqual(total, float(sum(range(24))))

    def test_for_export_probes_the_driver_before_anything_is_packed(self):
        backend = XpuVmmFdTransportBackend.for_export(DEVICE_INDEX)
        try:
            self.assertEqual(backend.device_type, "xpu")
            self.assertEqual(backend._device, torch.device("xpu", DEVICE_INDEX))
            self.assertEqual(backend._granularity % 4096, 0)
            self.assertEqual(backend.num_allocations, 0)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
