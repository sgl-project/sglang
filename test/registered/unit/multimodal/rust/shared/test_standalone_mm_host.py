"""Unit tests for the standalone-process MM fallback host.

Exercises the real ``_serve`` loop and ``StandaloneMmClient`` over a temp
``ipc://`` endpoint with a stubbed ``_process``, guarding the wire protocol
(frame order, pickle round-trip, shm wrap→unwrap), the 400-parity error path,
the per-thread REQ / ROUTER routing, and the dead-child liveness branch.
"""

import asyncio
import struct
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.managers import mm_utils  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.standalone_mm_host import (  # noqa: E402
    StandaloneMmClient,
    _serve,
)
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StubHost:
    """Duck-types the two members ``_serve`` consumes: ``native_spec()`` /
    ``_native`` (handshake) and ``_process`` (per request)."""

    def __init__(self, process, native=None):
        self._process = process
        self._native = native

    def native_spec(self):
        return None if self._native is None else "spec"


def _ids_bytes(ids):
    return struct.pack(f"<{len(ids)}q", *ids)


class TestStandaloneMmHost(CustomTestCase):
    def _start(self, host):
        """Run the real serve loop on a background thread; return a connected
        client once the handshake (sent strictly after bind) arrives. Both
        sides are torn down deterministically via cleanups so no zmq context
        blocks at GC time."""
        ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        handshake = {}
        ready = threading.Event()
        writer = SimpleNamespace(send=lambda msg: (handshake.update(msg), ready.set()))
        loop = asyncio.new_event_loop()

        def run():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    _serve(host=host, ipc_name=ipc_name, pipe_writer=writer)
                )
            except RuntimeError:
                pass  # loop.stop() from the cleanup below
            finally:
                loop.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(10), "serve loop never sent its handshake")
        # LIFO: close the client's sockets first, then stop + join the loop.
        self.addCleanup(thread.join, 10)
        self.addCleanup(loop.call_soon_threadsafe, loop.stop)
        client = StandaloneMmClient(
            ipc_name=ipc_name,
            proc=SimpleNamespace(is_alive=lambda: True),
            spec=handshake["spec"],
            native=handshake["native"],
        )
        self.addCleanup(client.close)
        return client

    def test_round_trip_through_shm(self):
        feature = torch.arange(24, dtype=torch.float32).reshape(4, 6)

        async def process(rid, payload):
            mm_inputs = MultimodalProcessorOutput(
                mm_items=[
                    MultimodalDataItem(
                        modality=Modality.IMAGE, hash=123, feature=feature.clone()
                    )
                ]
            )
            return [1, 2, 3], mm_inputs, [0, 0, 1]

        client = self._start(_StubHost(process))
        # Force the non-default transport gate so the serve side really wraps
        # the feature into /dev/shm and the client materializes it back.
        with (
            get_context().override_server_args(),
            patch.object(mm_utils, "_is_default_tensor_transport", False),
        ):
            out = client.handle_sync("r1", b"payload")

        self.assertEqual(out, _ids_bytes([1, 2, 3]))
        mm_inputs, token_type_ids = client.results.pop("r1")
        self.assertEqual(token_type_ids, [0, 0, 1])
        got = mm_inputs.mm_items[0].feature
        self.assertIsInstance(got, torch.Tensor)  # materialized, not a pointer
        self.assertTrue(torch.equal(got, feature))
        self.assertEqual(mm_inputs.mm_items[0].hash, 123)

    def test_error_propagates_as_400(self):
        async def process(rid, payload):
            raise ValueError("boom")

        client = self._start(_StubHost(process))
        with self.assertRaisesRegex(ValueError, "boom"):
            client.handle_sync("r1", b"payload")
        self.assertNotIn("r1", client.results)

    def test_concurrent_callers_route_by_identity(self):
        async def process(rid, payload):
            # Interleave replies so distinct REQ identities must route back.
            await asyncio.sleep(0.05 if rid.endswith("0") else 0.0)
            return list(payload), None, None

        client = self._start(_StubHost(process))
        errors = []

        def call(i):
            rid = f"rid-{i}"
            payload = bytes([i + 1, i + 2])
            try:
                out = client.handle_sync(rid, payload)
                assert out == _ids_bytes(list(payload)), rid
                assert client.results[rid] == (None, None), rid
            except Exception as e:  # surfaced below
                errors.append(e)

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(len(client.results), 4)

    def test_socket_pool_reused_across_foreign_threads(self):
        """Bug regression: the Rust mm workers are pyo3-attached foreign
        threads whose Python thread identity is NOT sticky across calls, so a
        ``threading.local`` socket cache minted a new REQ socket per request
        and leaked fds until the scheduler hit EMFILE (ZMQError: Too many open
        files under the mm A/B benchmark). Fresh threads per call reproduce
        the non-sticky identity; sequential calls must reuse one pooled
        socket."""

        async def process(rid, payload):
            return [1], None, None

        client = self._start(_StubHost(process))
        for i in range(8):
            t = threading.Thread(target=client.handle_sync, args=(f"r{i}", b"p"))
            t.start()
            t.join(10)
        self.assertEqual(len(client.results), 8)
        self.assertEqual(client._pool.qsize(), 1)

    def test_dead_host_detected_within_poll(self):
        # Endpoint nobody serves + a dead proc: the liveness poll must raise
        # instead of hanging the Rust worker pool.
        client = StandaloneMmClient(
            ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            proc=SimpleNamespace(is_alive=lambda: False),
            spec=None,
            native=None,
        )
        self.addCleanup(client.close)
        start = time.monotonic()
        with self.assertRaisesRegex(ValueError, "died"):
            client.handle_sync("r1", b"payload")
        self.assertLess(time.monotonic() - start, 5)


if __name__ == "__main__":
    unittest.main()
