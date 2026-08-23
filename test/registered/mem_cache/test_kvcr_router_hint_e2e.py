"""Two-instance KVCR e2e over the SGLang router-hint contract.

This exercises the ONE integration seam that Workstream B owns: a dynamo router
hint, parsed by our :class:`RouterHint` and matched by our
:class:`StrKeyHintAdapter`, must drive the real ``nvidia-kvcr`` core to fetch a
remote prefix from a peer instance. We stand up two real ``KVCR`` instances
(target + source) wired to in-process fakes for NIXL, the ZMQ control channel,
and primary pinning -- the same fake seams the core's own unit tests use -- and
assert that:

  1. ``target.submit_hint(...)`` + ``target.query(keys)`` reports the
     hint-covered keys as FETCHABLE (our adapter's ``matches`` is the gate), and
  2. ``target.deliver(...)`` makes the source issue a real NIXL WRITE of the
     covered blocks and both sides converge to a SUCCESS op result.

It is CPU-only and needs no torch / GPU / real NIXL, so it runs on the cheapest
CI tier. It is skipped if the ``kvcr`` wheel is not installed. This does NOT
cover the ``KVCRStore`` host-pool wiring (that needs a real HostKVCache); it
covers the hint -> core contract that the store's ``_build_kvcr`` depends on.

    python -m pytest test/registered/mem_cache/test_kvcr_router_hint_e2e.py -v
"""

from __future__ import annotations

import ctypes
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Collection, Mapping
from unittest.mock import patch

try:
    from kvcr import KVCR, KVCRBindings
    from kvcr import progress as kvcr_progress
    from kvcr.config import (
        KVCRBackendConfigs,
        KVCRConfig,
        LocalDramInput,
        RemoteFWDramOptions,
    )
    from kvcr.types import (
        BlockKey,
        CacheTier,
        MemDescriptor,
        OpEntryResult,
        OpEntryStatus,
        PinHandle,
        PinRequestId,
        QueryStatus,
    )

    _HAS_KVCR = True
except ImportError:  # pragma: no cover - wheel not installed on this tier
    _HAS_KVCR = False

if _HAS_KVCR:
    from sglang.srt.mem_cache.storage.kvcr.pin_adapter import NoFrameworkPinning
    from sglang.srt.mem_cache.storage.kvcr.router_hint import (
        RouterHint,
        StrKeyHintAdapter,
    )

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Minimal in-process fakes, mirroring nvidia-kvcr's own tests/unit/test_kvcr.py.
# Vendored (not imported) because the core's test module ships inside the wheel's
# source tree, not as an importable package.
# ---------------------------------------------------------------------------


def _mem_descriptor(addr: int = 128, size: int = 16) -> MemDescriptor:
    return MemDescriptor(
        end_point_name="primary",
        mem_type="DRAM",
        addr=addr,
        size=size,
        device_Id=0,
        info="",
    )


class FakePrimaryPinning:
    """The KVCR-to-framework pin triple over an in-memory descriptor table.

    ``request_pin`` is asynchronous by contract -- it returns an id and the
    result is handed back on a later ``poll_pin_results`` -- so this fake
    answers on the very next poll. ``missing_indices`` marks positions the
    source does NOT hold, so we can drive partial-prefix delivery.
    """

    def __init__(self, missing_indices: Collection[int] = ()):
        self.searches: list[tuple[BlockKey, ...]] = []
        self.releases: list[PinHandle] = []
        self.missing_indices = set(missing_indices)
        self._next_request = 0
        self._completed: list = []

    def request_pin(self, keys) -> PinRequestId:
        keys = tuple(keys)
        self.searches.append(keys)
        request = PinRequestId(self._next_request)
        self._next_request += 1
        self._completed.append(
            (
                request,
                (
                    "pin",
                    {
                        key: (
                            None
                            if index in self.missing_indices
                            else _mem_descriptor(addr=0)
                        )
                        for index, key in enumerate(keys)
                    },
                ),
            )
        )
        return request

    def poll_pin_results(self):
        completed = self._completed
        self._completed = []
        return completed

    def release_pin(self, pin_handle: PinHandle) -> bool:
        self.releases.append(pin_handle)
        return pin_handle == "pin"


class FakeBytesControl:
    """In-process stand-in for ZmqPeerControlChannel: send() appends, recv() drains."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.sent: list[tuple[str, bytes]] = []
        self.incoming: list[bytes] = []

    def send(self, endpoint: str, message: bytes) -> bool:
        self.sent.append((endpoint, message))
        return True

    def recv(self) -> list[bytes]:
        incoming = self.incoming
        self.incoming = []
        return incoming


class FakeNixlAgent:
    """NIXL agent whose transfer() performs a real ctypes.memmove for local WRITEs."""

    def __init__(self, metadata: bytes = b"metadata"):
        self.name = ""
        self.metadata = metadata
        self.registrations: list[Any] = []
        self.remote_agents: list[bytes] = []
        self.xfers: list[tuple[Any, ...]] = []
        self.transfers: list[int] = []
        self.sent_notifs: list[tuple[bytes, bytes]] = []
        self.released_xfers: list[int] = []
        self.removed_agents: list[bytes] = []
        self.notifs: dict[str, list[bytes]] = {}
        self.state = "PROC"

    def register_memory(self, descs, mem_type="DRAM"):
        self.registrations.append((list(descs), mem_type))
        return len(self.registrations)

    def deregister_memory(self, handle):
        pass

    def get_agent_metadata(self) -> bytes:
        return self.metadata

    def add_remote_agent(self, metadata: bytes) -> bytes:
        self.remote_agents.append(metadata)
        return f"remote-{len(self.remote_agents)}".encode()

    def get_xfer_descs(self, descs, mem_type="DRAM"):
        return list(descs)

    def initialize_xfer(
        self, op, local_descs, remote_descs, remote_agent, notif_msg=b"", backends=None
    ):
        local_descs = list(local_descs)
        remote_descs = list(remote_descs)
        self.xfers.append(
            (
                op,
                local_descs,
                list(range(len(local_descs))),
                remote_descs,
                remote_agent,
                notif_msg,
            )
        )
        return len(self.xfers)

    def transfer(self, handle):
        self.transfers.append(handle)
        op, local_descs, local_indices, remote_descs, remote_agent, _ = self.xfers[
            handle - 1
        ]
        if op == "WRITE" and remote_agent == self.name:
            for index in local_indices:
                src_addr, src_size, _ = local_descs[index]
                dst_addr, dst_size, _ = remote_descs[index]
                ctypes.memmove(dst_addr, src_addr, min(src_size, dst_size))
        return "PROC"

    def check_xfer_state(self, handle):
        return self.state

    def get_xfer_telemetry(self, handle):
        return SimpleNamespace(xferDuration=2000, postDuration=500, totalBytes=32)

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def remove_remote_agent(self, agent_name):
        self.removed_agents.append(agent_name)

    def send_notif(self, agent_name, notif_msg):
        self.sent_notifs.append((agent_name, notif_msg))

    def get_new_notifs(self):
        notifs = self.notifs
        self.notifs = {}
        return notifs


@contextmanager
def _use_nixl_agent(agent):
    def create_agent(name, *_):
        agent.name = name
        return agent

    with patch.multiple(
        kvcr_progress,
        nixl_agent=create_agent,
        nixl_agent_config=lambda **kwargs: kwargs,
    ):
        yield


def _poll_until(kvcr, predicate, *, timeout: float = 2):
    completed: list = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completed.extend(kvcr.poll_completed())
        if predicate(completed):
            return completed
        time.sleep(0.001)
    raise AssertionError("KVCR progress condition was not reached")


def _has_outstanding_operations(kvcr) -> bool:
    """Whether the core still owns an in-flight op.

    Replaces the ``has_pending_work()`` the facade used to expose; the count is
    now core-internal, and the core's own tests read it the same way.
    """
    return kvcr._core._outstanding_operations > 0


def _wait_until(predicate, *, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("KVCR control condition was not reached")


def _op_entries(entries: Mapping[BlockKey, bool]) -> dict:
    return {
        key: OpEntryResult(OpEntryStatus.SUCCESS if ok else OpEntryStatus.FAILED)
        for key, ok in entries.items()
    }


def _new_kvcr(
    agent,
    pinning,
    control,
    name,
    *,
    remote_options=None,
    key_hint_adapter=None,
    local_dram=None,
):
    config = replace(
        KVCRConfig(nixl_agent_name=name, inventory_report_interval_ms=0),
        nixl_agent_name=name,
        nixl_listen_port=1,
    )
    with _use_nixl_agent(agent):
        return KVCR(
            config,
            KVCRBindings(
                request_pin=pinning.request_pin,
                poll_pin_results=pinning.poll_pin_results,
                release_pin=pinning.release_pin,
                framework_control=control,
                key_hint_adapter=key_hint_adapter,
            ),
            KVCRBackendConfigs(
                remote_fw_dram=remote_options or RemoteFWDramOptions(),
                local_dram=local_dram,
            ),
        )


@unittest.skipUnless(_HAS_KVCR, "nvidia-kvcr wheel not installed")
class TestKVCRRouterHintE2E(unittest.TestCase):
    """Our RouterHint + StrKeyHintAdapter drive a real two-instance KVCR fetch."""

    def setUp(self):
        self._open: list[KVCR] = []

    def tearDown(self):
        while self._open:
            self._open.pop().close()

    def _pair(
        self,
        *,
        missing_indices=(),
        eager_ctrl_connect=False,
        source_pinning=None,
        source_local_dram=None,
    ):
        target_agent = FakeNixlAgent(metadata=b"target-md")
        source_agent = FakeNixlAgent(metadata=b"source-md")
        target_control = FakeBytesControl("tcp://target:1")
        source_control = FakeBytesControl("tcp://source:1")
        target = _new_kvcr(
            target_agent,
            FakePrimaryPinning(),
            target_control,
            "target",
            remote_options=RemoteFWDramOptions(eager_ctrl_connect=eager_ctrl_connect),
            # The load-bearing bit: the target matches queried keys against our
            # RouterHint via our real StrKeyHintAdapter.
            key_hint_adapter=StrKeyHintAdapter(),
        )
        source = _new_kvcr(
            source_agent,
            source_pinning or FakePrimaryPinning(missing_indices=missing_indices),
            source_control,
            "source",
            local_dram=source_local_dram,
        )
        source._core._clock = lambda: 0.0
        self._open += [target, source]
        return SimpleNamespace(
            target=target,
            source=source,
            target_agent=target_agent,
            source_agent=source_agent,
            target_control=target_control,
            source_control=source_control,
        )

    @staticmethod
    def _pump_control(from_control, to_control):
        to_control.incoming.extend(message for _, message in from_control.sent)
        from_control.sent = []

    def _run_transfer(self, *, missing_indices=(), completed_count=2, eager=False):
        env = self._pair(missing_indices=missing_indices, eager_ctrl_connect=eager)
        # Block-hash strings as they arrive from the dynamo router hint; these are
        # exactly what StrKeyHintAdapter.encode maps to BlockKeys.
        hash_strs = ["k0", "k1"]
        keys = tuple(BlockKey(h.encode("utf-8")) for h in hash_strs)

        hint = RouterHint(
            source_control_endpoint="tcp://source:1",
            block_hashes=hash_strs,
        )
        env.target.submit_hint((), src="tcp://source:1", request_id="req", hints=hint)

        # (1) query gate: our adapter.matches decides FETCHABLE vs MISS.
        self.assertEqual(
            env.target.query(keys, "req"),
            [
                (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
                (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2),
            ],
        )

        if eager:
            _wait_until(lambda: len(env.target_control.sent) == 1)
            self._pump_control(env.target_control, env.source_control)

        # (2) deliver -> target emits start_write on the control channel.
        op_handle = env.target.deliver(
            {key: _mem_descriptor(addr=8192 + i * 16) for i, key in enumerate(keys)},
            request_id="req",
        )
        _wait_until(lambda: bool(env.target_control.sent))
        self._pump_control(env.target_control, env.source_control)

        # Source pins, issues the NIXL WRITE for the covered prefix, notifies.
        self.assertEqual(
            _poll_until(env.source, lambda _: bool(env.source_agent.xfers)), []
        )
        self.assertEqual(env.source_agent.xfers[0][2], list(range(completed_count)))
        notification = env.source_agent.xfers[0][5]
        self.assertIsNotNone(notification)

        env.source_agent.state = "DONE"
        self.assertEqual(
            _poll_until(
                env.source, lambda _: not _has_outstanding_operations(env.source)
            ),
            [],
        )

        # Target consumes the write_done notification and resolves the op.
        env.target_agent.notifs["source"] = [notification]
        completed = _poll_until(env.target, lambda done: bool(done))
        self.assertEqual(
            completed,
            [
                (
                    op_handle,
                    _op_entries(
                        {key: i < completed_count for i, key in enumerate(keys)}
                    ),
                )
            ],
        )
        return env

    def test_full_prefix_delivered_from_hint(self):
        """Both hint-covered blocks are fetched from the source peer."""
        self._run_transfer(missing_indices=(), completed_count=2)

    def test_eager_ctrl_connect_prefix_delivered(self):
        """Same, with eager control-channel connect (pre-deliver handshake)."""
        self._run_transfer(missing_indices=(), completed_count=2, eager=True)

    def test_partial_prefix_when_source_missing_tail(self):
        """A tail block the source lacks yields a partial (k0 SUCCESS, k1 FAILED)."""
        self._run_transfer(missing_indices=(1,), completed_count=1)

    def test_evicted_block_is_not_served_out_of_the_deposited_host_page(self):
        """After eviction the source declines rather than reading host memory.

        Regression guard. The backend used to answer ``request_pin`` out of a
        dict of the ``HostKVCache`` descriptors it had handed to ``deposit``,
        and that dict outlived KVCR's own residency: once the local tier
        evicted a key, KVCR fell through to the framework callback and got the
        *host page* back. Nothing on that path errors -- block keys are token
        hashes with no content check -- so a peer would read whatever HiCache
        had since refilled that page into and decode from it.

        The eviction is the load-bearing part: while a key is still resident,
        ``_claim_local_dram_sources`` answers first and both the old and the
        new adapter serve the same slot address. Only the evicted key
        distinguishes them.

        One slot, two deposits: k1 evicts k0. The delivery then asks for
        (k1, k0), so the correct answer is a single-descriptor write covering
        k1 alone, addressed inside KVCR's slot region.
        """
        block_size = 16
        source_slots = ctypes.create_string_buffer(block_size)
        host_pages = [ctypes.create_string_buffer(block_size) for _ in range(2)]
        slots_address = ctypes.addressof(source_slots)
        slots_end = slots_address + len(source_slots)

        env = self._pair(
            source_pinning=NoFrameworkPinning(),
            source_local_dram=LocalDramInput(slots_address, block_size, 1),
        )
        evicted, resident = BlockKey(b"k0"), BlockKey(b"k1")

        for key, page in zip((evicted, resident), host_pages):
            deposit = env.source.deposit(
                {key: _mem_descriptor(addr=ctypes.addressof(page), size=block_size)}
            )
            _wait_until(lambda: bool(env.source_agent.transfers))
            env.source_agent.state = "DONE"
            self.assertEqual(
                _poll_until(env.source, lambda done: bool(done)),
                [(deposit, _op_entries({key: True}))],
            )
            env.source_agent.state = "PROC"
            env.source_agent.transfers.clear()
        env.source_agent.xfers.clear()
        self.assertEqual(
            env.source.query((evicted, resident)),
            [(QueryStatus.MISS, None), (QueryStatus.HIT, CacheTier.LOCAL_G2)],
        )

        hint = RouterHint(
            source_control_endpoint="tcp://source:1",
            block_hashes=["k1", "k0"],
        )
        env.target.submit_hint((), src="tcp://source:1", request_id="req", hints=hint)
        env.target.deliver(
            {
                resident: _mem_descriptor(addr=8192),
                evicted: _mem_descriptor(addr=8208),
            },
            request_id="req",
        )
        _wait_until(lambda: bool(env.target_control.sent))
        self._pump_control(env.target_control, env.source_control)
        self.assertEqual(
            _poll_until(env.source, lambda _: bool(env.source_agent.xfers)), []
        )

        source_descriptors = env.source_agent.xfers[0][1]
        self.assertEqual(
            len(source_descriptors),
            1,
            "the evicted key was served; the source is reading framework memory",
        )
        served_address = source_descriptors[0][0]
        self.assertTrue(
            slots_address <= served_address < slots_end,
            f"source served {served_address:#x}, outside KVCR's slot region "
            f"[{slots_address:#x}, {slots_end:#x})",
        )

    def test_query_misses_keys_outside_hint(self):
        """A key not in the hint's block_hashes is not FETCHABLE (adapter gate)."""
        env = self._pair()
        hint = RouterHint(source_control_endpoint="tcp://source:1", block_hashes=["k0"])
        env.target.submit_hint((), src="tcp://source:1", request_id="req", hints=hint)
        covered = BlockKey(b"k0")
        uncovered = BlockKey(b"k9")
        self.assertEqual(
            env.target.query((covered, uncovered), "req"),
            [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2), (QueryStatus.MISS, None)],
        )


if __name__ == "__main__":
    unittest.main()
