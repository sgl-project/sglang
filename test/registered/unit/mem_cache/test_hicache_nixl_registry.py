"""Unit tests for NixlRegistry acquire/release -- mocked agent, no NIXL."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.storage.nixl.nixl_registry import (
    NixlRegistry,
)
from sglang.test.test_utils import CustomTestCase

BUFFERS = [(0x1000, 4096), (0x2000, 4096)]
KEYS = ["/base/aa/key1", "/base/bb/key2"]


def make_agent(*, path_mode: bool) -> MagicMock:
    """Agent mock whose path-mode probe answers ``path_mode``.

    The probe registers a nonexistent path: a path-mode NIXL raises from
    register_memory, a pre-path-mode NIXL returns a handle. Calls after
    the probe succeed.
    """
    agent = MagicMock()
    probe_seen = {"done": False}

    def register_memory(reg_descs):
        if not probe_seen["done"]:
            probe_seen["done"] = True
            if path_mode:
                raise RuntimeError("path-mode probe: open failed")
            return MagicMock(name="probe_reg")
        return MagicMock(name="reg")

    agent.register_memory.side_effect = register_memory
    return agent


class TestPathModeLifecycle(CustomTestCase):
    """Acquire must register path-mode tuples and release must deregister
    the same handle. Guards the register/deregister pairing the async
    backend (RFC #32841) will rely on across a transfer's lifetime."""

    def setUp(self):
        self.agent = make_agent(path_mode=True)
        self.file_manager = MagicMock(use_direct_io=False)
        self.registry = NixlRegistry(self.agent, "FILE", self.file_manager)
        self.assertTrue(self.registry.path_mode)

    def test_acquire_release_pairs_registration(self):
        registration = self.registry.acquire_storage(BUFFERS, KEYS, "WRITE")

        self.assertIsNotNone(registration)
        # Registration tuples carry the path-mode spec, one devId per key.
        items, mem_type = self.agent.get_reg_descs.call_args_list[-1][0]
        self.assertEqual(mem_type, "FILE")
        self.assertEqual(
            items,
            [
                (0, 4096, 1, "rw,create:/base/aa/key1"),
                (0, 4096, 2, "rw,create:/base/bb/key2"),
            ],
        )
        # descs come from trimming the registration handle.
        self.assertIs(registration.descs, registration.reg.trim.return_value)
        # No fd path involved in path mode.
        self.file_manager.open_file.assert_not_called()

        self.registry.release_storage(registration)
        self.agent.deregister_memory.assert_called_once_with(registration.reg)

    def test_read_direction_uses_ro_spec_without_create(self):
        self.registry.acquire_storage(BUFFERS, KEYS, "READ")
        items, _ = self.agent.get_reg_descs.call_args_list[-1][0]
        self.assertEqual(items[0][3], "ro:/base/aa/key1")

    def test_register_failure_returns_none(self):
        self.agent.register_memory.side_effect = RuntimeError("no space")
        self.assertIsNone(self.registry.acquire_storage(BUFFERS, KEYS, "WRITE"))
        self.agent.deregister_memory.assert_not_called()


class TestFdModeLifecycle(CustomTestCase):
    """fd-mode ordering contract: fds must outlive the registration.

    Acquire opens fds before registering; release deregisters before
    closing. Breaking either order hands NIXL dead fds."""

    def setUp(self):
        self.agent = make_agent(path_mode=False)
        self.calls: list[str] = []
        self.agent.deregister_memory.side_effect = lambda reg: self.calls.append(
            "deregister"
        )
        self.file_manager = MagicMock(use_direct_io=False)
        fd_iter = iter([10, 11])
        self.file_manager.open_file.side_effect = lambda path, create: (
            self.calls.append(f"open:{path}"),
            next(fd_iter),
        )[1]
        self.file_manager.close_file.side_effect = lambda fd: self.calls.append(
            f"close:{fd}"
        )
        self.registry = NixlRegistry(self.agent, "FILE", self.file_manager)
        self.assertFalse(self.registry.path_mode)
        # The path-mode probe registers and deregisters once during
        # construction; drop it so tests assert on transfer calls only.
        self.calls.clear()

    def test_release_deregisters_before_closing_fds(self):
        registration = self.registry.acquire_storage(BUFFERS, KEYS, "WRITE")

        self.assertIsNotNone(registration)
        self.assertEqual(registration.fds, [10, 11])
        items, _ = self.agent.get_reg_descs.call_args_list[-1][0]
        self.assertEqual(items[0], (0, 4096, 10, "/base/aa/key1"))

        self.registry.release_storage(registration)
        self.assertEqual(
            self.calls,
            [
                "open:/base/aa/key1",
                "open:/base/bb/key2",
                "deregister",
                "close:10",
                "close:11",
            ],
        )

    def test_open_failure_closes_already_opened_fds(self):
        self.file_manager.open_file.side_effect = [10, None]
        self.assertIsNone(self.registry.acquire_storage(BUFFERS, KEYS, "WRITE"))
        self.file_manager.close_file.assert_called_once_with(10)

    def test_register_failure_closes_fds(self):
        self.agent.register_memory.side_effect = RuntimeError("no space")
        self.assertIsNone(self.registry.acquire_storage(BUFFERS, KEYS, "WRITE"))
        # Both fds closed and nothing deregistered after construction
        # (the probe's own register/deregister pair was cleared in setUp).
        self.assertEqual(
            self.calls,
            ["open:/base/aa/key1", "open:/base/bb/key2", "close:10", "close:11"],
        )


class TestObjDevIds(CustomTestCase):
    """Consecutive acquires must use disjoint devId ranges: the OBJ
    plugin's devIdToObjKey_ map is process-wide and unlocked, so reusing
    a devId across in-flight registrations corrupts the key mapping."""

    def test_devid_ranges_are_disjoint_across_acquires(self):
        agent = MagicMock()
        registry = NixlRegistry(agent, "OBJ", None)

        registry.acquire_storage(BUFFERS, KEYS, "WRITE")
        first_items, _ = agent.get_reg_descs.call_args_list[-1][0]
        registry.acquire_storage(BUFFERS, KEYS, "WRITE")
        second_items, _ = agent.get_reg_descs.call_args_list[-1][0]

        first_ids = {item[2] for item in first_items}
        second_ids = {item[2] for item in second_items}
        self.assertEqual(len(first_ids & second_ids), 0)


class TestStorageContextManager(CustomTestCase):
    """storage() is a thin acquire/release wrapper and must keep its
    original contract: yield descs (or None) and always release."""

    def test_yields_descs_and_releases_on_exit(self):
        agent = make_agent(path_mode=True)
        registry = NixlRegistry(agent, "FILE", MagicMock(use_direct_io=False))

        with registry.storage(BUFFERS, KEYS, "WRITE") as descs:
            self.assertIsNotNone(descs)
            agent.deregister_memory.assert_not_called()
        agent.deregister_memory.assert_called_once()

    def test_yields_none_without_release_when_acquire_fails(self):
        agent = make_agent(path_mode=True)
        registry = NixlRegistry(agent, "FILE", MagicMock(use_direct_io=False))
        agent.register_memory.side_effect = RuntimeError("no space")

        with registry.storage(BUFFERS, KEYS, "WRITE") as descs:
            self.assertIsNone(descs)
        agent.deregister_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
