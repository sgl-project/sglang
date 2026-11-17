"""Check READ batching descriptors and fail-closed transport ownership."""

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.hixl_transport import HixlExpertTransport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Buffer:
    def __init__(self, address, stride):
        self.address, self.stride = address, stride

    def __getitem__(self, index):
        return SimpleNamespace(data_ptr=lambda: self.address + index * self.stride)


class TestHixlReadBatch(unittest.TestCase):
    def transport(self, status=0):
        transport = HixlExpertTransport.__new__(HixlExpertTransport)
        transport._engine = ctypes.c_void_p(1)
        transport._context, transport._poisoned, transport.timeout_ms = 9, False, 3000
        transport.staging = [Buffer(0x1000, 4), Buffer(0x2000, 8)]
        transport.remote = [
            (
                str(rank),
                {7: [(0xA000 + rank * 0x100, 4, 4), (0xB000 + rank * 0x100, 4, 8)]},
            )
            for rank in range(3)
        ]
        calls = []

        def read(engine, names, pointers, counts, count, timeout):
            calls.append(
                (
                    [
                        (
                            names[i].decode(),
                            tuple(pointers[i][j] for j in range(counts[i] * 3)),
                        )
                        for i in range(count)
                    ],
                    timeout,
                )
            )
            return status

        transport._lib = SimpleNamespace(sgl_hixl_read=read)
        return transport, calls

    def read(self, transport, reads):
        acl = SimpleNamespace(rt=SimpleNamespace(set_context=lambda context: 0))
        with patch.dict(sys.modules, {"acl": acl}):
            transport.read(7, reads)

    def test_multiple_peers_share_one_native_submission(self):
        transport, calls = self.transport()
        self.read(transport, [(2, 3, 1), (1, 0, 2), (2, 1, 3)])
        self.assertEqual(
            calls,
            [
                (
                    [
                        (
                            "2",
                            (
                                0x1004,
                                0xA20C,
                                4,
                                0x2008,
                                0xB218,
                                8,
                                0x100C,
                                0xA204,
                                4,
                                0x2018,
                                0xB208,
                                8,
                            ),
                        ),
                        ("1", (0x1008, 0xA100, 4, 0x2010, 0xB100, 8)),
                    ],
                    3000,
                )
            ],
        )

    def test_single_peer_preserves_rows_and_strides(self):
        transport, calls = self.transport()
        self.read(transport, [(1, 2, 0)])
        self.assertEqual(
            calls, [([("1", (0x1000, 0xA108, 4, 0x2000, 0xB110, 8))], 3000)]
        )

    def test_empty_plan_does_not_submit(self):
        transport, calls = self.transport()
        self.read(transport, [])
        self.assertEqual(calls, [])

    def test_partial_batch_failure_prevents_resource_release(self):
        transport, _ = self.transport(status=7)
        with self.assertRaisesRegex(RuntimeError, "READ failed"):
            self.read(transport, [(1, 0, 0), (2, 0, 1)])
        self.assertTrue(transport._poisoned)
        with self.assertRaisesRegex(RuntimeError, "registrations must survive"):
            transport.drain()


if __name__ == "__main__":
    unittest.main()
